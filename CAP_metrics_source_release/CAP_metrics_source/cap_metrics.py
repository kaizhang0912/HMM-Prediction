"""Compute OCR, dwell time, transition probability, and trajectory entropy.

Input: {subject_id: {run_id: [zero-based state labels]}}.
Run boundaries are respected for dwell times and transitions.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

LOG = logging.getLogger(__name__)


def validate_states(data, k_clusters):
    if type(k_clusters) is not int or k_clusters < 1:
        raise ValueError("k_clusters must be a positive integer.")
    if not isinstance(data, dict):
        raise TypeError("State data must be a subject-to-runs dictionary.")
    output = {}
    for subject, runs in data.items():
        if not isinstance(subject, str) or not isinstance(runs, dict):
            raise TypeError("Subject IDs must be strings and each subject must map to a run dictionary.")
        output[subject] = {}
        for run, labels in runs.items():
            array = np.asarray(labels)
            if array.ndim != 1:
                raise ValueError(f"State labels must be one-dimensional: {subject}/{run}")
            if array.size:
                if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                    raise ValueError(f"State labels must be numeric integers: {subject}/{run}")
                if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
                    raise ValueError(f"Invalid state labels: {subject}/{run}")
                if np.any(array < 0) or np.any(array >= k_clusters):
                    raise ValueError(f"State labels outside 0..{k_clusters - 1}: {subject}/{run}")
            output[subject][run] = array.astype(int).tolist()
    return output


def compute_subject_ocr(state_index_dict, k_clusters=8):
    """Frame-weighted occupancy across runs; empty participants receive NaN."""
    data = validate_states(state_index_dict, k_clusters)
    rows = []
    for subject, runs in data.items():
        labels = [state for sequence in runs.values() for state in sequence]
        values = np.full(k_clusters, np.nan)
        if labels:
            values = np.bincount(labels, minlength=k_clusters) / len(labels)
        rows.append([subject, *values])
    return pd.DataFrame(rows, columns=["Subject"] + [f"State{s}_prop" for s in range(k_clusters)])


def compute_dwell_times(state_index_dict, k_clusters=8):
    """Contiguous visit durations in frames; visits never span run boundaries."""
    data = validate_states(state_index_dict, k_clusters)
    output = {subject: {state: [] for state in range(k_clusters)} for subject in data}
    for subject, runs in data.items():
        for sequence in runs.values():
            if not sequence:
                continue
            array = np.asarray(sequence)
            boundaries = np.r_[0, np.flatnonzero(np.diff(array)) + 1, len(array)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                output[subject][int(array[start])].append(int(end - start))
    return output


def dwell_table(dwell_times_dict, k_clusters=8, tr_seconds=None):
    """Participant means over all visits; an unvisited state receives NaN."""
    unit, factor = ("frames", 1.0) if tr_seconds is None else ("seconds", tr_seconds)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("TR must be a finite positive number.")
    rows = []
    for subject, states in dwell_times_dict.items():
        values = [np.mean(states.get(s, [])) * factor if states.get(s, []) else np.nan
                  for s in range(k_clusters)]
        rows.append([subject, *values])
    return pd.DataFrame(rows, columns=["Subject"] + [f"State{s}_DT_{unit}" for s in range(k_clusters)])


def summarize_dwell(dwell_times_dict, k_clusters=8):
    """Equal-participant summary of mean dwell durations (frames; SD ddof=0)."""
    table = dwell_table(dwell_times_dict, k_clusters)
    rows = []
    for state in range(k_clusters):
        values = table[f"State{state}_DT_frames"].dropna().to_numpy()
        rows.append({"State": state, "Mean_DT_frames": values.mean() if len(values) else np.nan,
                     "SD_DT_frames": values.std(ddof=0) if len(values) else np.nan, "N": len(values)})
    return pd.DataFrame(rows)


def transition_counts(state_index_dict, k_clusters=8):
    """Count adjacent-frame transitions, including self-transitions, within runs."""
    data = validate_states(state_index_dict, k_clusters)
    output = {}
    for subject, runs in data.items():
        counts = np.zeros((k_clusters, k_clusters), dtype=np.int64)
        for labels in runs.values():
            if len(labels) > 1:
                np.add.at(counts, (labels[:-1], labels[1:]), 1)
        output[subject] = counts
    return output


def compute_transition_matrices(state_index_dict, k_clusters=8):
    """Row-normalized counts; rows without observed departures are zero-filled.

Zero-filled rows are placeholders, not estimated probability distributions.
They are explicitly flagged in the exported QC table and rejected by EMT.
"""
    output = {}
    for subject, counts in transition_counts(state_index_dict, k_clusters).items():
        totals = counts.sum(axis=1, keepdims=True)
        output[subject] = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    return output


def validate_transition_matrix(P):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or not P.shape[0]:
        raise ValueError("Transition matrix must be nonempty and square.")
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("Transition matrix contains nonfinite or negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Transition matrix has unobserved or non-stochastic rows.")
    return P


def local_entropy(P):
    """Row entropy in bits, using 0 * log2(0) = 0."""
    P = validate_transition_matrix(P)
    terms = np.zeros_like(P)
    positive = P > 0
    terms[positive] = P[positive] * np.log2(P[positive])
    return -terms.sum(axis=1, keepdims=True)


def stationary_distribution(P):
    """Unique stationary distribution for a finite irreducible Markov chain."""
    P = validate_transition_matrix(P)
    n = len(P)
    if connected_components((P > 0).astype(int), directed=True, connection="strong", return_labels=False) != 1:
        raise ValueError("Reducible transition matrix: the supplied full-state EMT formula is not applied.")
    system = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    target = np.r_[np.zeros(n), 1.0]
    mu = np.linalg.lstsq(system, target, rcond=None)[0]
    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("Stationary probabilities must be strictly positive for this EMT formula.")
    mu /= mu.sum()
    if not np.allclose(mu @ P, mu, atol=1e-10, rtol=1e-8):
        raise ValueError("Stationary-distribution residual is too large.")
    return mu


def trajectory_entropy(P):
    """Original trajectory-entropy formula, with validated stationary solution.

Returns an n-state by n-state matrix in bits, including the original diagonal
convention H[i,i] = entropy_rate / stationary_probability[i].
"""
    P = validate_transition_matrix(P)
    n = len(P)
    mu = stationary_distribution(P)
    A = np.tile(mu, (n, 1))
    h = local_entropy(P)
    H_star = np.tile(h, (1, n))
    entropy_rate = float(mu @ h[:, 0])
    H_delta = np.diag(entropy_rate / mu)
    system = np.eye(n) - P + A
    K = np.linalg.solve(system, H_star - H_delta)
    H = K - np.tile(np.diag(K), (n, 1)) + H_delta
    if not np.isfinite(H).all():
        raise ValueError("Nonfinite trajectory entropy.")
    if np.any(H < -1e-8):
        raise ValueError("Negative trajectory entropy exceeds numerical tolerance.")
    return H


def matrix_table(matrices, ids, k_clusters):
    """Row-major layout: From0_To0, From0_To1, ..., FromK_ToK."""
    columns = [f"From{i}_To{j}" for i in range(k_clusters) for j in range(k_clusters)]
    values = np.stack([matrices[subject] for subject in ids]).reshape(len(ids), -1)
    table = pd.DataFrame(values, columns=columns)
    table.insert(0, "Subject", ids)
    return table


def export_metrics(state_index_dict, output_dir, k_clusters=8, tr_seconds=2.0, strict_entropy=False):
    data = validate_states(state_index_dict, k_clusters)
    if not data:
        raise ValueError("No participants supplied.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ids = list(data)
    ocr = compute_subject_ocr(data, k_clusters)
    dwell = compute_dwell_times(data, k_clusters)
    counts = transition_counts(data, k_clusters)
    transitions = compute_transition_matrices(data, k_clusters)
    entropy, qc = {}, []
    for subject in ids:
        reason = ""
        try:
            entropy[subject] = trajectory_entropy(transitions[subject])
        except (ValueError, np.linalg.LinAlgError) as error:
            if strict_entropy:
                raise ValueError(f"EMT failed for {subject}: {error}") from error
            reason = str(error)
            entropy[subject] = np.full((k_clusters, k_clusters), np.nan)
            LOG.warning("EMT unavailable for %s: %s", subject, reason)
        qc.append({"Subject": subject, "Total_frames": sum(map(len, data[subject].values())),
                   "Nonempty_runs": sum(bool(x) for x in data[subject].values()),
                   "Transition_count": int(counts[subject].sum()),
                   "Unobserved_departure_states": ";".join(map(str, np.flatnonzero(counts[subject].sum(axis=1) == 0))),
                   "EMT_valid": not bool(reason), "EMT_reason": reason})
    ocr.to_csv(root / "OCR.csv", index=False)
    dwell_table(dwell, k_clusters).to_csv(root / "DT_frames.csv", index=False)
    dwell_table(dwell, k_clusters, tr_seconds).to_csv(root / "DT_seconds.csv", index=False)
    summarize_dwell(dwell, k_clusters).to_csv(root / "DT_group_summary.csv", index=False)
    matrix_table(counts, ids, k_clusters).to_csv(root / "transition_counts.csv", index=False)
    matrix_table(transitions, ids, k_clusters).to_csv(root / "TP.csv", index=False)
    matrix_table(entropy, ids, k_clusters).to_csv(root / "EMT.csv", index=False)
    means = np.asarray([entropy[s].mean() for s in ids])
    pd.DataFrame({"Subject": ids, "Mean_EMT_bits": means}).to_csv(root / "EMT_mean.csv", index=False)
    pd.DataFrame(qc).to_csv(root / "QC.csv", index=False)
    np.savez_compressed(root / "metrics.npz", subject_ids=np.asarray(ids, dtype=str),
                        OCR=ocr.iloc[:, 1:].to_numpy(), DT_frames=dwell_table(dwell, k_clusters).iloc[:, 1:].to_numpy(),
                        TP=np.stack([transitions[s] for s in ids]), EMT=np.stack([entropy[s] for s in ids]),
                        EMT_mean=means, k_states=k_clusters, tr_seconds=tr_seconds)
    (root / "dwell_visits.json").write_text(json.dumps(dwell, indent=2))
    metadata = {"k_states": k_clusters, "tr_seconds": tr_seconds, "state_labels": "zero_based",
                "OCR": "pooled_frame_fraction", "DT": "pooled_visit_mean_within_participant",
                "TP": "pooled_within_run_adjacent_frame_counts_row_normalized_including_self_transitions",
                "TP_zero_row": "zero_placeholder_flagged_in_QC",
                "EMT": "original_matrix_formula_with_validated_unique_positive_stationary_distribution",
                "EMT_units": "bits", "EMT_mean": "mean_of_all_matrix_entries_including_diagonal",
                "invalid_EMT": "error" if strict_entropy else "all_NaN_matrix_with_QC_reason"}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"OCR": ocr, "dwell_visits": dwell, "TP": transitions, "EMT": entropy, "QC": pd.DataFrame(qc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON subject/run/state dictionary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--strict-entropy", action="store_true", help="Stop rather than flag invalid EMT")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = json.loads(args.input.read_text())
    export_metrics(data, args.output, args.states, args.tr, args.strict_entropy)


if __name__ == "__main__":
    main()
