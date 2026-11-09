"""Fold-specific HMM -> statewise FC -> repeated cross-validated rCCA.

Run with --stage hmm, inspect training maps and edit state_mapping.json,
then run with --stage rcca. No CovBat or permutation inference is implemented.
"""
import argparse
import hashlib
import json
import logging
import pickle
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
STATES = ("State2", "State4")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def save_pickle(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def normalize_id(value):
    value = str(value).strip()
    return value if value.startswith("sub-") else "sub-" + value


def normalize_session_id(value):
    if value.startswith(("sub-PCS", "sub-CCS")) and value[-1].isalpha():
        return value[:-1]
    return value


def load_timeseries(path):
    # Load only trusted local NumPy dictionaries (pickle-backed input).
    raw = np.load(path, allow_pickle=True).item()
    output = {}
    for subject, tasks in raw.items():
        subject = normalize_session_id(subject)
        if subject in output:
            raise ValueError("Duplicate subject after session-ID normalization.")
        output[subject] = tasks
    return output


def motion_mask(fd, threshold, expansion):
    """Preserve the original sequential, in-place expansion algorithm."""
    mask = np.isnan(fd) | (fd > threshold)
    for shift in range(1, expansion + 1):
        mask[:-shift] |= mask[shift:]
        mask[shift:] |= mask[:-shift]
    return mask


def load_cleaned_ts(data, confounds_template, phase, group, cfg):
    cleaned, records = {}, []
    for subject, tasks in data.items():
        cleaned[subject] = {}
        session_id = subject
        if subject.startswith(("sub-PCS", "sub-CCS")):
            session_id += "A" if phase == "pre" else "C"
        for task, series in tasks.items():
            path = Path(confounds_template.format(subject=subject, session_id=session_id,
                                                 task=task, phase=phase, group=group))
            if not path.is_file():
                LOG.warning("Missing confounds: %s/%s/%s/%s", group, phase, subject, task)
                continue
            confounds = pd.read_csv(path, sep="\t", low_memory=False)
            fd = confounds["framewise_displacement"].to_numpy()[cfg["skip_fd_tr"]:]
            series = np.asarray(series, dtype=float)
            if series.ndim != 2 or len(fd) != len(series) or not len(series):
                LOG.warning("Invalid shape or FD length: %s/%s/%s/%s", group, phase, subject, task)
                continue
            mask = motion_mask(fd, cfg["fd_threshold"], cfg["expand_frames"])
            removed = np.flatnonzero(mask)
            cleaned[subject][task] = {"ts": series[~mask], "removed_idx": removed.tolist()}
            records.append({"Subject": subject, "Group": group, "Phase": phase, "Task": task,
                            "Total_TRs": len(series), "Removed_TRs": len(removed),
                            "Removed_Ratio": round(len(removed) / len(series), 4)})
    return cleaned, records


def interpolate_volumes(array, sample_mask, tr):
    """Cubic interpolation; clamp leading/trailing outliers to retained volumes."""
    from nilearn import signal

    array = np.asarray(array, dtype=float).copy()
    retained = np.flatnonzero(sample_mask)
    if len(retained) < 2:
        raise ValueError("Interpolation requires at least two retained volumes.")
    if sample_mask.all():
        return array
    result = signal._interpolate_volumes(array, sample_mask=sample_mask, t_r=tr, extrapolate=True)
    first, last = retained[0], retained[-1]
    result[:first] = result[first]
    result[last + 1:] = result[last]
    return result


def prepare_data(cfg, root):
    prepared, records, lengths = {}, [], []
    for group, spec in cfg["datasets"].items():
        prepared[group] = {}
        excluded = set(cfg["excluded_subjects"].get(group, []))
        for phase in ("pre", "post"):
            original = load_timeseries(spec[phase])
            cleaned, rows = load_cleaned_ts(original, spec["confounds_template"], phase, group, cfg)
            records.extend(rows)
            output = {}
            for subject, tasks in cleaned.items():
                if subject in excluded:
                    continue
                output[subject] = {}
                for task, entry in tasks.items():
                    series = np.asarray(original[subject][task], dtype=float)
                    mask = np.ones(len(series), dtype=bool)
                    mask[entry["removed_idx"]] = False
                    result = interpolate_volumes(series, mask, cfg["tr_seconds"])
                    output[subject][task] = result
                    lengths.append({"Subject": subject, "Group": group, "Phase": phase,
                                    "Task": task, "Original_TRs": len(series),
                                    "Interpolated_TRs": len(result), "Match": len(result) == len(series)})
            prepared[group][phase] = output
    report = pd.DataFrame(records)
    report.to_csv(root / "motion_by_run.csv", index=False)
    if not report.empty:
        summary = report.groupby(["Group", "Subject"])[["Total_TRs", "Removed_TRs"]].sum()
        summary["Removed_Ratio"] = summary["Removed_TRs"] / summary["Total_TRs"]
        summary.to_csv(root / "motion_by_subject.csv")
        # Descriptive flag only; exclusions come exclusively from the configuration.
        summary[summary["Removed_Ratio"] > cfg["motion_report_threshold"]].to_csv(root / "motion_flags.csv")
    pd.DataFrame(lengths).to_csv(root / "length_check.csv", index=False)
    return prepared["PT"]["pre"], prepared["PT"]["post"]


def load_clinical(cfg, pre, post):
    table = pd.read_csv(cfg["clinical_file"], dtype=str)
    id_column = table.columns[0]
    columns = cfg.get("clinical_columns") or table.columns[1:9].tolist()
    if table[id_column].isna().any() or table[id_column].str.strip().eq("").any():
        raise ValueError("Missing clinical subject ID.")
    table[id_column] = table[id_column].map(normalize_id)
    if table[id_column].duplicated().any():
        raise ValueError("Duplicate clinical subject IDs.")
    table = table.set_index(id_column)
    table[columns] = table[columns].apply(pd.to_numeric, errors="raise")
    ids = [subject for subject in pre if subject in post]
    if len(ids) < cfg["n_splits"]:
        raise ValueError("Fewer participants than CV folds.")
    missing = set(ids) - set(table.index)
    if missing:
        raise ValueError(f"Missing clinical records: {sorted(missing)}")
    if not np.isfinite(table.loc[ids, columns].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite clinical values.")
    anchor = cfg.get("anchor_column") or columns[0]
    if anchor not in columns:
        raise ValueError("Anchor must be one of the clinical columns.")
    return table, columns, anchor, ids


def data_digest(pre, post, table, columns, ids):
    """Fingerprint actual analysis inputs, including run order and clinical scores."""
    digest = hashlib.sha256()
    for phase, data in (("pre", pre), ("post", post)):
        for subject in ids:
            for task, array in data[subject].items():
                array = np.ascontiguousarray(array, dtype=np.float64)
                digest.update(json.dumps([phase, subject, task, array.shape]).encode())
                digest.update(array.tobytes())
    digest.update(np.ascontiguousarray(table.loc[ids, columns], dtype=np.float64).tobytes())
    return digest.hexdigest()


def initialize_plan(root, cfg, ids, columns, anchor, digest):
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "nilearn", "hmmlearn", "cca-zoo")
    manifest = {"configuration": cfg, "ids": ids, "clinical_columns": columns,
                "anchor_column": anchor, "input_sha256": digest,
                "versions": {name: version(name) for name in packages},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "harmonized": False}
    path = root / "analysis_manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Inputs, configuration, code, or versions changed. Use a new output directory.")
    else:
        if any(root.glob("rep*_fold*")) or (root / "cv_plan.json").exists():
            raise ValueError("Existing outputs lack a matching manifest. Use a new output directory.")
        write_json(path, manifest)
    plan = []
    for repeat in range(cfg["n_repeats"]):
        cv = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["split_seed"] + repeat)
        for fold, (train, test) in enumerate(cv.split(ids)):
            plan.append({"key": f"rep{repeat + 1:02d}_fold{fold + 1:02d}",
                         "repeat": repeat + 1, "fold": fold + 1,
                         "train_idx": train.tolist(), "test_idx": test.tolist(),
                         "train_ids": [ids[i] for i in train], "test_ids": [ids[i] for i in test]})
    plan_path = root / "cv_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Saved CV plan differs from the generated plan.")
    write_json(plan_path, plan)
    mapping_path = root / "state_mapping.json"
    if not mapping_path.exists():
        write_json(mapping_path, {item["key"]: {name: None for name in STATES} for item in plan})
    return plan


def collect_runs(pre, post, ids):
    arrays, lengths, subjects, tasks, phases = [], [], [], [], []
    for phase, data in (("pre", pre), ("post", post)):
        for subject in ids:
            valid_runs = 0
            for task, array in data[subject].items():
                array = np.asarray(array, dtype=float)
                if array.ndim != 2 or len(array) < 2:
                    LOG.warning("Skipping short/invalid run: %s/%s/%s", phase, subject, task)
                    continue
                if not np.isfinite(array).all() or np.any(array.std(axis=0, ddof=1) == 0):
                    LOG.warning("Skipping nonfinite/constant run: %s/%s/%s", phase, subject, task)
                    continue
                arrays.append(stats.zscore(array, axis=0, ddof=1))
                lengths.append(len(array))
                subjects.extend([subject] * len(array))
                tasks.extend([task] * len(array))
                phases.extend([phase] * len(array))
                valid_runs += 1
            if not valid_runs:
                raise ValueError(f"No valid runs: {phase}/{subject}")
    return np.vstack(arrays), lengths, subjects, tasks, phases


def make_state_dict(subjects, tasks, phases, states):
    if len({len(x) for x in (subjects, tasks, phases, states)}) != 1:
        raise ValueError("Sequence metadata lengths differ.")
    output = {"pre": {}, "post": {}}
    for subject, task, phase, state in zip(subjects, tasks, phases, states):
        output[phase].setdefault(subject, {}).setdefault(task, []).append(int(state))
    return output["pre"], output["post"]


def fit_hmms(plan, root, pre, post, cfg):
    from hmmlearn.hmm import GaussianHMM

    for item in plan:
        folder = root / item["key"]
        folder.mkdir(exist_ok=True)
        path = folder / "hmm.pkl"
        if path.exists():
            LOG.info("%s: saved HMM found; skipping", item["key"])
            continue
        LOG.info("%s: fitting HMM", item["key"])
        x, lengths, subjects, tasks, phases = collect_runs(pre, post, item["train_ids"])
        model = GaussianHMM(n_components=cfg["n_states"], covariance_type="diag",
                            n_iter=cfg["hmm_iterations"], random_state=cfg["hmm_seed"], tol=cfg["hmm_tol"])
        model.fit(x, lengths)
        labels = model.predict(x, lengths)
        pre_train, post_train = make_state_dict(subjects, tasks, phases, labels)
        counts = np.bincount(labels, minlength=cfg["n_states"])
        maps = np.full((cfg["n_states"], x.shape[1]), np.nan)
        for state, count in enumerate(counts):
            if count:
                maps[state] = x[labels == state].mean(axis=0)
        history = list(model.monitor_.history)
        gain = history[-1] - history[-2] if len(history) > 1 else np.nan
        if not np.isfinite(gain) or gain < -1e-3 or gain >= model.tol:
            warnings.warn(f"{item['key']}: inspect HMM convergence; final likelihood gain={gain}")
        del x, labels
        x, lengths, subjects, tasks, phases = collect_runs(pre, post, item["test_ids"])
        labels = model.predict(x, lengths)
        pre_test, post_test = make_state_dict(subjects, tasks, phases, labels)
        result = {**item, "model": model, "train_state_maps": maps, "train_frame_counts": counts,
                  "pre_train": pre_train, "post_train": post_train,
                  "pre_test": pre_test, "post_test": post_test, "log_likelihood_history": history}
        np.save(folder / "train_state_maps.npy", maps)
        save_pickle(path, result)
        LOG.info("%s: saved; training frame counts=%s", item["key"], counts)


def compute_statewise_fc(source, state_dict, ids, states, n_regions, min_tr):
    from nilearn.connectome import ConnectivityMeasure

    fc = {s: np.full((len(ids), n_regions, n_regions), np.nan) for s in states}
    counts = {s: np.zeros(len(ids), dtype=int) for s in states}
    for index, subject in enumerate(ids):
        chunks = {s: [] for s in states}
        if not state_dict.get(subject):
            raise ValueError(f"No state labels for {subject}")
        for task, labels in state_dict[subject].items():
            array, labels = np.asarray(source[subject][task]), np.asarray(labels)
            if array.ndim != 2 or array.shape[1] != n_regions or len(array) != len(labels):
                raise ValueError(f"Time series/label shape mismatch: {subject}/{task}")
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite time series: {subject}/{task}")
            for state in states:
                selected = array[labels == state]
                if len(selected):
                    chunks[state].append(selected)
        for state in states:
            if not chunks[state]:
                continue
            array = np.vstack(chunks[state])
            counts[state][index] = len(array)
            if len(array) < min_tr or np.any(array.std(axis=0) == 0):
                continue
            measure = ConnectivityMeasure(kind="correlation", vectorize=False, standardize=True)
            matrix = measure.fit_transform([array])[0]
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite FC: {subject}, state {state}")
            fc[state][index] = matrix
    return fc, counts


def row_zscore(x):
    sd = x.std(axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(sd == 0):
        raise ValueError("Invalid or constant feature row.")
    return (x - x.mean(axis=1, keepdims=True)) / sd


def safe_corr(a, b, method="pearson"):
    if len(a) < 3 or not np.isfinite(np.column_stack([a, b])).all():
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(function(a, b)[0])


def fit_one_rcca(x_train, x_test, y_train, y_test, columns, anchor, cfg):
    from cca_zoo.linear import rCCA

    # No CovBat is applied in this analysis.
    x_train, x_test = row_zscore(x_train), row_zscore(x_test)
    centerer = StandardScaler(with_std=False)
    x_train, x_test = centerer.fit_transform(x_train), centerer.transform(x_test)
    if np.any(y_train.std(axis=0) == 0):
        raise ValueError("Constant clinical variable in training data.")
    scaler = StandardScaler()
    y_train, y_test = scaler.fit_transform(y_train), scaler.transform(y_test)
    model = rCCA(latent_dimensions=1, c=cfg["fixed_c"])
    model.fit((x_train, y_train))
    a_train, b_train = [x[:, 0] for x in model.transform((x_train, y_train))]
    a_test, b_test = [x[:, 0] for x in model.transform((x_test, y_test))]
    anchor_r = safe_corr(y_train[:, columns.index(anchor)], b_train)
    if not np.isfinite(anchor_r):
        raise ValueError("Invalid training anchor correlation.")
    if abs(anchor_r) < 0.05:
        warnings.warn("Weak clinical anchor; inspect cross-fold loading stability.")
    sign = 1.0 if anchor_r >= 0 else -1.0
    a_train, b_train, a_test, b_test = [sign * x for x in (a_train, b_train, a_test, b_test)]
    means = np.array([a_train.mean(), b_train.mean()])
    scales = np.array([a_train.std(ddof=1), b_train.std(ddof=1)])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid canonical score variance.")
    fitted = {"model": model, "X_centerer": centerer, "Y_scaler": scaler,
              "sign": sign, "score_mean": means, "score_sd": scales, "row_zscore_X": True,
              "clinical_loadings": [safe_corr(y_train[:, j], b_train) for j in range(len(columns))],
              "clinical_columns": columns, "anchor_column": anchor,
              "c": cfg["fixed_c"], "feature_key": cfg["feature_key"], "harmonized": False}
    return (a_test - means[0]) / scales[0], (b_test - means[1]) / scales[1], fitted


def run_fc_rcca(plan, root, pre, post, table, columns, anchor, cfg):
    mappings = read_json(root / "state_mapping.json")
    for item in plan:
        pair = mappings.get(item["key"], {})
        values = [pair.get(name) for name in STATES]
        if any(type(v) is not int or not 0 <= v < cfg["n_states"] for v in values):
            raise ValueError(f"Complete training-only state mapping for {item['key']} before rCCA.")
        if values[0] == values[1]:
            raise ValueError("Target states must have distinct labels.")
        if not (root / item["key"] / "hmm.pkl").exists():
            raise FileNotFoundError(f"Missing HMM for {item['key']}")
    locked = root / "evaluated_state_mapping.json"
    if locked.exists() and read_json(locked) != mappings:
        raise ValueError("State mappings changed after evaluation started. Use a new analysis directory.")
    write_json(locked, mappings)
    scores_all, loadings_all = [], []
    for item in plan:
        folder = root / item["key"]
        with (folder / "hmm.pkl").open("rb") as stream:
            result = pickle.load(stream)
        if any(result[name] != item[name] for name in ("train_ids", "test_ids")):
            raise ValueError("Saved HMM participant IDs differ from the CV plan.")
        mapping = {name: mappings[item["key"]][name] for name in STATES}
        regions = result["model"].means_.shape[1]
        edges = np.triu_indices(regions, k=1)
        fc_data = {}
        for split in ("train", "test"):
            for phase, source in (("pre", pre), ("post", post)):
                matrices, counts = compute_statewise_fc(source, result[f"{phase}_{split}"],
                    item[f"{split}_ids"], list(mapping.values()), regions, cfg["min_tr"])
                fc_data[split, phase] = {name: (matrices[s][:, edges[0], edges[1]], counts[s])
                                        for name, s in mapping.items()}
                del matrices
        for state in STATES:
            x = {}
            for split in ("train", "test"):
                r_pre, n_pre = fc_data[split, "pre"][state]
                r_post, n_post = fc_data[split, "post"][state]
                delta_r = r_post - r_pre
                delta_z = (np.arctanh(np.clip(r_post, -1 + 1e-7, 1 - 1e-7))
                           - np.arctanh(np.clip(r_pre, -1 + 1e-7, 1 - 1e-7)))
                x[split] = {"delta_r": delta_r, "delta_z": delta_z}[cfg["feature_key"]]
                valid = np.isfinite(x[split]).all(axis=1)
                np.savez_compressed(folder / f"{split}_{state}_FC.npz",
                    subject_ids=np.asarray(item[f"{split}_ids"], dtype=str),
                    r_pre=r_pre, r_post=r_post, delta_r=delta_r, delta_z=delta_z,
                    n_frames_pre=n_pre, n_frames_post=n_post, valid=valid,
                    state_label=mapping[state], edge_i=edges[0], edge_j=edges[1])
                if not valid.all():
                    bad = np.asarray(item[f"{split}_ids"])[~valid].tolist()
                    raise ValueError(f"Missing FC: {item['key']}/{state}/{split}: {bad}")
            y_train = table.loc[item["train_ids"], columns].to_numpy(dtype=float)
            y_test = table.loc[item["test_ids"], columns].to_numpy(dtype=float)
            a, b, fitted = fit_one_rcca(x["train"], x["test"], y_train, y_test, columns, anchor, cfg)
            fitted.update(train_ids=item["train_ids"], test_ids=item["test_ids"], state_mapping=mapping)
            save_pickle(folder / f"{state}_rCCA_noCovBat.pkl", fitted)
            scores = pd.DataFrame({"Subject": item["test_ids"], "Repeat": item["repeat"],
                "Fold": item["fold"], "State": state, "a_test": a, "b_test": b})
            scores.to_csv(folder / f"{state}_OOF_noCovBat.csv", index=False)
            scores_all.append(scores)
            loadings_all.append({"Repeat": item["repeat"], "Fold": item["fold"], "State": state,
                                 **dict(zip(columns, fitted["clinical_loadings"]))})
            LOG.info("%s %s: test r=%.4f", item["key"], state, safe_corr(a, b))
    output = pd.concat(scores_all, ignore_index=True)
    output.to_csv(root / "all_OOF_scores_noCovBat.csv", index=False)
    pd.DataFrame(loadings_all).to_csv(root / "clinical_loadings_noCovBat.csv", index=False)
    return output


def fisher_mean(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return np.nan
    return np.tanh(np.arctanh(np.clip(values, -0.999999, 0.999999)).mean())


def summarize(scores, ids, root, n_repeats):
    expected = {(repeat, state) for repeat in range(1, n_repeats + 1) for state in STATES}
    if set(scores.groupby(["Repeat", "State"]).groups) != expected:
        raise ValueError("Incomplete repeat/state coverage.")
    rows = []
    for (repeat, state), group in scores.groupby(["Repeat", "State"]):
        if group["Subject"].duplicated().any() or set(group["Subject"]) != set(ids):
            raise ValueError("Incomplete or duplicated out-of-fold participant coverage.")
        group = group.set_index("Subject").loc[ids]
        a, b = group["a_test"].to_numpy(), group["b_test"].to_numpy()
        if not np.isfinite(np.column_stack([a, b])).all():
            raise ValueError("Nonfinite out-of-fold scores.")
        rows.append({"Repeat": repeat, "State": state, "N": len(group),
                     "Pearson_r": safe_corr(a, b), "Spearman_rho": safe_corr(a, b, "spearman")})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "CV_summary_by_repeat_noCovBat.csv", index=False)
    overall = summary.groupby("State").agg(Mean_Pearson_r=("Pearson_r", "mean"),
        Fisher_mean_r=("Pearson_r", fisher_mean), Min_Pearson_r=("Pearson_r", "min"),
        Max_Pearson_r=("Pearson_r", "max"))
    overall.to_csv(root / "CV_summary_overall_noCovBat.csv")
    print(summary.to_string(index=False))
    print(overall.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("hmm", "rcca"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = read_json(args.config)
    if not cfg.get("exclusions_reviewed", False):
        raise ValueError("Populate the original exclusion lists and set exclusions_reviewed=true.")
    if cfg["feature_key"] not in ("delta_r", "delta_z"):
        raise ValueError("feature_key must be delta_r or delta_z.")
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    pre, post = prepare_data(cfg, root)
    table, columns, anchor, ids = load_clinical(cfg, pre, post)
    plan = initialize_plan(root, cfg, ids, columns, anchor, data_digest(pre, post, table, columns, ids))
    if args.stage == "hmm":
        fit_hmms(plan, root, pre, post, cfg)
        LOG.info("Inspect training maps and complete state_mapping.json before the rCCA stage.")
    else:
        scores = run_fc_rcca(plan, root, pre, post, table, columns, anchor, cfg)
        summarize(scores, ids, root, cfg["n_repeats"])


if __name__ == "__main__":
    main()
