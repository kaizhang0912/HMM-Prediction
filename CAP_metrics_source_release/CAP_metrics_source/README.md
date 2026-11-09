# CAP/HMM temporal metrics

English-only, source-only implementation of occupancy rate (OCR), dwell time (DT), transition probability (TP), and the supplied entropy-of-Markov-trajectories (EMT) matrix formula. No local study paths, participant IDs, or study data are embedded. The metrics operate on state-label sequences and are independent of how the states were estimated.

## Use

Dependencies: NumPy, pandas, SciPy. Place `cap_metrics.py` and `example_usage.py` on the Python import path.

```python
from example_usage import export_four_datasets

results = export_four_datasets(
    state_index_dict_pre_pt, state_index_dict_post_pt,
    state_index_dict_pre_hc, state_index_dict_post_hc,
    output_dir="./results/cap_metrics", k_states=8, tr=2.0
)
```

Alternatively, save a state dictionary to JSON and use the command line:

```bash
python cap_metrics.py --input states.json --output results/PT_pre --states 8 --tr 2.0
```

Input structure: `{subject_id: {run_id: [0, 1, 1, ...]}}`. State labels are zero-based integers in `0..K-1`. Each dictionary entry must represent an actual independent run; do not merge multiple runs under a single key before computing DT/TP. The code validates labels and preserves subject insertion order. Pair pre/post results by `Subject`, not row position.

## Definitions

- **OCR:** the fraction of all available frames assigned to each state across runs. Runs contribute in proportion to frame count; this is not an equal-run average.
- **DT:** contiguous visit lengths within each run, then a mean over all visits for each participant/state. Visits are never joined across run boundaries. Boundary visits are counted at their observed duration without censoring correction. Unvisited states receive NaN, not zero. Group summaries average participant means with equal participant weights, exclude NaNs, and use population SD (`ddof=0`), matching the supplied calculation. Both frame and second units are exported.
- **TP:** pooled counts of adjacent-frame transitions within runs, including self-transitions, divided by each row's departure count. No cross-run transitions are introduced. Rows with no departures are zero-filled and flagged; they are not valid estimated probability distributions.
- **EMT:** the original supplied matrix formula, in bits (`log2`). All K-by-K entries are retained. The original diagonal convention, entropy rate divided by stationary probability, is retained; the diagonal is not set to zero. The scalar mean includes all entries, including the diagonal, exactly as the supplied `H.mean()` step. This is not transfer entropy, nor is it simply local transition entropy or the entropy rate.

The script does not perform statistical tests, FDR correction, harmonization, imputation, or state matching. Separately fitted PT/HC or cross-validation HMMs do not automatically share numeric state labels. Establish state correspondence separately before comparing them.

## Outputs for each dataset

`OCR.csv`, `DT_frames.csv`, `DT_seconds.csv`, `DT_group_summary.csv`, `transition_counts.csv`, `TP.csv`, `EMT.csv`, `EMT_mean.csv`, `QC.csv`, `dwell_visits.json`, `metadata.json`, and `metrics.npz`.

TP/EMT CSV columns use row-major order: `From0_To0`, `From0_To1`, ..., `From7_To7` for eight states. Rows of the source matrix are origins and columns are destinations. NPZ files retain full matrices and explicit subject IDs. Invalid entropy matrices are entirely NaN and accompanied by a reason in QC; no participants are silently removed. Add `--strict-entropy` to stop at the first invalid matrix. Do not replace invalid values with zero before inference.

## Explicit numerical corrections

This release is **not merely a formatting-only rewrite**:

1. The supplied `np.divide(..., where=...)` did not provide `out`, leaving uncomputed zero-count rows undefined. The release initializes those rows to zero and flags them. Values for rows with observed departures follow the original definition.
2. The supplied stationary-distribution function took the first eigenvector returned by `scipy.linalg.eig`, without checking its eigenvalue. Eigenvector ordering does not ensure that this is the vector for eigenvalue 1; taking absolute values does not fix the issue. The release solves `mu @ P = mu` with `sum(mu)=1`, checks the residual and positive probabilities, and requires irreducibility for this full-state formula. It does not require aperiodicity.
3. The matrix inverse is replaced with a linear solve using the same right-hand side. No smoothing or pseudocounts are added. Reducible/non-stochastic matrices are explicitly flagged rather than producing an arbitrary result.
4. Empty participants and unvisited states are retained with NaN where needed; the original OCR code skipped empty participants. Dwell summaries now return labeled tables rather than unlabeled lists. Functions accept explicit state counts rather than notebook globals.

These corrections can change previously computed EMT values or expose unavailable estimates. Compare the released code with the original study outputs and rerun affected downstream statistics before claiming reproduction. If reproducing an earlier publication's exact software behavior is required, archive the original implementation separately rather than silently substituting corrected results.

## Verification and scope

Synthetic checks cover run boundaries, unequal-run OCR weighting, missing states, zero-departure rows, known stationary distributions, zero entropy for a deterministic alternating chain, the two-state uniform-chain entropy matrix, and output/subject alignment. Source syntax and command-line help were checked. The original participant data and manuscript results have not been executed or validated here.

The files generated when running this code contain participant identifiers. Only source files are included in this package; de-identify any result files separately before sharing. Record the actual analysis environment versions for a reproducible research release.
