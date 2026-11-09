# Brain-State Dynamics and Cross-Validated Brain–Clinical Associations

Python pipelines for quantifying brain-state temporal dynamics and evaluating brain–clinical associations using fold-specific hidden Markov models (HMMs) and regularized canonical correlation analysis (rCCA).

## Repository Structure

```text
HMM_rCCA_source/
    hmm_rcca_pipeline.py
    config.example.json
    requirements.txt
    README.md

CAP_metrics_source/
    cap_metrics.py
    example_usage.py
    test_cap_metrics.py
    requirements.txt
    README.md
```

## Requirements

Install the dependencies for each pipeline:

```bash
pip install -r HMM_rCCA_source/requirements.txt
pip install -r CAP_metrics_source/requirements.txt
```

Use the original analysis environment for numerical reproduction. The dependency lists are not version-locked; compatibility with other Nilearn or CCA-Zoo versions is not guaranteed.

## 1. HMM–rCCA Cross-Validation

This pipeline performs:

1. Motion censoring and time-series interpolation.
2. Participant-level 10-fold cross-validation repeated five times.
3. HMM estimation using training participants only.
4. Held-out state decoding using the fixed training-derived HMM.
5. State-specific functional connectivity estimation.
6. Training-only rCCA fitting and held-out score calculation.
7. Aggregation of out-of-fold brain–clinical associations.

Each participant’s pre- and post-treatment data remain in the same fold.

### Input

- Pre- and post-treatment time-series dictionaries:

```python
{
    "subject_001": {
        "run_1": time_by_roi_array,
        "run_2": time_by_roi_array
    }
}
```

- Corresponding confound TSV files containing framewise displacement.
- A clinical CSV with participant IDs in the first column.
- Participant exclusion lists supplied through a private configuration file.

Clinical variables are used as supplied; symptom changes are not calculated automatically.

### Configuration

Copy `config.example.json` to `config.local.json`. Update the paths, restore the original exclusion lists, and set:

```json
"exclusions_reviewed": true
```

### Run HMM Estimation

```bash
python HMM_rCCA_source/hmm_rcca_pipeline.py \
    --config config.local.json \
    --stage hmm
```

Inspect each fold’s `train_state_maps.npy` and complete the generated `state_mapping.json`.

State labels are zero-based. Determine correspondence using training-derived maps, without reference to held-out outcomes.

### Run FC and rCCA

```bash
python HMM_rCCA_source/hmm_rcca_pipeline.py \
    --config config.local.json \
    --stage rcca
```

### Main Outputs

- CV participant assignments and analysis configuration.
- Per-fold HMM models and training-state maps.
- State-specific connectivity features and frame counts.
- Fitted rCCA models and training clinical loadings.
- Individual out-of-fold canonical scores.
- Pearson and Spearman correlations for each repetition.
- Summary correlations across repetitions.

This implementation uses one canonical dimension and a fixed regularization parameter of `c = 0.7`. CovBat, hyperparameter optimization, and permutation inference are not implemented.

Because imaging features include post-treatment data, the results describe held-out brain–clinical associations rather than baseline-only prognosis.

## 2. Brain-State Temporal Metrics

This pipeline computes:

| Metric | Description |
|--------|-------------|
| OCR | Fraction of frames assigned to each state |
| DT | Mean contiguous state-visit duration |
| TP | Within-run transition probability matrix |
| EMT | Entropy of Markov trajectories |

### Input

State sequences are organized by participant and run:

```python
state_dict = {
    "subject_001": {
        "run_1": [0, 0, 1, 2, 2, 1],
        "run_2": [1, 1, 0, 2, 2, 0]
    }
}
```

Labels must be integers between `0` and `k_states - 1`. Separate runs must remain separate.

### Python Usage

From the `CAP_metrics_source` directory:

```python
from cap_metrics import export_metrics

results = export_metrics(
    state_dict,
    output_dir="./results/temporal_metrics",
    k_clusters=8,
    tr_seconds=2.0
)
```

The `example_usage.py` module also supports exporting patient/control and pre/post datasets together.

### Command-Line Usage

```bash
python CAP_metrics_source/cap_metrics.py \
    --input states.json \
    --output results/temporal_metrics \
    --states 8 \
    --tr 2.0
```

### Main Outputs

- `OCR.csv`
- `DT_frames.csv` and `DT_seconds.csv`
- `DT_group_summary.csv`
- `transition_counts.csv` and `TP.csv`
- `EMT.csv` and `EMT_mean.csv`
- `QC.csv`
- `metrics.npz`

Additional files contain individual dwell durations and calculation metadata.

## Methodological Notes

- DT and TP do not count across run boundaries.
- TP includes self-transitions.
- Statewise FC pools state-assigned frames across runs within each participant and assessment.
- Unvisited states receive missing DT values.
- TP rows without observed departures are zero-filled and flagged.
- EMT uses a validated stationary distribution and requires an irreducible transition matrix. Unavailable estimates are flagged as missing.
- Mean EMT includes all matrix entries, including the diagonal.
- Independently estimated HMMs do not automatically share state labels; state correspondence must be established separately.
- Sign alignment does not guarantee that every CV fold recovers the same canonical dimension.
- CV repetitions are not independent samples. No inferential p-values are generated.

The temporal-metric implementation corrects undefined zero-count transition rows and the original stationary-eigenvector selection. These corrections may affect earlier EMT results and should be checked against the original analysis.

## Validation

Run the temporal-metric checks:

```bash
cd CAP_metrics_source
python -m unittest test_cap_metrics.py
```

Syntax and selected synthetic workflow checks were performed. Numerical equivalence on the original study data has not been established.

## Data and Privacy

This repository contains source code only. Local paths and participant exclusion IDs are supplied through private configuration files. Generated outputs may contain participant identifiers and clinical or imaging information and require separate review before public sharing.

See the README within each pipeline directory for detailed input requirements and implementation notes.
