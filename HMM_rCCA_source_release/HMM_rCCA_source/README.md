# Fold-specific HMM and rCCA cross-validation

This source-only release organizes the supplied analysis into a standalone script. It includes motion censoring and interpolation, participant-level 10-fold CV repeated five times, training-only HMM estimation, fixed-model held-out decoding, statewise FC, rCCA, and out-of-fold summaries. It does **not** implement CovBat, hyperparameter selection, or permutation p-values.

## Files

- `hmm_rcca_pipeline.py`: complete analysis script.
- `config.example.json`: portable configuration template; no personal absolute paths or participant exclusion IDs.
- `requirements.txt`: dependency names, not a validated locked environment.

## Configure and run

Use the same Python environment as the original running analysis. Copy `config.example.json` to `config.local.json` and update input/output paths. Relative paths resolve from the working directory. Restore the actual original exclusion lists in the private configuration, then set `exclusions_reviewed` to `true`. The empty example lists are not the study's actual exclusions. Do not upload the private configuration or generated subject-level outputs as part of this source-only release.

```bash
python hmm_rcca_pipeline.py --config config.local.json --stage hmm
```

The HMM stage creates `rep01_fold01` through `rep05_fold10`. Inspect each fold's `train_state_maps.npy`, and fill **every** entry in `state_mapping.json` with the appropriate distinct zero-based labels:

```json
{"rep01_fold01": {"State2": 0, "State4": 1}}
```

This is an illustration for one fold, not a universal state mapping. Use training maps only; do not choose mappings based on held-out FC, symptoms, or rCCA performance. Document the state-matching rule and the origin of any reference template separately. A full-sample template or original outcome-driven target-state selection limits claims of completely independent validation.

```bash
python hmm_rcca_pipeline.py --config config.local.json --stage rcca
```

Re-running the HMM command skips atomically saved `hmm.pkl` files. The rCCA stage recomputes FC/rCCA and overwrites its own outputs deterministically. State mappings are locked when evaluation starts. A manifest checks prepared input content/order, clinical data, configuration, package versions, and script identity before reusing models. Use a **new output directory** for this release; it does not import checkpoints from the earlier notebook format. Do not interrupt or replace the ongoing original analysis merely to switch to this release.

## Input contract

NumPy files contain trusted pickle-backed dictionaries of the form `{subject_id: {task_name: time_by_roi_array}}`. Original arrays must already correspond to the confound rows remaining after `skip_fd_tr=5`; the script trims the FD vector, not the time series. Identifiers beginning with the original study prefixes PCS/CCS have their trailing session letter removed and receive A/C for pre/post confound lookup. Adapt this rule if using another dataset.

The clinical CSV's first column contains participant IDs. By default, columns 2-9 provide the eight clinical variables. Specify `clinical_columns` explicitly if needed. The script uses the supplied values as-is; it does not compute symptom change. The fixed sign anchor defaults to the first clinical variable. Choose it before evaluation.

PT participants are the ordered intersection of pre/post imaging dictionaries after the configured exclusions. Missing clinical records are errors. One-session-only participants are omitted by this intersection, preserving the original code. HC preprocessing is retained but HC data are not used for PT HMM/rCCA; remove the HC configuration block if HC processing is not needed.

## Preserved analytical choices

- FD threshold: 0.3; NaN FD frames are censored. The original sequential in-place neighbor expansion is preserved exactly. This is **not** replaced by a conventional symmetric dilation. With the supplied `expand_frames=1`, an isolated interior outlier and its immediate neighbors are marked. With larger expansion settings, sequential updates can extend beyond the requested radius. Review the original scientific intent before changing it; a change would require rerunning the analysis.
- `motion_report_threshold=0.4` is only a subject-level descriptive flag. It is not an automatic run-retention or exclusion rule. The actual exclusions are supplied explicitly. In particular, the script does not implement a separate 'at least 50% low-motion frames per run' gate. The manuscript must describe the actual rule used.
- Cubic interpolation uses the original private Nilearn `_interpolate_volumes` function. Boundary outliers are replaced by the nearest retained volume. No-outlier runs are returned unchanged; fewer than two retained frames raises an explicit error. Use the original compatible Nilearn environment.
- HMM input is standardized per run and per ROI with `ddof=1`. Invalid runs are skipped with a warning; a participant with no valid run in either phase triggers an error. This can change the effective task/rest composition even when planned acquisitions are matched.
- Eight diagonal-covariance states, 200 iterations, seed 42; each participant's pre/post data remain on the same CV side. All run lengths are retained. The supplied implementation uses both pre and post imaging and is not a baseline-only prognosis model.
- Raw interpolated frames assigned to each state are pooled across runs within participant/session before FC estimation. This is neither an average of task-specific FCs nor task-regressed FC. Nilearn correlation uses its default covariance estimator (Ledoit-Wolf in the originating implementation). A minimum of three frames is a computational threshold, not an assurance of FC stability.
- Both raw-r and Fisher-z post-minus-pre edge changes are saved. rCCA uses raw-r changes by default, row z-scoring, training-feature centering, training-only clinical standardization, one component, and fixed `c=0.7`.
- The training clinical anchor jointly orients both score vectors; training score moments determine held-out scaling. Inspect cross-fold clinical loadings: sign alignment does not establish that every fold recovers the same latent dimension.
- Each repetition contributes one held-out score pair per participant. Pearson/Spearman correlations are descriptive OOF associations. Arithmetic and Fisher-transformed means are saved separately. Repetitions are not independent samples. No inferential p-values are produced.

## Outputs and privacy

Outputs include motion reports, length checks, the analysis manifest, CV membership, per-fold HMMs/training maps, FC edges and frame counts, rCCA models, clinical loadings, individual held-out scores, and summary correlations. These files can contain participant identifiers and clinical/imaging information; excluding local paths from the source does not anonymize generated results. The public bundle contains no study data or fitted models.

## Changes from the submitted notebook text

Removed repeated imports, Chinese comments/messages, local absolute paths, literal exclusion IDs, notebook-only display calls, and obsolete single-fold calls. Added the missing `ConnectivityMeasure` import and CV-plan initialization; corrected function execution order and inconsistent checkpoint directory handling. Motion reports now include group and phase and correctly describe the 0.4 flag. Replaced verbose boundary-outlier bookkeeping with direct first/last retained-frame clamping; retained Nilearn's interpolation routine. Added input/version fingerprints, duplicate-ID checks, atomic checkpoints, and a two-stage command-line interface.

The supplied exclusion lists must be restored privately for numerical reproduction. Record the original environment with `python -m pip freeze > environment_original.txt` and review that file for local paths before sharing. No package versions have been invented. Compatibility with future `cca-zoo` or Nilearn versions is not assumed.

Validation here covers Python syntax, CLI help, and targeted synthetic checks of routing, ID normalization, mask behavior, fold coverage, and summary assembly. Full HMM/FC/rCCA execution on the study data has not been performed in this environment; numerical equivalence to the running notebook has not been established.
