# Longitudinal temperature-cycled chemical-sensing benchmark

Research code and frozen results for the seven experiments reported in *A Longitudinal Multi-Device Benchmark for Temperature-Cycled Chemical Sensing*. All experiments use the original measurement records and preprocessing implementation; no model scores were regenerated while preparing this artifact.

## Layout

- `src/esdata/`: canonical ingestion, segmentation, temperature alignment, and air referencing (unaltered scientific implementation).
- `experiments/`: seven ordered Jupyter notebooks (E1–E7), with code and methodological decisions kept intact.
- `results/`: original result tables, per-record predictions, and figures. Values are frozen evidence from the executed experiments.
- `data/README.md`: required raw archives and access instructions.
- `scripts/verify_artifact.py` and `scripts/verify_paper_scores.py`: source integrity and headline-score checks.
- `scripts/reproduce_e6_retrospective.py`: the supplementary E6 comparison of day-grouped and forward-in-time concentration calibration.

## Installation

Python 3.11 is recommended. In a clean environment from the repository root:

```bash
python -m venv .venv
# Activate .venv with your shell's usual command.
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python scripts/restore_large_results.py
python scripts/verify_artifact.py
python scripts/verify_paper_scores.py
```

Obtain the raw archives separately through an authorized anonymous dataset release. Their expected checksums are in `data/archive_checksums.sha256`. Place them under `data/raw/` with the original archive basenames, or set `ESMELL_DATA_DIR` to the folder containing them. The archives are **not** redistributed in this code package.

Run notebooks in numerical order, either from the repository root or opened in the `experiments/` directory. The default raw-data path is resolved from the repository root; set `ESMELL_DATA_DIR` to an absolute path for a different data location. Each notebook recomputes its own analysis and writes `outputs/` or `src/outputs/`; the frozen results in `results/` are never overwritten. Some runs require substantial CPU time or optional GPU acceleration. Full execution requires the original datasets and compatible `sktime`/`numba` versions.

## Supplementary E6 protocol comparison

The manuscript additionally reports a retrospective comparison using the original
ExtraTrees regression configuration from `notebooks/multi_label_exp.ipynb`
(160 trees, four samples per leaf, `max_features=0.85`, a separate regressor
for each present gas, and `log10(ppb)` targets). This result is **not** one
of the seven main E6 model configurations; it tests how evaluation protocol
changes the original calibration result.

First run E5 or E6 to create the verified temperature-aligned signal cache.
Then, from the repository root:

```bash
python scripts/reproduce_e6_retrospective.py
```

The script requires the two original raw archives (130 and 146), or their
location in `ESMELL_DATA_DIR`. It verifies archive SHA-256 hashes, `src/esdata`
source hashes, and the cache signature. `--cache PATH` selects an existing
E5/E6 cache, and `--data-dir PATH` specifies the raw-data directory. The
original notebook ran on Windows, so the signature check supports both
Windows and POSIX paths without relaxing the source-integrity check.

Outputs go to `outputs/E6_retrospective/`: a three-protocol table, six gas ×
exposure-type errors per protocol, positive-component predictions, and the
run configuration. Frozen copies are under `results/E6/retrospective_*`.
Running the script does not modify any main-experiment results.

The day-grouped three-fold comparison uses all 2,561 included records from
campaign 130 through September 28. Macro-6 is computed on **pooled
out-of-fold predictions**, because one fold contains no held-out mixtures.
For both forward tests the model is fitted only on the 2,409 records through
September 27: it predicts the next 152 records on September 28 and all 916
records of campaign 146. The additional nine October 5 records of campaign
130 are not part of the fixed E5/E6 cohort. No test-day labels are used for
fitting. This is an exploratory protocol diagnostic, not a new confirmatory
test of temporal drift.

## Scope of experiments

| Experiment | Question | Data |
| --- | --- | --- |
| E1 | Frozen calibration on later campaigns; later recalibration | 130, 146, 166, 169 |
| E2 | Paired-device transfer and labeled target calibration | 171/172; 129/130 diagnostic |
| E3 | Air referencing and thermal-phase ablations | 171, 172 |
| E4 | Single-gas model comparison | 171, 172 |
| E5 | Multi-label mixture composition | 130 → 146 |
| E6 | Concentration estimation with oracle composition | 130 → 146 |
| E7 | End-to-end composition and concentration without oracle inputs | later 130 and 146 |

Numbers 120/122 refer to excluded faulty campaigns. They are absent from the seven publication notebooks and result collections. Different tasks are not pooled into one metric when their analyte vocabularies or evaluation splits differ.

## Reproducibility boundaries

Training-test separation is described in each notebook. The E5/E6 and E7 splits are not interchangeable: E7 reserves 28 September as a within-campaign future test, while E5/E6 used it for validation. Random, paired, and chronological evaluations answer different questions. Concentrations denote nominal commanded levels, not verified delivered doses. Campaign 146 was inspected during earlier experimental development, so the diagnostic extensions are retrospective rather than fresh independent confirmation.

See `RELEASE_NOTES.md` for verification scope and the separate dataset-access requirement.

## Anonymous GitHub mirror: large result tables

Anonymous GitHub does not serve individual files above 8 MB. To preserve all
per-record predictions without changing their bytes, three large CSVs are
shipped in `compressed_results/full_prediction_tables.zip` (<8 MB). Restore
them after cloning with `python scripts/restore_large_results.py`; then run
the checksum verification commands above. The restored tables match the
original experiment outputs byte-for-byte. This packaging does not modify
training, splits, model parameters, or results.

The compressed bundle is binary, so anonymous.4open.science cannot scrub its
contents. The release owner must separately check its contents and secure
all dataset/software distribution permissions before publication.
