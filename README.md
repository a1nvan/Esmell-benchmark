# Temperature-Cycled Chemical Sensing Benchmark

Code and results for the experiments in *A Longitudinal Multi-Device Benchmark for Temperature-Cycled Chemical Sensing*.

## Contents

- `src/esdata/` — data loading, cycle segmentation, temperature alignment, and reference-air subtraction.
- `experiments/` — Jupyter notebooks for experiments E1–E7.
- `results/` — evaluation tables, predictions, and figures.
- `scripts/` — result verification and the additional E6 protocol comparison.
- `data/` — archive names and checksums. Raw measurements are distributed in the companion dataset.

## Setup

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e . -r requirements.txt
python scripts/restore_large_results.py
python scripts/verify_artifact.py
python scripts/verify_paper_scores.py
```

Download the eight raw measurement archives from the dataset record linked in the paper. Keep their original filenames and place them in `data/raw/`, or set `ESMELL_DATA_DIR` to the directory containing them. Expected SHA-256 hashes are in `data/archive_checksums.sha256`.

Open the notebooks in `experiments/` and run their cells in order. Each notebook specifies its split, model settings, and evaluation metrics. Outputs are written to `outputs/` or `src/outputs/`; archived results in `results/` are retained for comparison.

## Experiments

| Notebook | Evaluation |
| --- | --- |
| E1 | Longitudinal calibration reuse (130, 146, 166, 169) |
| E2 | Paired-device transfer and labeled recalibration (171/172, 129/130) |
| E3 | Air-reference and thermal-phase ablations (171, 172) |
| E4 | Single-gas classification (171, 172) |
| E5 | Mixture-component detection (130 → 146) |
| E6 | Concentration regression with known composition (130 → 146) |
| E7 | Joint detection and quantification (future measurements within 130 and in 146) |

E5/E6 use September 28 for model selection; E7 uses it as a future test with a different training/validation split. Protocol definitions and limitations are documented in the notebooks.

## Additional E6 comparison

The article also reports a comparison of day-grouped and chronological concentration evaluation using the original ExtraTrees configuration (160 trees, `min_samples_leaf=4`, `max_features=0.85`, gas-specific `log10(ppb)` regression). This analysis is implemented in `scripts/reproduce_e6_retrospective.py`.

Run E5 or E6 first to generate the temperature-aligned signal cache, then execute:

```bash
python scripts/reproduce_e6_retrospective.py
```

Results are written to `outputs/E6_retrospective/`; reference outputs are in `results/E6/retrospective_*`. The script accepts `--cache PATH` and `--data-dir PATH` if the cache or raw archives are elsewhere.

## Full prediction tables

Three prediction CSVs are stored in `compressed_results/full_prediction_tables.zip` to meet the anonymous code mirror's file-size limit. `scripts/restore_large_results.py` extracts them without changing their contents. Run this before checking result hashes.
