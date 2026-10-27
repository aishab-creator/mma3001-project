# Turbofan Engine Remaining Useful Life Prediction

MMA3001 project investigating machine learning prediction of remaining useful life (RUL) using NASA C-MAPSS turbofan degradation data. RUL is measured in operating cycles.

The project develops a Random Forest baseline, investigates temporal and operating-regime features, examines near-failure overestimation, and compares different training RUL caps across FD001–FD004. This is an academic study using simulated engines, not a validated aircraft maintenance decision system.

## Data

Obtain the C-MAPSS Turbofan Engine Degradation Simulation Data Set from NASA's Prognostics Center of Excellence data repository. Download the original FD001–FD004 text files and place them in `data/raw/`. Retain the accompanying NASA documentation and check its usage and redistribution terms before distributing raw files. Raw NASA data are not bundled with this README.

For each subset, the notebooks expect:

| File pattern | Purpose |
| --- | --- |
| `train_FD001.txt` through `train_FD004.txt` | Complete run-to-failure training histories; training RUL is calculated as the engine's last cycle minus the current cycle. |
| `test_FD001.txt` through `test_FD004.txt` | Truncated histories; one prediction is made at each engine's last visible cycle. |
| `RUL_FD001.txt` through `RUL_FD004.txt` | Actual remaining life at each official test stopping point, ordered by engine ID. These labels are used for evaluation. |

Each train/test row contains an engine identifier, cycle, three operating settings and 21 sensor measurements. Engine identifiers are local to each subset; subsets are trained separately.

| Subset | Operating conditions | Fault modes |
| --- | ---: | ---: |
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

## Suggested repository layout

| Path | Contents |
| --- | --- |
| `notebooks/` | Development and evaluation notebooks listed below |
| `data/raw/` | The 12 NASA train, test and RUL text files |
| `results/metrics/` | Exported model metrics and predictions |
| `results/figures/` | Development figures |
| `results/calibration/` | Notebook 07 calibration outputs |
| `results/training_cap_experiment/` | Notebook 08 outputs, grouped by configuration signature |
| `results/final_evaluation/` | Notebook 06 report figures and tables |

These are expected workflow locations, not a claim that every directory or supporting document is already present. Some earlier notebooks use configurable paths; inspect their setup cells before running.

## Environment setup

Use Python 3 in an isolated environment. From the repository root, run the following commands in a terminal on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab ipykernel joblib
python -m ipykernel install --user --name cmapss-project --display-name "Python (C-MAPSS)"
```

In VS Code, install the Python and Jupyter extensions, open the repository folder, then select **Python (C-MAPSS)** as the notebook kernel. Alternatively, run `jupyter lab` from the repository root.

These commands install the main notebook dependencies; they do not define a pinned reproduction environment. After a successful run, record the actual package versions, for example:

```bash
python -m pip freeze > requirements.txt
```

To recreate that recorded environment later, activate a new virtual environment and run `python -m pip install -r requirements.txt`.

## Notebook execution order

Run notebooks with the working directory set to the repository root or its `notebooks/` folder. Restart the kernel and use **Run All** for reproducible execution. Do not rely on variables left over from another notebook.

| Order | Notebook | Role |
| --- | --- | --- |
| 1 | `01_baseline_all_datasets.ipynb` | Establish baseline models for FD001–FD004. |
| 2 | `02_test_evaluation_all_datasets.ipynb` | Evaluate baseline predictions against official uncapped RUL. |
| 3 | `03_model_improvements_all_datasets.ipynb` | Explore target capping, temporal features and Random Forest settings. |
| 4 | `05_multi_dataset_improvements.ipynb` | Compare feature variants with a fixed 100-cycle training cap and export selected-model predictions. |
| 5 | `05_rul_robustness_and_safety.ipynb` | Compare caps and feature variants for FD002/FD004 using repeated validation and near-failure diagnostics. |
| 6 | `07_rul_validation_calibration.ipynb` | Select conservative offsets from validation predictions and evaluate their trade-offs. |
| 7 | `08_training_cap_comparison.ipynb` | Compare training caps across all four subsets using uncapped validation RMSE. |
| 8 | `06_final_rul_evaluation_v2.ipynb` | Summarise exported results and produce report figures/tables. |

Notebook 06 is run last despite its number because it consumes previous outputs. The supplied v2 summarises earlier stages and calibration; it still needs an added section to ingest Notebook 08 outputs. Until that update is made, use Notebook 08's tables and cap-comparison figure directly for the new report subsection.

The earlier `01_fd001_baseline.ipynb`, `02_test_evaluation.ipynb` and `03_model_improvements.ipynb` document initial single-subset development. An earlier `04_improved_test_evaluation_all_datasets.ipynb` is also a separate historical workflow; it is not interchangeable with `04_multi_dataset_improvements.ipynb`. Use the named workflow consistently and avoid mixing exports from different experiments.

## Validation and target definitions

Training/validation partitions are separated by engine, so the same engine contributes no rows to both partitions within a split. Temporal features use observations from the same engine's available history. Validation methods vary by stage: earlier feature experiments can score all validation cycles, whereas later robustness and cap experiments use truncated engine histories and score one final visible observation per engine.

A training cap changes the labels learned by the model:

`training_target = min(actual_RUL, training_cap)`

Values below the cap stay unchanged; larger values are clipped. This is not scaling RUL to 0–1. Uncapped training uses the original RUL labels.

Keep these evaluation objectives separate:

- **Uncapped RUL:** prediction error against actual remaining lifetime. This is the primary selection objective in Notebook 08.
- **Common-100 diagnostic:** performance within a common 100-cycle horizon. Notebook 08 clips both predictions and truth to 100 for this diagnostic. Earlier evaluation notebooks may cap truth only, so check formulas before comparing exported columns across stages.
- **Near-failure diagnostics:** errors for engines with actual RUL of 30 cycles or less.

MAE and RMSE are measured in cycles; smaller values are better. RMSE gives larger errors more weight. Higher R² indicates better fit relative to predicting the evaluated target's mean. Signed bias is `predicted_RUL - actual_RUL`: positive bias means overestimating remaining life. RMSE is not a percentage accuracy measure.

Hyperparameters, caps and calibration offsets are selected using training-engine validation. Official test labels do not participate in those selection procedures. However, official test results were already examined during project development, so later extensions are exploratory rather than evaluations on a previously untouched holdout.

## Training-cap experiment

Notebook 08 tests caps of **75, 100, 125, 150, 175 and 200 cycles**, plus **uncapped** training. Features, Random Forest settings, validation engines and observation points remain fixed across cap candidates.

Defaults use three engine-separated splits (seeds 42, 52 and 62), 20% held-out engines per split and 150 trees. Validation observation points are sampled between approximately 40% and 100% of the engine's lifetime. Trajectories are truncated before feature calculation. The cap with the lowest mean uncapped validation RMSE is selected separately for each dataset. Split standard deviations describe sensitivity, not independent confidence intervals.

The final comparison retrains the selected cap and a matched fixed-100 reference on all training engines. No calibration offset from Notebook 07 is automatically transferred to these new models.

Notebook 08 writes results under `results/training_cap_experiment/<configuration-signature>/`:

| Output | Contents |
| --- | --- |
| `config.json` | Experiment settings and Python/scikit-learn versions |
| `validation_by_split.csv` | Metrics for each dataset, cap and validation split |
| `validation_predictions.csv` | Per-engine validation predictions |
| `validation_summary.csv` | Mean validation metrics and RMSE split standard deviations |
| `selected_caps.csv` | Validation-selected cap for each dataset |
| `validation_cap_comparison.png` | Validation uncapped RMSE against training cap |
| `official_test_metrics.csv` | Test metrics for selected and fixed-100 models |
| `official_test_predictions.csv` | Per-engine test predictions for both methods |
| `official_test_comparison.csv` | Test RMSE and percentage reduction relative to fixed 100 |

Outputs are grouped by settings, but the signature does not fingerprint input data or source code. Rerunning identical settings overwrites that run's exports. Archive outputs with the code commit and environment used for a report result. If there are multiple runs, select one complete run directory explicitly rather than combining its files with another run.

### Observed cap-comparison results

The supplied experiment exports gave the following official-test results against uncapped RUL:

| Dataset | Selected training cap | Fixed-100 RMSE | Selected RMSE | RMSE reduction |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 100 | 21.05 | 21.05 | 0.00% |
| FD002 | 100 | 34.65 | 34.65 | 0.00% |
| FD003 | 175 | 21.33 | 26.05 | −22.10% |
| FD004 | 150 | 37.11 | 27.15 | 26.85% |

These are results for the matched Notebook 08 models, not every earlier feature-selected model. FD004 improved overall, but its near-failure MAE increased from 11.47 to 12.20 cycles. FD003 improved during validation but deteriorated on the official test set. Retain both outcomes; do not replace the selected cap based on test performance. Results may differ with package versions or changed settings.

## Package, tests and documentation

Reusable code lives in `src/cmapss/` (data loading and validation, leakage-safe
features, metrics, engine-separated validation, model pipeline). The notebooks
can import it:

```python
import sys; sys.path.insert(0, "../src")      # from the notebooks/ folder
from cmapss.features import build_features
from cmapss.metrics import score, capped_target
```

| Path | Contents |
| --- | --- |
| `src/cmapss/` | Documented package (NumPy-style docstrings, inline comments) |
| `docs/` | HTML API documentation generated by pdoc (open `docs/index.html`) |
| `tests/` | pytest suite using small synthetic data (no NASA files needed) |
| `tests/report/` | Test report: `test_report.html`, `test_report.xml`, `test_report.txt`, `coverage/` |

Run the tests and regenerate the report and docs from the repository root:

```bash
python -m pip install -r requirements-dev.txt numpy pandas scikit-learn
python -m pytest                        # unit tests + docstring examples
bash scripts/build_docs_and_test_report.sh
```

The tests check, among other things, that temporal features at cycle *t* do not
change when later cycles or other engines change (no leakage), that metrics match
hand calculations, that malformed inputs are rejected with a clear `ValueError`,
and that validation splits never share an engine.

## Reproducibility and checks

Notebook 08 records its settings, fixes random seeds, checks input dimensions and finite values, rejects duplicate engine-cycle pairs, asserts engine separation and checks test engine/label alignment. It exports predictions so metrics can be recalculated independently. The notebook was exercised end-to-end on synthetic data during preparation; the supplied NASA exports were produced by the project run. Reusable logic is also covered by the pytest suite in tests/ (see tests/report/).

## Computational cost and troubleshooting

The default cap experiment requires 84 validation fits (four subsets × three splits × seven targets), followed by full-training fits for the final comparisons. CPU and memory use depend on tree count, data size and available cores. The Random Forest uses `n_jobs=-1`.

- **Missing Python package:** install it in the environment selected by the notebook kernel; restart the kernel afterwards.
- **Missing NASA file:** check that all 12 `.txt` files are directly under `data/raw/` and that filenames match the expected case.
- **Missing exported CSV:** run the producing notebook first and check its output directory. Historical `iteration5_*.csv` files are produced by Notebook 05 despite its filename.
- **Multiple matching files in Notebook 06:** inspect the printed selected paths and remove ambiguity in its input search directories.
- **Long run or memory pressure:** try `DATASETS = ["FD001"]` or a smaller `N_ESTIMATORS` value for an exploratory run. Use consistent final settings across every candidate in the reported comparison.
- **Unexpected differences from report results:** check model settings, input files, selected run directory and package versions before interpreting the difference.

## Limitations and acknowledgement

C-MAPSS represents simulated degradation. Performance depends on operating conditions, fault modes, observation horizon and validation sampling. Capped predictions do not recover exact long remaining lifetimes, and improved overall error does not establish safe near-failure behaviour. The project does not model maintenance costs, interventions or calibrated failure probabilities.

Data acknowledgement: NASA Prognostics Center of Excellence, C-MAPSS Turbofan Engine Degradation Simulation Data Set. Background: Saxena, Goebel, Simon and Eklund (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*.

AI assistance was used in notebook and documentation development. The report's AI-use statement should describe the actual assistance and the student's own implementation, execution, checking and interpretation.Code is released under the MIT licence; NASA data terms apply separately. 
