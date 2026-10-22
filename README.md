# San Francisco Rent Prediction

Analyze San Francisco housing records and estimate reported monthly rent with a decision-tree regression model. The project connects data cleaning, exploratory analysis, Tableau visualizations and a reproducible machine-learning evaluation.

**477,394 modeling records · 94,935 test records · 31.92% lower MAE than the median baseline**

[Explore the analysis](Data_Analysis.ipynb) · [View the model](ml/evaluate_rent.py) · [Inspect the results](results/decision_tree_metrics.json) · [Run it locally](#run-it-locally)

The prediction target is the **midpoint of a reported rent range**. It estimates rent-band midpoints in the housing inventory; it does not predict exact lease payments or current asking rents.

## Project overview

Public housing records contain rent ranges, inconsistent room categories, missing values and repeated submissions across years. This project prepares those records for analysis and tests whether housing characteristics and location can estimate reported rent.

The analysis explores:

- How rent varies with bedrooms, bathrooms, unit size and property age.
- How reported rent differs across neighborhoods and submission years.
- How occupancy and missing rent information affect the records available for analysis.
- How a decision tree performs on city blocks excluded from training.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn and Tableau. Joblib saves the fitted model and its preprocessing pipeline.

## Verified model results

The October 22, 2026 evaluation used **477,394 eligible records** from a **551,551-row** cleaned snapshot. The test set contains **94,935 records across 767 city blocks**.

| Test metric | Decision tree | Training-median baseline |
| --- | ---: | ---: |
| Mean absolute error (MAE) | **$640.18** | $940.34 |
| Root mean squared error (RMSE) | $846.91 | $1,202.03 |
| R² | **0.5005** | -0.0062 |

MAE is the average absolute error in dollars. The tree's MAE is **31.92% lower** than always predicting the training-set median rent midpoint. RMSE gives larger errors more weight. R² compares squared prediction error with predicting the test-set mean; this model explains about half of the variation in the held-out rent midpoints.

The [full result file](results/decision_tree_metrics.json) records the data checksum, features, exclusions, model settings, dependency versions and verification checks.

## Workflow

1. **Prepare the housing data.** Standardize bedroom and bathroom categories, parse rent and square-footage ranges, extract coordinates, convert dates and derive numeric features in [Data_Analysis.ipynb](Data_Analysis.ipynb).
2. **Explore the records.** Compare rent distributions, room counts, occupancy, neighborhoods and submission years. Export the cleaned data for modeling and Tableau.
3. **Split by city block.** Assign each block entirely to training or testing, so records from the same block cannot appear on both sides.
4. **Fit preprocessing and the model.** Learn missing-value replacements and categorical encodings from training rows, then fit the decision tree.
5. **Evaluate and verify.** Compare predictions with a median baseline, independently recalculate MAE and R², and confirm that the saved model produces identical predictions after reload.

### Model and validation design

- **Inputs:** housing size, bedrooms, bathrooms, property age, unit count, location, submission year, occupancy, submission type and utility-inclusion fields.
- **Excluded predictors:** record identifiers, the raw rent range, both rent bounds and the target midpoint. `block_num` forms the split groups only.
- **Training set:** 382,459 records across 3,067 city blocks.
- **Test split:** `GroupShuffleSplit` holds out 20% of eligible blocks with `random_state=42`; block overlap is zero.
- **Numeric preprocessing:** median imputation with missing-value indicators.
- **Categorical preprocessing:** fill missing values with `Missing`, then one-hot encode categories; unseen categories are ignored at prediction time.
- **Model:** `DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=42, criterion="squared_error")`.

All preprocessing is fitted on training rows only. The model settings were fixed before test evaluation, and no test-set tuning was performed.

## Repository guide

| File | Purpose |
| --- | --- |
| [Data_Analysis.ipynb](Data_Analysis.ipynb) | Data assessment, cleaning, feature preparation, exploratory charts and cleaned CSV export. |
| [Assignment6_Rent_EDA_Updated_Executed (2).ipynb](Assignment6_Rent_EDA_Updated_Executed%20%282%29.ipynb) | Additional cleaning validation and univariate, bivariate and multivariate analysis. |
| [grp_230.twbx](grp_230.twbx) | Packaged Tableau workbook from the group project; open with Tableau. |
| [RAPIDS_Housing_EDA.ipynb](RAPIDS_Housing_EDA.ipynb) | Optional GPU-based exploratory analysis using RAPIDS/cuDF. |
| [ml/evaluate_rent.py](ml/evaluate_rent.py) | Command-line training, holdout evaluation and integrity checks. |
| [ml/requirements.txt](ml/requirements.txt) | Pinned dependencies for the model evaluation. |
| [results/decision_tree_metrics.json](results/decision_tree_metrics.json) | Verified reference run, including metrics and dataset checksum. |
| [tests/test_evaluate_rent.py](tests/test_evaluate_rent.py) | Checks for valid targets, block separation and excluded predictors. |

The other cleaning notebooks retain the group's earlier analysis and preparation work.

## Data

Source: [San Francisco Rent Board Housing Inventory](https://data.sf.gov/d/gdc7-dmcn). The recorded run uses the **October 4, 2026** snapshot, with submission years **2022–2026**.

The evaluation requires `sf_rent_cleaned.csv`, exported by `Data_Analysis.ipynb`. It excludes records with missing or open-ended rent midpoints, zero or negative rent, invalid bounds or missing block IDs. In the reference snapshot, 70,562 records had missing/open-ended midpoints and 3,595 had nonpositive midpoints; none were excluded for invalid bounds or missing blocks.

Large source and cleaned CSV files are kept outside Git. The cleaned reference file has SHA-256:

```text
fd355af3485a18ddf9a410d5a90412ba129a41b002e211ec487ed76bc17d6464
```

The live dataset changes over time. Exact score reproduction requires the same cleaned file, row order, dependency versions and settings. Each new evaluation records its own checksum; a newer download may produce different scores.

## Run it locally

Use **Python 3.12** for the published evaluation. The model evaluation runs on the CPU; the optional RAPIDS notebook has separate GPU requirements.

### 1. Set up the environment

```bash
git clone https://github.com/ishuapurva1996/sf-rent-prediction.git
cd sf-rent-prediction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

### 2. Prepare the cleaned CSV

If you already have `sf_rent_cleaned.csv`, continue to the evaluation. Otherwise:

1. Download the housing inventory CSV from the [source dataset](https://data.sf.gov/d/gdc7-dmcn).
2. Open `Data_Analysis.ipynb` and update its input CSV path for your machine.
3. Run its cleaning and feature-preparation cells, followed by the `sf_rent.to_csv('sf_rent_cleaned.csv')` export.
4. Use **2026** when deriving property age to reproduce the reference snapshot. The notebook currently derives age from the current year.

To run the analysis notebook, install its notebook and plotting tools in the same environment:

```bash
python -m pip install jupyterlab matplotlib seaborn
jupyter lab Data_Analysis.ipynb
```

The pinned requirements cover the model evaluation; these additional tools support the analysis notebook.

### 3. Train and evaluate

```bash
python ml/evaluate_rent.py \
  --data /path/to/sf_rent_cleaned.csv \
  --output-dir artifacts/rent-evaluation
```

Replace the data path with the location of your cleaned CSV. The command prints the report and writes:

| Output | Contents |
| --- | --- |
| `metrics.json` | Scores, baseline, split, environment and integrity checks. |
| `decision_tree_rent_midpoint.joblib` | Fitted preprocessing pipeline and decision tree. |
| `holdout_predictions.npz` | Test-row positions, targets, predictions and baseline predictions. |
| `split_indices.npz` | Source-row positions assigned to training and testing. |

Generated models, predictions, datasets and local environments are ignored by Git. The committed result file remains the verified reference run.

### 4. Run the checks

```bash
python -m unittest discover -s tests -v
```

The four tests cover missing, zero and open-ended targets; invalid midpoint values; city-block separation; and the exclusion of rent answers and identifiers from prediction inputs. They use small in-memory fixtures and do not require the housing CSV. The full evaluation also checks source integrity, finite predictions, independent metric calculations and identical predictions after model reload.

## Interpretation and limitations

- **Range-based target:** rent midpoints approximate reported bands, so the scores do not establish accuracy for exact lease payments.
- **Spatial validation:** the test set measures performance on unseen blocks within the observed 2022–2026 records. It does not measure future-year forecasting or current market-rent prediction.
- **Eligible population:** excluding unknown, open-ended and nonpositive rents limits the records represented by the scores.

## Group project and fork contributions

This repository is a fork of the [DATA 230 group project](https://github.com/surabhi17182000/DATA_230_PROJ). The original notebooks, Tableau workbook and contribution history remain available with their group attribution.

This fork adds the decision-tree evaluation script, fixed block-based split, median baseline, verified result report, focused tests and reproduction instructions. These additions provide a runnable ML evaluation alongside the group's exploratory analysis.
