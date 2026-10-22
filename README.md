# San Francisco Rent Prediction

Explore San Francisco Rent Board housing records and estimate reported monthly rent using a decision-tree regression model. The target is the **midpoint of a reported rent range**, not an exact lease payment or a current market-rent quote.

This repository is a fork of the [DATA 230 group project](https://github.com/surabhi17182000/DATA_230_PROJ). It preserves the original notebooks, Tableau workbook, contribution history and group attribution. This fork adds the reproducible model evaluation below.

## Verified model results

The October 22, 2026 evaluation used 477,394 eligible records from a 551,551-row cleaned housing snapshot. Entire city blocks were assigned to either training or testing, so records from the same block could not appear on both sides.

| Measure | Decision tree | Training-median baseline |
| --- | ---: | ---: |
| Test mean absolute error (MAE) | **$640.18** | $940.34 |
| Test root mean squared error (RMSE) | $846.91 | $1,202.03 |
| Test R² | **0.5005** | -0.0062 |

The model reduced test MAE by **31.92%** compared with always predicting the training-set median rent midpoint. MAE is the average absolute prediction error in dollars. R² measures improvement in squared prediction error relative to predicting the test-set mean; the model explains about half of the variation in the held-out rent midpoints.

- Training: **382,459 records across 3,067 city blocks**.
- Testing: **94,935 records across 767 city blocks**.
- Split: 20% of eligible city blocks held out using a fixed random seed of 42.
- Model: `DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=42)`.
- Model settings were fixed before evaluating the test set.

The complete metrics, data checksum, input fields, exclusions and verification checks are in [results/decision_tree_metrics.json](results/decision_tree_metrics.json).

## Data and preparation

Source: [San Francisco Rent Board Housing Inventory](https://data.sf.gov/d/gdc7-dmcn).

The recorded evaluation uses `sf_rent_cleaned.csv`, exported by [Data_Analysis.ipynb](Data_Analysis.ipynb) from the October 4, 2026 inventory snapshot. Update the notebook's input CSV path for your machine, run the cleaning and feature-preparation cells, and execute its `to_csv` export. The notebook derives property age from the current year; use **2026** when reproducing this snapshot.

The large source and cleaned CSV files are kept outside Git. The cleaned file used for the recorded results has SHA-256:

```text
fd355af3485a18ddf9a410d5a90412ba129a41b002e211ec487ed76bc17d6464
```

The live source dataset changes over time. Exact score reproduction requires the same cleaned file, row order, dependency versions and settings; a newer download will produce a different checksum and may produce different scores. Every run records its own checksum.

The model uses housing size, bedrooms, bathrooms, property age, location, submission year, occupancy, submission type and utility-inclusion fields. It excludes identifiers, the raw rent range, both rent bounds and the target midpoint from its predictors. `block_num` is used only to form the train/test groups.

Records with missing or open-ended rent midpoints, zero rent, invalid bounds or missing block IDs are excluded. Numeric medians and categorical encodings are learned from the training partition only.

## Run the evaluation

Use Python **3.12**, the version used for the published result.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements.txt
python ml/evaluate_rent.py --data /path/to/sf_rent_cleaned.csv --output-dir artifacts/rent-evaluation
```

The output directory contains:

- `metrics.json`: scores, baseline, split, environment and integrity checks.
- `decision_tree_rent_midpoint.joblib`: the fitted preprocessing pipeline and tree.
- `holdout_predictions.npz`: held-out row positions, targets, predictions and baseline predictions.
- `split_indices.npz`: source row positions assigned to training and testing.

Generated models, predictions, datasets and local environments are ignored by Git. The committed result file is the verified reference run, separate from each new run's output.

## Check the code

```bash
python -m unittest discover -s tests -v
```

The checks cover target validity, city-block separation and the exclusion of rent answers and identifiers from prediction inputs. The evaluation also independently recomputes MAE and R² and confirms that reloading the saved model preserves its predictions.

## Interpretation and limits

The scores describe rent-band midpoint estimation for unseen city blocks within the observed **2022–2026** data. They do not measure future-year forecasting or current asking rents. Excluding missing, open-ended and zero-rent records also limits the population represented by the evaluation.

The original notebooks and Tableau workbook remain available for the exploratory analysis, housing trends and visualizations. The new evaluation is in [ml/evaluate_rent.py](ml/evaluate_rent.py).
