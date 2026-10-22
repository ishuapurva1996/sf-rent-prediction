"""Checks for rent-target validity, spatial separation and target leakage."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from evaluate_rent import CATEGORICAL, NUMERIC, TARGET, build_model, prepare_data, split_by_block


def fixture(rows=400):
    frame = pd.DataFrame({name: np.ones(rows) for name in NUMERIC})
    for name in CATEGORICAL:
        frame[name] = "Included"
    frame["unique_id"] = [f"unit-{i}" for i in range(rows)]
    frame["block_num"] = np.arange(rows) % 20
    frame[TARGET] = np.where(np.arange(rows) % 2, 3000.0, 1000.0)
    frame["monthly_rent"] = frame[TARGET].astype(str)
    frame["lower_bound_monthly_rent"] = frame[TARGET] - 125
    frame["upper_bound_monthly_rent"] = frame[TARGET] + 125
    return frame


class RentEvaluationTests(unittest.TestCase):
    def test_unknown_zero_and_open_ended_targets_are_not_model_answers(self):
        frame = fixture(6)
        frame.loc[0, TARGET] = np.nan
        frame.loc[1, [TARGET, "lower_bound_monthly_rent", "upper_bound_monthly_rent"]] = 0
        frame.loc[2, [TARGET, "upper_bound_monthly_rent"]] = np.nan
        frame.loc[3, "block_num"] = np.nan
        clean, exclusions = prepare_data(frame)
        self.assertEqual(clean.index.tolist(), [4, 5])
        self.assertEqual(exclusions["missing_or_open_ended_target"], 2)
        self.assertEqual(exclusions["nonpositive_target_with_known_midpoint"], 1)

    def test_all_records_for_a_city_block_stay_on_one_side(self):
        clean, _ = prepare_data(fixture())
        train, test = split_by_block(clean)
        train_blocks = set(clean.iloc[train]["block_num"])
        test_blocks = set(clean.iloc[test]["block_num"])
        self.assertFalse(train_blocks & test_blocks)
        self.assertEqual(len(test_blocks), 4)
        self.assertEqual(len(train) + len(test), len(clean))
        self.assertFalse(set(train) & set(test))

    def test_rent_answers_and_identifiers_do_not_affect_predictions(self):
        clean, _ = prepare_data(fixture())
        model = build_model().fit(clean, clean[TARGET])
        before = model.predict(clean)
        changed = clean.copy()
        for name in [TARGET, "monthly_rent", "lower_bound_monthly_rent", "upper_bound_monthly_rent", "unique_id", "block_num"]:
            changed[name] = "unavailable-at-prediction-time"
        np.testing.assert_array_equal(model.predict(changed), before)

    def test_inconsistent_midpoint_is_rejected(self):
        frame = fixture()
        frame.loc[0, TARGET] = 100000
        with self.assertRaisesRegex(ValueError, "midpoint"):
            prepare_data(frame)


if __name__ == "__main__":
    unittest.main()
