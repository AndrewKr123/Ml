"""Checks for fold isolation and reproducible repeated partitions."""
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.check_cv_stability import make_splits, prepare_fold
from src.iris_preprocessing import load_raw


class StabilityTests(unittest.TestCase):
    def test_repeats_cover_train_once_and_exclude_holdout(self):
        raw = load_raw(raw_dir=None)
        train, holdout = train_test_split(raw, test_size=.2, stratify=raw.target, random_state=42)
        splits = make_splits(train, 'iris', 5, 3, 2026)
        again = make_splits(train, 'iris', 5, 3, 2026)
        self.assertEqual(len(splits), 15)
        for (fit, valid), (fit_again, valid_again) in zip(splits, again):
            self.assertFalse(set(fit) & set(valid))
            self.assertFalse(set(train.index[valid]) & set(holdout.index))
            np.testing.assert_array_equal(fit, fit_again)
            np.testing.assert_array_equal(valid, valid_again)
        for repeat in range(3):
            rows = np.concatenate([valid for _, valid in splits[repeat * 5:(repeat + 1) * 5]])
            np.testing.assert_array_equal(np.sort(rows), np.arange(len(train)))
        self.assertFalse(np.array_equal(splits[0][1], splits[5][1]))

    def test_titanic_validation_ages_do_not_change_training_imputation(self):
        frame = pd.DataFrame({
            'Name': ['A, Mr. A'] * 4, 'SibSp': [0] * 4, 'Parch': [0] * 4,
            'Cabin': [None] * 4, 'Fare': [10.] * 4, 'Age': [20., 40., np.nan, 100.],
            'Survived': [0, 1, 0, 1],
        })
        train = frame.iloc[:3]
        valid = frame.iloc[3:].copy()
        first_train, _ = prepare_fold('titanic', train, valid)
        valid['Age'] = np.nan
        second_train, second_valid = prepare_fold('titanic', train, valid)
        pd.testing.assert_frame_equal(first_train, second_train)
        self.assertEqual(second_train.Age.iloc[2], 30.)
        self.assertEqual(second_valid.Age.iloc[0], 30.)
        self.assertTrue(pd.isna(train.Age.iloc[2]))


if __name__ == '__main__':
    unittest.main()
