"""Проверки контрактов обучения без полного перебора гиперпараметров.

Запуск: python -m unittest discover -s tests -v
"""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.experiments import house_prices, iris
from src.experiments.result import TrainingResult
from src.iris_preprocessing import load_raw, prepare_features


def small_search(estimator, param_grid, **kwargs):
    """Настоящий CV и refit, но только один кандидат из каждой сетки."""
    kwargs["n_jobs"] = 1
    params = {key: [values[0]] for key, values in param_grid.items()}
    return GridSearchCV(estimator, params, **kwargs)


class TrainingTests(unittest.TestCase):
    def test_iris_results_match_fitted_models(self):
        # Встроенный датасет: тест не зависит от локальных CSV или сети.
        frame = load_raw(raw_dir=None)
        train, holdout = train_test_split(
            frame, test_size=0.2, stratify=frame["target"], random_state=42
        )
        X_train, X_holdout = prepare_features(train), prepare_features(holdout)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        functions = [
            iris.train_baseline,
            iris.train_logistic_cv,
            iris.train_svm_linear_cv,
            iris.train_svm_rbf_cv,
            iris.train_decision_tree_cv,
        ]
        with patch.object(iris, "GridSearchCV", side_effect=small_search):
            for train_model in functions:
                with self.subTest(model=train_model.__name__), contextlib.redirect_stdout(io.StringIO()):
                    kwargs = {"random_state": 42}
                    if train_model is not iris.train_baseline:
                        kwargs["cv"] = cv
                    result = train_model(
                        X_train, train["target"], X_holdout, holdout["target"], **kwargs
                    )
                    self.assertIsInstance(result, TrainingResult)
                    np.testing.assert_array_equal(result.predictions, result.model.predict(X_holdout))
                    np.testing.assert_allclose(result.probabilities, result.model.predict_proba(X_holdout))
                    self.assertEqual(result.probabilities.shape, (len(holdout), 3))
                    self.assertAlmostEqual(
                        result.metrics["accuracy"], np.mean(result.predictions == holdout["target"])
                    )
                    if train_model is iris.train_baseline:
                        self.assertIsNone(result.search)
                    else:
                        self.assertIs(result.model, result.search.best_estimator_)

    def test_house_tree_runs_before_ridge_and_preserves_shared_preprocessor(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"area": rng.uniform(20, 200, 60)})
        frame["SalePrice"] = np.exp(10 + frame["area"] / 200)
        train, holdout = train_test_split(frame, test_size=0.2, random_state=42)
        preprocessor = ColumnTransformer([("num", StandardScaler(), ["area"])])
        y_train_log = np.log1p(train["SalePrice"])
        cv = KFold(n_splits=2, shuffle=True, random_state=42)

        with patch.object(house_prices, "GridSearchCV", side_effect=small_search), contextlib.redirect_stdout(io.StringIO()):
            # Дерево не должно требовать результата предварительного обучения Ridge.
            tree = house_prices.train_decision_tree_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )
            before_ridge = tree.model.predict(holdout).copy()
            ridge = house_prices.train_ridge_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )

        self.assertFalse(hasattr(preprocessor, "transformers_"))
        self.assertIsNot(tree.model.named_steps["preprocess"], ridge.model.named_steps["preprocess"])
        np.testing.assert_array_equal(before_ridge, tree.model.predict(holdout))
        for result in (tree, ridge):
            self.assertIsNone(result.probabilities)
            np.testing.assert_allclose(result.predictions, np.expm1(result.model.predict(holdout)))
            expected_rmse = np.sqrt(np.mean((holdout["SalePrice"] - result.predictions) ** 2))
            self.assertAlmostEqual(result.metrics["rmse"], expected_rmse)


if __name__ == "__main__":
    unittest.main()
