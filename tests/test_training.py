"""Проверки контрактов обучения без полного перебора гиперпараметров.

Запуск: python -m unittest discover -s tests -v
"""

import contextlib
import io
import unittest
import json
import tempfile
from pathlib import Path

import joblib
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.experiments import house_prices, iris, titanic
from src.evaluation import classification_metrics, classification_metrics_multiclass, regression_metrics
from src.reporting.random_forest import save_random_forest_report
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
            iris.train_xgboost_cv,
            iris.train_random_forest_cv,
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
                    if train_model is iris.train_random_forest_cv:
                        self.assertEqual(result.train_metrics, classification_metrics_multiclass(
                            train["target"], result.model.predict(X_train), result.model.predict_proba(X_train)
                        ))
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
            # Дерево и бустинг не должны требовать результата обучения Ridge.
            tree = house_prices.train_decision_tree_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )
            boost = house_prices.train_lightgbm_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )
            forest = house_prices.train_random_forest_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )
            before_ridge = tree.model.predict(holdout).copy()
            ridge = house_prices.train_ridge_cv(
                train, holdout, preprocessor, y_train_log, cv, 42
            )

        self.assertFalse(hasattr(preprocessor, "transformers_"))
        self.assertIsNot(tree.model.named_steps["preprocess"], ridge.model.named_steps["preprocess"])
        np.testing.assert_array_equal(before_ridge, tree.model.predict(holdout))
        for result in (tree, boost, ridge, forest):
            self.assertIsNone(result.probabilities)
            np.testing.assert_allclose(result.predictions, np.expm1(result.model.predict(holdout)))
            expected_rmse = np.sqrt(np.mean((holdout["SalePrice"] - result.predictions) ** 2))
            self.assertAlmostEqual(result.metrics["rmse"], expected_rmse)

        self.assertEqual(forest.train_metrics, regression_metrics(
            train["SalePrice"], np.expm1(forest.model.predict(train))
        ))

    def test_titanic_forest_and_report_roundtrip(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "Pclass": rng.integers(1, 4, 80), "Age": rng.uniform(1, 80, 80),
            "Fare": rng.uniform(1, 5, 80), "FamilySize": rng.integers(1, 5, 80),
            "IsAlone": rng.integers(0, 2, 80), "Sex": ["male", "female"] * 40,
            "Embarked": "S", "Title": "Mr", "Deck": "Unknown",
            "Survived": [0, 1] * 40,
        })
        train, holdout = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.Survived)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        with patch.object(titanic, "GridSearchCV", side_effect=small_search), contextlib.redirect_stdout(io.StringIO()):
            forest = titanic.train_random_forest_cv(train, holdout, cv, 42)
        np.testing.assert_allclose(forest.probabilities, forest.model.predict_proba(holdout)[:, 1])
        self.assertEqual(forest.train_metrics, classification_metrics(
            train.Survived, forest.model.predict(train), forest.model.predict_proba(train)[:, 1]
        ))
        self.assertEqual(forest.metrics, classification_metrics(
            holdout.Survived, forest.predictions, forest.probabilities
        ))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "metrics.json").write_text('{"baseline": {"accuracy": 0.5}}')
            save_random_forest_report(
                forest, directory, test_size=0.2, random_state=42,
                test_fe=holdout, test_raw=pd.DataFrame({"PassengerId": np.arange(len(holdout))}),
            )
            payload = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(payload["baseline"], {"accuracy": 0.5})
            self.assertEqual(payload["random_forest"]["train"], forest.train_metrics)
            self.assertEqual(payload["random_forest"]["test"], forest.metrics)
            np.testing.assert_array_equal(
                joblib.load(directory / "random_forest.joblib").predict(holdout), forest.predictions
            )
            submission = pd.read_csv(directory / "random_forest_submission.csv")
            np.testing.assert_array_equal(submission.Survived, forest.predictions)


if __name__ == "__main__":
    unittest.main()
