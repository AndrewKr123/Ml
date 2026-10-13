
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

from src import house_prices_preprocessing as hp, iris_preprocessing as ip, titanic_preprocessing as tp
from src.evaluation import classification_metrics, classification_metrics_multiclass, regression_metrics, save_json

MODELS = {
    "iris": {"random_forest": "random_forest.joblib", "xgboost": "xgboost.joblib"},
    "titanic": {"random_forest": "random_forest.joblib", "catboost": "catboost.joblib"},
    "house_prices": {"random_forest": "random_forest.joblib", "lightgbm": "lightgbm.joblib"},
}
TARGETS = {"iris": "target", "titanic": "Survived", "house_prices": "SalePrice"}


def load_training_data(dataset):
    """Reconstruct the saved split; the holdout is never used for fitting or scoring."""
    directory = Path("models") / dataset
    metadata = json.loads((directory / "metrics.json").read_text())
    forest = metadata["random_forest"]
    for key in ("test_size", "random_state"):
        if metadata[key] != forest[key]:
            raise ValueError(f"{dataset}: forest and boosting used different {key}")
    raw = ip.load_raw() if dataset == "iris" else {"titanic": tp, "house_prices": hp}[dataset].load_raw()[0]
    train, holdout = train_test_split(
        raw, test_size=metadata["test_size"], random_state=metadata["random_state"],
        stratify=raw[TARGETS[dataset]] if dataset != "house_prices" else None,
    )
    return train, holdout.index.tolist(), metadata


def make_splits(frame, dataset, folds, repeats, seed):
    splitter = (RepeatedKFold if dataset == "house_prices" else RepeatedStratifiedKFold)(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )
    return list(splitter.split(frame, frame[TARGETS[dataset]]))


def prepare_fold(dataset, train, valid):
    """All learned imputation statistics and skew selection use only fold-training rows."""
    if dataset == "iris":
        return ip.prepare_features(train), ip.prepare_features(valid)
    if dataset == "titanic":
        return tp.impute_age_by_title(*map(tp.engineer_features, (train, valid)))
    train, valid = hp.fill_missing(*map(hp.engineer_and_clean, (train, valid)))
    return hp.log_transform_skewed(train, valid)


def evaluate_fold(dataset, frame, indices, templates, split_id, folds):
    train, valid = (frame.iloc[index] for index in indices)
    X_train, X_valid = prepare_fold(dataset, train, valid)
    target = TARGETS[dataset]
    y_train, y_valid = train[target], valid[target]
    rows = []
    for name, template in templates.items():
        model = clone(template)
        if dataset == "house_prices":
            # Rebuild column selection within this fold; TargetEncoder cross-fitting stays in Pipeline.
            model.set_params(preprocess=hp.build_preprocessor(hp.get_final_feature_columns(X_train)))
        model.fit(X_train, np.log1p(y_train) if dataset == "house_prices" else y_train)
        row = {"dataset": dataset, "model": name, "repeat": split_id // folds + 1,
               "fold": split_id % folds + 1, "n_train": len(train), "n_valid": len(valid)}
        for part, X, y in (("train", X_train, y_train), ("valid", X_valid, y_valid)):
            pred = model.predict(X).ravel()
            if dataset == "house_prices":
                metrics = regression_metrics(y, np.expm1(pred))
                metrics["rmse_log"] = float(root_mean_squared_error(np.log1p(y), pred))
            else:
                prob = model.predict_proba(X)
                metrics = (classification_metrics(y, pred, prob[:, 1]) if dataset == "titanic"
                           else classification_metrics_multiclass(y, pred, prob))
            row.update({f"{part}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return rows


def describe(values):
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def run(datasets, folds=5, repeats=5, seed=2026, jobs=4):
    for dataset in datasets:
        frame, holdout_indices, metadata = load_training_data(dataset)
        directory = Path("models") / dataset
        templates = {name: clone(joblib.load(directory / filename)) for name, filename in MODELS[dataset].items()}
        # Fix model randomness so the experiment isolates variation of CV partitions.
        for template in templates.values():
            params = template.get_params()
            if "model__n_jobs" in params:
                template.set_params(model__n_jobs=1)
            if "model__thread_count" in params:
                template.set_params(model__thread_count=1)
        splits = make_splits(frame, dataset, folds, repeats, seed)
        print(f"{dataset}: {len(frame)} rows, {folds} folds x {repeats} repeats, {len(templates)} models", flush=True)
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=jobs)(
                joblib.delayed(evaluate_fold)(dataset, frame, indices, templates, i, folds)
                for i, indices in enumerate(splits)
            )
        table = pd.DataFrame([row for result in results for row in result])
        table.to_csv(directory / "repeated_cv_folds.csv", index=False)
        metrics = [column for column in table if column.startswith(("train_", "valid_"))]
        repeat_means = table.groupby(["model", "repeat"])[metrics].mean().reset_index()
        repeat_means.to_csv(directory / "repeated_cv_repeats.csv", index=False)
        primary = {"iris": "accuracy", "titanic": "roc_auc", "house_prices": "rmse_log"}[dataset]
        paired = table.pivot(index=["repeat", "fold"], columns="model", values=f"valid_{primary}")
        boost_name = next(name for name in templates if name != "random_forest")
        delta = paired[boost_name] - paired["random_forest"]
        payload = {
            "protocol": "Fixed selected hyperparameters; repeated CV on original training split; fold-local preprocessing. Not nested CV.",
            "limitations": "Hyperparameters were selected on these same training rows. Fold scores overlap and are not independent; std is descriptive, not a confidence interval.",
            "folds": folds, "repeats": repeats, "cv_seed": seed,
            "original_split": {key: metadata[key] for key in ("test_size", "random_state")},
            "n_train_pool": len(frame), "excluded_holdout_indices": holdout_indices,
            "splits": [{"repeat": i // folds + 1, "fold": i % folds + 1,
                        "validation_indices": frame.index[valid].tolist()}
                       for i, (_, valid) in enumerate(splits)],
            "model_parameters": {name: model.named_steps["model"].get_params() for name, model in templates.items()},
            "models": {name: {
                "fold_scores": {metric: describe(group[metric]) for metric in metrics},
                "repeat_mean_scores": {metric: describe(repeat_means.loc[repeat_means.model == name, metric]) for metric in metrics},
            } for name, group in table.groupby("model")},
            "paired_comparison": {"metric": primary, "difference": f"{boost_name} minus random_forest",
                                  **describe(delta), "boost_wins": int(((delta < 0) if dataset == "house_prices" else (delta > 0)).sum()),
                                  "ties": int((delta == 0).sum()), "n_pairs": len(delta)},
        }
        save_json(payload, directory / "repeated_cv_summary.json")
        for name, stats in payload["models"].items():
            score = stats["fold_scores"][f"valid_{primary}"]
            print(f"  {name}: {primary}={score['mean']:.5f} +/- {score['std']:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 2:
        parser.error("--folds and --repeats must be at least 2")
    run(args.datasets, args.folds, args.repeats, args.seed, args.jobs)
