"""Unique experiment directories and common, backend-independent metadata."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from uuid import uuid4

from src.evaluation import save_json
from src.paths import ARTIFACTS


def create_run(dataset, model, config, *, root=ARTIFACTS):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
    directory = root / dataset / model / run_id
    directory.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ('numpy', 'pandas', 'scikit-learn', 'torch', 'joblib'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    save_json(config, directory / 'config.json')
    save_json({'schema_version': 1, 'dataset': dataset, 'model': model,
               'run_id': run_id, 'time_zone': 'UTC', 'versions': versions,
               'status': 'running'}, directory / 'run.json')
    return directory


def finish_run(directory, results, *, elapsed_seconds):
    import json
    save_json({'schema_version': 1, 'models': results,
               'elapsed_seconds': elapsed_seconds}, directory / 'run_metrics.json')
    manifest = json.loads((directory / 'run.json').read_text())
    manifest['status'] = 'completed'
    save_json(manifest, directory / 'run.json')


def classical_metrics(results):
    """Normalize classical results; retain missing train metrics as null."""
    return {name: {'train': result.train_metrics, 'validation': None,
                   'test': result.metrics,
                   'cv': None if result.search is None else {
                       'scoring': result.search.scoring,
                       'best_score': float(result.search.best_score_),
                       'best_params': result.search.best_params_}}
            for name, result in results.items()}
