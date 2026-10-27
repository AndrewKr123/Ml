"""Migration compatibility and split isolation."""
import tempfile
import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.datasets.iris import load_raw
from src.datasets.splits import split_train_validation_holdout
from src.reporting.runs import create_run


class LayoutTests(unittest.TestCase):
    def test_old_imports_resolve_to_same_implementation(self):
        from src.experiments import iris as old_experiment
        from src.classical import iris
        from src import iris_preprocessing
        from src.datasets import iris as dataset
        from scripts import train_iris
        from scripts.classical import train_iris as new_script
        self.assertIs(old_experiment, iris)
        self.assertIs(iris_preprocessing, dataset)
        self.assertIs(train_iris, new_script)

    def test_inner_validation_preserves_historical_holdout(self):
        raw = load_raw(raw_dir=None)
        train, validation, holdout = split_train_validation_holdout(raw, 'target')
        _, historical = train_test_split(raw, test_size=.2, stratify=raw.target, random_state=42)
        self.assertEqual(holdout.index.tolist(), historical.index.tolist())
        partitions = [set(frame.index) for frame in (train, validation, holdout)]
        self.assertEqual(set.union(*partitions), set(raw.index))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(partitions[a] & partitions[b])

    def test_run_directories_never_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = create_run('iris', 'knn', {}, root=Path(tmp))
            b = create_run('iris', 'knn', {}, root=Path(tmp))
            self.assertNotEqual(a, b)
            self.assertTrue((a / 'config.json').exists())



if __name__ == '__main__':
    unittest.main()
