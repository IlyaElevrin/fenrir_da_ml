import unittest

import pandas as pd

from func.ml.models import cross_validate_model, create_model, train_model


class MachineLearningTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "length": [1.0, 1.5, 2.0, 9.0, 10.0, 11.0],
                "width": [0.8, 1.0, 1.2, 8.0, 9.0, 9.5],
                "segment": ["small", "small", "small", "large", "large", "large"],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )

    def test_create_model_supports_requested_model_builders(self):
        names = [
            "decision_tree",
            "random_forest",
            "linear_regression",
            "logistic_regression",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertIsNotNone(create_model(name, random_state=13))

    def test_create_model_applies_supported_parameters(self):
        model = create_model(
            "random_forest",
            random_state=13,
            model_params={"n_estimators": 11, "max_depth": 3, "min_samples_leaf": 2},
        )

        self.assertEqual(model.n_estimators, 11)
        self.assertEqual(model.max_depth, 3)
        self.assertEqual(model.min_samples_leaf, 2)

    def test_train_model_uses_selected_columns_and_returns_score(self):
        result = train_model(
            self.frame,
            model_name="decision_tree",
            feature_columns=["length", "width", "segment"],
            target_column="target",
            random_state=2,
            test_size=0.33,
        )

        self.assertGreaterEqual(result.score, 0.5)
        self.assertEqual(result.target_column, "target")
        self.assertIn("length", result.feature_names)
        self.assertTrue(any(name.startswith("segment_") for name in result.feature_names))

    def test_train_model_forwards_model_parameters(self):
        result = train_model(
            self.frame,
            model_name="decision_tree",
            feature_columns=["length", "width"],
            target_column="target",
            random_state=2,
            test_size=0.33,
            model_params={"max_depth": 1, "min_samples_leaf": 1},
        )

        self.assertEqual(result.model.max_depth, 1)

    def test_cross_validate_model_returns_fold_scores(self):
        frame = pd.DataFrame(
            {
                "length": [1.0, 1.2, 1.4, 1.6, 8.0, 8.2, 8.4, 8.6],
                "width": [0.8, 0.9, 1.0, 1.1, 7.8, 8.0, 8.1, 8.3],
                "segment": ["small", "small", "small", "small", "large", "large", "large", "large"],
                "target": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )

        result = cross_validate_model(
            frame,
            model_name="logistic_regression",
            feature_columns=["length", "width", "segment"],
            target_column="target",
            cv_folds=4,
            random_state=7,
            model_params={"max_iter": 200, "C": 1.0},
        )

        self.assertEqual(result.fold_count, 4)
        self.assertEqual(len(result.fold_scores), 4)
        self.assertEqual(result.metric, "accuracy")
        self.assertGreaterEqual(result.mean_score, 0.0)
        self.assertLessEqual(result.mean_score, 1.0)


if __name__ == "__main__":
    unittest.main()
