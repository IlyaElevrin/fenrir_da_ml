import unittest

import pandas as pd

from func.ml.models import create_model, train_model


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


if __name__ == "__main__":
    unittest.main()
