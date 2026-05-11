import math
import unittest

import numpy as np
import pandas as pd

from func.preprocess.preprocessing import (
    JOIN_METHODS,
    MISSING_STRATEGIES,
    drop_missing,
    group_table,
    impute_missing,
    join_tables,
    pivot_table,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "B"],
                "revenue": [10.0, np.nan, 20.0, np.nan, 30.0],
                "clicks": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )

    def test_strategy_and_join_constants_expose_required_options(self):
        for strategy in ("drop", "mean", "median", "knn"):
            self.assertIn(strategy, MISSING_STRATEGIES)
        for method in ("inner", "left", "right", "outer"):
            self.assertIn(method, JOIN_METHODS)

    def test_drop_missing_removes_rows(self):
        cleaned = drop_missing(self.frame)
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(len(cleaned), 2)

    def test_impute_missing_with_mean_fills_numeric_columns(self):
        filled = impute_missing(self.frame, strategy="mean")
        self.assertFalse(filled[["revenue", "clicks"]].isna().any().any())
        self.assertAlmostEqual(filled.loc[1, "revenue"], 20.0)

    def test_impute_missing_with_median_fills_numeric_columns(self):
        filled = impute_missing(self.frame, strategy="median")
        self.assertFalse(filled[["revenue", "clicks"]].isna().any().any())
        self.assertAlmostEqual(filled.loc[1, "revenue"], 20.0)

    def test_impute_missing_with_knn_fills_numeric_columns(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")

        filled = impute_missing(self.frame, strategy="knn", knn_neighbors=2)
        self.assertFalse(filled[["revenue", "clicks"]].isna().any().any())
        self.assertFalse(math.isnan(filled.loc[3, "revenue"]))

    def test_pivot_table_returns_aggregated_frame(self):
        result = pivot_table(
            self.frame.dropna(),
            index=["group"],
            values=["revenue", "clicks"],
            aggfunc="mean",
        )
        self.assertIn("group", result.columns)
        self.assertIn("revenue", result.columns)

    def test_join_tables_supports_inner_join(self):
        left = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        right = pd.DataFrame({"id": [2, 3, 4], "value": [20, 30, 40]})

        result = join_tables(left, right, how="inner", on="id")
        self.assertEqual(sorted(result["id"].tolist()), [2, 3])
        self.assertIn("name", result.columns)
        self.assertIn("value", result.columns)

    def test_group_table_aggregates_numeric_columns(self):
        result = group_table(self.frame.dropna(), by=["group"], aggfunc="sum")
        self.assertIn("group", result.columns)
        self.assertIn("revenue", result.columns)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
