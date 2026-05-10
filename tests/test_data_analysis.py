import math
import unittest

import pandas as pd

from func.da.ab_testing import run_ab_test
from func.da.correlation import correlation_matrix
from func.da.statistics import describe_columns, dataset_overview, selectable_columns


class DataAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "revenue": [10.0, 14.0, 20.0, 22.0],
                "clicks": [1, 2, 3, 5],
                "segment": ["new", "returning", "new", "returning"],
            }
        )

    def test_dataset_overview_includes_first_five_rows_and_column_groups(self):
        overview = dataset_overview(self.frame)

        self.assertEqual(overview.row_count, 4)
        self.assertEqual(overview.column_count, 4)
        self.assertEqual(len(overview.preview), 4)
        self.assertEqual(overview.numeric_columns, ["revenue", "clicks"])
        self.assertEqual(overview.categorical_columns, ["group", "segment"])

    def test_selectable_columns_returns_all_columns_in_order(self):
        self.assertEqual(selectable_columns(self.frame), ["group", "revenue", "clicks", "segment"])

    def test_describe_columns_summarizes_selected_numeric_columns(self):
        summary = describe_columns(self.frame, ["revenue", "clicks"])

        self.assertAlmostEqual(summary["revenue"]["mean"], 16.5)
        self.assertAlmostEqual(summary["clicks"]["max"], 5.0)

    def test_correlation_matrix_limits_work_to_selected_columns(self):
        matrix = correlation_matrix(self.frame, ["revenue", "clicks"])

        self.assertEqual(list(matrix.columns), ["revenue", "clicks"])
        self.assertAlmostEqual(matrix.loc["revenue", "revenue"], 1.0)
        self.assertFalse(math.isnan(matrix.loc["revenue", "clicks"]))

    def test_ab_test_compares_named_control_and_treatment_groups(self):
        result = run_ab_test(
            self.frame,
            group_column="group",
            value_column="revenue",
            control_value="A",
            treatment_value="B",
        )

        self.assertEqual(result.control_count, 2)
        self.assertEqual(result.treatment_count, 2)
        self.assertAlmostEqual(result.control_mean, 12.0)
        self.assertAlmostEqual(result.treatment_mean, 21.0)
        self.assertAlmostEqual(result.mean_delta, 9.0)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
