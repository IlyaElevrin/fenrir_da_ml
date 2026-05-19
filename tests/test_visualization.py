import unittest

import pandas as pd

from func.da.visualization import (
    VISUALIZATION_METHODS,
    VisualizationConfig,
    validate_chart_config,
)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "revenue": [10.0, 12.0, 18.0],
                "clicks": [1, 2, 3],
                "segment": ["A", "B", "A"],
            }
        )

    def test_visualization_methods_cover_requested_chart_types(self):
        for chart_type in ("line", "bar", "pie", "heatmap", "scatter", "histogram", "box"):
            self.assertIn(chart_type, VISUALIZATION_METHODS)

    def test_validate_chart_config_accepts_supported_columns(self):
        config = validate_chart_config(
            self.frame,
            VisualizationConfig("scatter", x_column="clicks", y_column="revenue"),
        )

        self.assertEqual(config.chart_type, "scatter")

    def test_validate_chart_config_rejects_missing_column(self):
        with self.assertRaises(ValueError):
            validate_chart_config(
                self.frame,
                VisualizationConfig("line", x_column="date", y_column="missing"),
            )


if __name__ == "__main__":
    unittest.main()
