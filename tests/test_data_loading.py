import os
import sqlite3
import tempfile
import unittest

import pandas as pd

from func.io.loaders import (
    SUPPORTED_FILE_EXTENSIONS,
    list_database_tables,
    load_database_table,
    load_table_from_file,
)


class DataLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fenrir-tests-")
        self.frame = pd.DataFrame(
            {"group": ["A", "B", "C"], "value": [1, 2, 3], "name": ["x", "y", "z"]}
        )

    def tearDown(self) -> None:
        for root, _dirs, files in os.walk(self.tmpdir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.tmpdir)

    def test_supported_extensions_cover_required_formats(self):
        for extension in (".csv", ".xlsx", ".txt", ".parquet"):
            with self.subTest(extension=extension):
                self.assertIn(extension, SUPPORTED_FILE_EXTENSIONS)

    def test_load_csv_returns_dataframe(self):
        path = os.path.join(self.tmpdir, "data.csv")
        self.frame.to_csv(path, index=False)

        loaded = load_table_from_file(path)
        self.assertEqual(list(loaded.columns), list(self.frame.columns))
        self.assertEqual(len(loaded), 3)

    def test_load_txt_autodetects_delimiter(self):
        path = os.path.join(self.tmpdir, "data.txt")
        self.frame.to_csv(path, sep="\t", index=False)

        loaded = load_table_from_file(path)
        self.assertEqual(list(loaded.columns), list(self.frame.columns))
        self.assertEqual(len(loaded), 3)

    def test_load_parquet_round_trip(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not installed")
        path = os.path.join(self.tmpdir, "data.parquet")
        self.frame.to_parquet(path)

        loaded = load_table_from_file(path)
        self.assertEqual(list(loaded.columns), list(self.frame.columns))
        self.assertEqual(len(loaded), 3)

    def test_load_unsupported_extension_raises(self):
        path = os.path.join(self.tmpdir, "data.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        with self.assertRaises(ValueError):
            load_table_from_file(path)

    def test_load_database_table_from_sqlite(self):
        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            self.skipTest("SQLAlchemy is not installed")

        path = os.path.join(self.tmpdir, "data.sqlite")
        connection = f"sqlite:///{path}"
        with sqlite3.connect(path) as raw:
            self.frame.to_sql("clients", raw, index=False)
            self.frame.to_sql("orders", raw, index=False)

        tables = list_database_tables(connection)
        self.assertEqual(sorted(tables), ["clients", "orders"])

        loaded = load_database_table(connection, "clients")
        self.assertEqual(len(loaded), 3)
        self.assertIn("value", loaded.columns)


if __name__ == "__main__":
    unittest.main()
