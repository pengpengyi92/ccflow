from decimal import Decimal
from unittest import TestCase

import pandas as pd
import polars as pl
import pyarrow as pa
from packaging import version

from ccflow.utils.arrow import add_field_metadata, convert_decimal_types_to_float, convert_large_types, get_field_metadata


class TestArrowUtil(TestCase):
    def test_decimal_conversion_preserves_metadata(self):
        schema = pa.schema(
            [
                pa.field("price", pa.decimal128(8, 2), nullable=False, metadata={b"unit": b"USD"}),
                pa.field("symbol", pa.string(), metadata={b"role": b"identifier"}),
            ],
            metadata={b"dataset": b"prices", b"binary": b"\x00\xff"},
        )
        table = pa.Table.from_pydict({"price": [Decimal("1.25")], "symbol": ["A"]}, schema=schema)
        for target_type in (None, pa.float32()):
            with self.subTest(target_type=target_type):
                result = convert_decimal_types_to_float(table, target_type)
                expected = pa.schema(
                    [schema.field("price").with_type(target_type or pa.float64()), schema.field("symbol")],
                    metadata=schema.metadata,
                )
                self.assertTrue(result.schema.equals(expected, check_metadata=True))
                self.assertEqual(result.to_pydict(), {"price": [1.25], "symbol": ["A"]})
                self.assertTrue(table.schema.equals(schema, check_metadata=True))

    def test_large_conversion_preserves_metadata(self):
        item = pa.field("element", pa.int64(), nullable=False, metadata={b"unit": b"lots"})
        schema = pa.schema(
            [
                pa.field("sizes", pa.large_list(item), nullable=False, metadata={b"role": b"depth"}),
                pa.field("symbol", pa.large_string(), metadata={b"role": b"identifier"}),
                pa.field("payload", pa.large_binary(), metadata={b"encoding": b"opaque"}),
                pa.field("count", pa.int64(), metadata={b"unit": b"rows"}),
            ],
            metadata={b"dataset": b"quotes", b"binary": b"\x00\xff"},
        )
        table = pa.Table.from_pydict({"sizes": [[1, 2], []], "symbol": ["A", None], "payload": [b"x", None], "count": [2, 0]}, schema=schema)
        result = convert_large_types(table)
        expected = pa.schema(
            [
                schema.field("sizes").with_type(pa.list_(item)),
                schema.field("symbol").with_type(pa.string()),
                schema.field("payload").with_type(pa.binary()),
                schema.field("count"),
            ],
            metadata=schema.metadata,
        )
        self.assertTrue(result.schema.equals(expected, check_metadata=True))
        self.assertEqual(result.to_pydict(), table.to_pydict())
        self.assertTrue(table.schema.equals(schema, check_metadata=True))

    def test_conversions_preserve_metadata_without_matching_fields(self):
        schema = pa.schema([pa.field("count", pa.int64(), metadata={b"unit": b"rows"})], metadata={b"source": b"test"})
        table = pa.Table.from_pydict({"count": [1]}, schema=schema)
        for convert in (convert_decimal_types_to_float, convert_large_types):
            with self.subTest(convert=convert.__name__):
                result = convert(table)
                self.assertTrue(result.schema.equals(schema, check_metadata=True))
                self.assertEqual(result.to_pydict(), table.to_pydict())

    def test_conversions_preserve_pandas_index(self):
        for index in (pd.RangeIndex(10, 12, name="row"), pd.Index(["trade_a", "trade_b"], name="trade")):
            for convert in (convert_decimal_types_to_float, convert_large_types):
                with self.subTest(index=index, convert=convert.__name__):
                    frame = pd.DataFrame({"price": [10.0, 20.0]}, index=index)
                    result = convert(pa.Table.from_pandas(frame))
                    pd.testing.assert_frame_equal(result.to_pandas(), frame)

    def test_decimal_conversion_preserves_pandas_index_with_conversion(self):
        index = pd.Index(["trade_a", "trade_b"], name="trade")
        frame = pd.DataFrame({"price": [Decimal("1.25"), Decimal("2.50")]}, index=index)
        result = convert_decimal_types_to_float(pa.Table.from_pandas(frame))
        pd.testing.assert_frame_equal(result.to_pandas(), pd.DataFrame({"price": [1.25, 2.5]}, index=index))

    def test_convert_large_types(self):
        schema = pa.schema(
            [
                pa.field("int", pa.int16()),
                pa.field("str", pa.string()),
                pa.field("large_list_float", pa.large_list(pa.float32())),
                pa.field("large_list_str", pa.large_list(pa.string())),
                pa.field("large_string", pa.large_string()),
                pa.field("large_binary", pa.large_binary()),
            ]
        )
        t = pa.Table.from_pydict(
            {
                "int": [1, 2],
                "str": ["foo", "bar"],
                "large_list_float": [[1.0, 2.0], [3.0, 4.0]],
                "large_list_str": [["foo", "bar"], ["baz", "qux"]],
                "large_string": ["foo", "bar"],
                "large_binary": [b"abc", b"def"],
            },
            schema=schema,
        )
        out = convert_large_types(t)

        target_schema = pa.schema(
            [
                pa.field("int", pa.int16()),
                pa.field("str", pa.string()),
                pa.field("large_list_float", pa.list_(pa.float32())),
                pa.field("large_list_str", pa.list_(pa.string())),
                pa.field("large_string", pa.string()),
                pa.field("large_binary", pa.binary()),
            ]
        )
        self.assertEqual(out.schema, target_schema)
        pd.testing.assert_frame_equal(out.to_pandas(), t.to_pandas())

    def test_polars_large_list(self):
        """The function convert_large_types is necessary because polars uses large list.
        This test continues to verify that this is the case in polars. If it fails,
        it may be possible to remove the convert_large_types logic."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        if version.parse(pl.__version__) < version.parse("0.18"):
            t = df.select(pl.col("a").list()).to_arrow()
        else:
            # list() was renamed to implode().
            t = df.select(pl.col("a").implode()).to_arrow()

        # This is the schema that polars returns, which necessitates convert_large_types
        target_schema = pa.schema([pa.field("a", pa.large_list(pa.int64()))])
        self.assertEqual(t.schema, target_schema)

    def test_convert_decimal_types_to_float(self):
        table = pa.Table.from_pydict(
            {"sym": ["A", "B", "C"], "price": [Decimal("2.0"), Decimal("3.0"), Decimal("50.0")]},
            pa.schema([pa.field("sym", pa.string()), pa.field("price", pa.decimal128(8, 3))]),
        )

        converted_table = convert_decimal_types_to_float(table, pa.float64())
        target_schema = pa.schema([pa.field("sym", pa.string()), pa.field("price", pa.float64())])
        target_df = pd.DataFrame({"sym": ["A", "B", "C"], "price": [2.0, 3.0, 50.0]})

        self.assertEqual(converted_table.schema, target_schema)
        pd.testing.assert_frame_equal(target_df, converted_table.to_pandas())


class TestMetaData(TestCase):
    def test_add_field_metadata(self):
        t = pa.Table.from_pydict(
            {
                "int": [1, 2],
                "str": ["foo", "bar"],
                "list_float": [[1.0, 2.0], [3.0, 4.0]],
                "list_str": [["foo", "bar"], ["baz", "qux"]],
            },
        )
        self.assertEqual(get_field_metadata(t), {})
        metadata = {"int": {"foo": 4}, "str": {"bar": ["a", "b"], "baz": None}}
        t2 = add_field_metadata(t, metadata)
        self.assertEqual(t2.schema.field("int").metadata[b"foo"], b"4")
        self.assertEqual(t2.schema.field("str").metadata[b"bar"], b'["a","b"]')
        self.assertEqual(t2.schema.field("str").metadata[b"baz"], b"null")

        # Get it back in "normal" form
        self.assertDictEqual(get_field_metadata(t2), metadata)

    def test_add_field_metadata_preserves_schema_metadata(self):
        schema = pa.schema(
            [
                pa.field("price", pa.float64(), nullable=False, metadata={"description": '"closing price"'}),
                pa.field("symbol", pa.string(), metadata={"description": '"ticker"'}),
            ],
            metadata={b"dataset": b"prices", b"binary": b"\x00\xff"},
        )
        table = pa.Table.from_pydict({"price": [10.0, 20.0], "symbol": ["A", "B"]}, schema=schema)

        for metadata in ({"price": {"unit": "USD"}}, {}, {"missing": {"unit": "USD"}}):
            with self.subTest(metadata=metadata):
                result = add_field_metadata(table, metadata)

                self.assertEqual(result.schema.metadata, table.schema.metadata)
                self.assertTrue(result.schema.field("symbol").equals(schema.field("symbol"), check_metadata=True))
                self.assertFalse(result.schema.field("price").nullable)
                self.assertEqual(result.to_pydict(), table.to_pydict())

    def test_add_field_metadata_preserves_pandas_index(self):
        indexes = [
            pd.RangeIndex(10, 12, name="row"),
            pd.Index(["trade_a", "trade_b"], name="trade"),
            pd.MultiIndex.from_tuples([("A", 1), ("B", 2)], names=["symbol", "trade"]),
        ]
        for index in indexes:
            with self.subTest(index=index):
                frame = pd.DataFrame({"price": [10.0, 20.0]}, index=index)
                table = pa.Table.from_pandas(frame)

                result = add_field_metadata(table, {"price": {"unit": "USD"}})

                pd.testing.assert_frame_equal(result.to_pandas(), frame)

    def test_add_field_metadata_still_replaces_selected_field_metadata(self):
        schema = pa.schema([pa.field("price", pa.float64(), metadata={"description": '"closing price"'})])
        table = pa.Table.from_pydict({"price": [10.0]}, schema=schema)

        result = add_field_metadata(table, {"price": {"unit": "USD"}})

        self.assertEqual(result.schema.field("price").metadata, {b"unit": b'"USD"'})
        self.assertIsNone(result.schema.metadata)
