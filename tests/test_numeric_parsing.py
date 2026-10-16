from __future__ import annotations

import unittest

import pandas as pd

from clean_catalogs import trim_cell
from matching_model import (
    DATA_DIR,
    is_bounded_numeric_value,
    infer_features,
    load_catalog,
    parse_numeric_values,
    recommend,
)


class NumericParsingTests(unittest.TestCase):
    def test_supported_complete_values(self) -> None:
        cases = {
            "24": (24.0,),
            "\t24/-18\xa0": (24.0, 18.0),
            "\u00b120": (20.0,),
            "+20/-16": (20.0, 16.0),
            "\u00b110/\u00b112": (10.0, 12.0),
            "5e-05": (5e-05,),
            "\u00b130KV": (30.0,),
            ".5": (0.5,),
            "5.": (5.0,),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_numeric_values(value), expected)

    def test_malformed_values_are_not_partially_parsed(self) -> None:
        values = (
            ">1.2",
            "2\t4/-18",
            "abc24",
            "24abc",
            "1-2",
            "1..2",
            "24/18/12",
            "1,000",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(parse_numeric_values(value), ())

    def test_unsigned_and_symmetric_plus_minus_values_are_numerically_equal(self) -> None:
        self.assertEqual(parse_numeric_values("20"), parse_numeric_values("±20"))

    def test_bounded_value_is_identified_but_not_parsed_as_exact(self) -> None:
        self.assertTrue(is_bounded_numeric_value(">1.2"))
        self.assertEqual(parse_numeric_values(">1.2"), ())

    def test_trim_only_removes_outer_noise(self) -> None:
        self.assertEqual(trim_cell("\u200b\t24/-18\xa0"), "24/-18")
        self.assertEqual(trim_cell("2\t4/-18"), "2\t4/-18")


class CatalogRegressionTests(unittest.TestCase):
    def test_power_mosfet_unsigned_and_plus_minus_vgs_rank_identically(self) -> None:
        path = next(DATA_DIR.glob("*power-mosfets*.csv"))
        catalog = load_catalog(path)
        vgs_column = "Gate-Source Voltage VGS (V)"
        source = catalog[catalog[vgs_column].map(bool)].iloc[0]
        base_query = {"Package Type": source["Package Type"]}

        unsigned = recommend(
            catalog,
            {**base_query, vgs_column: "20"},
            top_k=20,
        )
        symmetric = recommend(
            catalog,
            {**base_query, vgs_column: "±20"},
            top_k=20,
        )

        pd.testing.assert_frame_equal(unsigned, symmetric)

    def test_active_and_preferred_statuses_do_not_create_risk_warnings(self) -> None:
        catalog = pd.DataFrame(
            {
                "Product": ["ACTIVE-PART", "PREFERRED-PART", "ALLOCATION-PART"],
                "Manufacture": ["MCC", "MCC", "MCC"],
                "Status": ["Active", "  PREFERRED  ", "Allocation"],
                "Package Type": ["SOT-23", "SOT-23", "SOT-23"],
                "Test Parameter (V)": [5.0, 5.0, 5.0],
            }
        )

        results = recommend(
            catalog,
            {"Package Type": "SOT-23", "Test Parameter (V)": 5.0},
            top_k=3,
        ).set_index("Product")

        self.assertEqual(results.loc["ACTIVE-PART", "风险提示"], "无")
        self.assertEqual(results.loc["PREFERRED-PART", "风险提示"], "无")
        self.assertEqual(results.loc["ACTIVE-PART", "可信度"], "高")
        self.assertEqual(results.loc["PREFERRED-PART", "可信度"], "高")
        self.assertIn("状态为 Allocation", results.loc["ALLOCATION-PART", "风险提示"])

    def test_recommendation_includes_parameters_missing_from_query(self) -> None:
        path = next(DATA_DIR.glob("*power-mosfets*.csv"))
        catalog = load_catalog(path)
        numeric, categorical = infer_features(catalog)
        source = catalog.iloc[0]
        active_column = next(
            column for column in numeric
            if parse_numeric_values(source[column])
        )
        results = recommend(
            catalog,
            {
                "Package Type": source["Package Type"],
                active_column: source[active_column],
            },
            top_k=1,
        )

        self.assertFalse(results.empty)
        expected_parameters = {"Package Type", *categorical, *numeric}
        self.assertTrue(expected_parameters.issubset(results.columns))

    def test_invalid_query_is_reported_even_when_hard_constraints_have_no_match(self) -> None:
        path = next(DATA_DIR.glob("*power-mosfets*.csv"))
        catalog = load_catalog(path)
        query = {
            "Package Type": "不存在的封装",
            "Drain-Source Voltage VDS (V)": "2\t4",
        }
        with self.assertRaisesRegex(ValueError, "Drain-Source Voltage VDS"):
            recommend(catalog, query)

    def test_mcgd016np04l_id_is_clean_and_pair_parseable(self) -> None:
        path = next(DATA_DIR.glob("*power-mosfets*.csv"))
        catalog = load_catalog(path)
        row = catalog[catalog["Product"] == "MCGD016NP04L"].iloc[0]
        column = "Drain Current ID (A)"
        self.assertEqual(row[column], "24/-18")
        self.assertEqual(parse_numeric_values(row[column]), (24.0, 18.0))

    def test_known_trr_bound_remains_raw_and_is_not_exact(self) -> None:
        path = next(DATA_DIR.glob("*standard-recovery*.csv"))
        catalog = load_catalog(path)
        row = catalog[catalog["Product"] == "R4000GPS"].iloc[0]
        column = next(name for name in catalog.columns if name.startswith("TRR"))
        self.assertEqual(row[column], ">1.2")
        self.assertEqual(parse_numeric_values(row[column]), ())


if __name__ == "__main__":
    unittest.main()
