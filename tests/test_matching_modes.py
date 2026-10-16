from __future__ import annotations

import math
import unittest

import pandas as pd

from matching_model import (
    BALANCED_MODE,
    DATA_DIR,
    STRICT_MODE,
    _component_continuous_rule_score,
    infer_features,
    load_catalog,
    parse_numeric_values,
    recommend,
)
from matching_rules import (
    AT_LEAST,
    AT_MOST,
    CLOSE,
    EXACT,
    PRODUCT_RULES,
    ParameterRule,
    continuous_rule_score,
    violation_ratio,
)


class ContinuousRuleTests(unittest.TestCase):
    def test_directional_violation_is_zero_when_rule_passes(self) -> None:
        at_most = ParameterRule("max", (), AT_MOST)
        at_least = ParameterRule("min", (), AT_LEAST)

        self.assertEqual(violation_ratio(at_most, 100.0, 90.0), 0.0)
        self.assertEqual(violation_ratio(at_most, 100.0, 100.0), 0.0)
        self.assertEqual(violation_ratio(at_least, 100.0, 110.0), 0.0)
        self.assertEqual(violation_ratio(at_least, 100.0, 100.0), 0.0)

    def test_directional_violation_and_score_are_continuous(self) -> None:
        at_most = ParameterRule("max", (), AT_MOST)
        at_least = ParameterRule("min", (), AT_LEAST)

        self.assertAlmostEqual(violation_ratio(at_most, 100.0, 110.0), 0.10)
        self.assertAlmostEqual(violation_ratio(at_least, 100.0, 90.0), 0.10)
        self.assertAlmostEqual(
            continuous_rule_score(at_most, 100.0, 110.0),
            1.0 / 1.1,
        )
        self.assertGreater(
            continuous_rule_score(at_most, 100.0, 105.0),
            continuous_rule_score(at_most, 100.0, 120.0),
        )

    def test_close_only_penalizes_amount_outside_tolerance(self) -> None:
        close = ParameterRule("close", (), CLOSE, tolerance=0.20)

        self.assertEqual(violation_ratio(close, 100.0, 120.0), 0.0)
        self.assertAlmostEqual(violation_ratio(close, 100.0, 130.0), 0.10)

    def test_exact_and_zero_query_remain_finite(self) -> None:
        exact = ParameterRule("exact", (), EXACT)
        at_most = ParameterRule("max", (), AT_MOST)

        self.assertEqual(violation_ratio(exact, 100.0, 100.0), 0.0)
        self.assertAlmostEqual(violation_ratio(exact, 100.0, 101.0), 0.01)
        zero_score = continuous_rule_score(at_most, 0.0, 1.0)
        self.assertTrue(math.isfinite(zero_score))
        self.assertGreaterEqual(zero_score, 0.0)
        self.assertLess(zero_score, 1.0)

    def test_pair_uses_the_worst_component(self) -> None:
        at_most = ParameterRule("max", (), AT_MOST, balanced_tradeoff=True)
        score = _component_continuous_rule_score(
            at_most,
            (100.0, 100.0),
            (100.0, 150.0),
        )
        self.assertAlmostEqual(score, 1.0 / 1.5)


class RuleClassificationTests(unittest.TestCase):
    def test_only_performance_type_critical_rules_are_tradeable(self) -> None:
        expected = {
            "mosfet": {"VGS(th)", "RDS(on)", "Qg", "Trr", "Qrr"},
            "switching_diode": {"Trr"},
            "sbd": {"VF", "IR"},
            "fred": {"Trr"},
            "transistor": {"fT", "HFE range"},
            "zener": {"IZT"},
            "tvs": {"CJ"},
            "esd": {"CJ"},
        }
        actual = {
            family: {
                rule.name
                for rule in rules
                if rule.critical and rule.balanced_tradeoff
            }
            for family, rules in PRODUCT_RULES.items()
        }
        self.assertEqual(actual, expected)

    def test_safety_and_function_rules_are_not_tradeable(self) -> None:
        protected = {
            "mosfet": {"VDS", "ID"},
            "zener": {"VZ"},
            "tvs": {"VRWM", "VC", "IPP"},
            "esd": {"VRWM", "VC", "IPP"},
        }
        for family, names in protected.items():
            rules = {rule.name: rule for rule in PRODUCT_RULES[family]}
            for name in names:
                with self.subTest(family=family, name=name):
                    self.assertTrue(rules[name].critical)
                    self.assertFalse(rules[name].balanced_tradeoff)

    def test_tradeoff_limits_are_positive_and_parameter_specific(self) -> None:
        tradeoff_rules = [
            rule
            for rules in PRODUCT_RULES.values()
            for rule in rules
            if rule.balanced_tradeoff
        ]
        self.assertTrue(all(rule.balanced_limit > 0 for rule in tradeoff_rules))
        self.assertGreater(
            next(rule for rule in PRODUCT_RULES["sbd"] if rule.name == "IR").balanced_limit,
            next(rule for rule in PRODUCT_RULES["sbd"] if rule.name == "VF").balanced_limit,
        )


class MatchingModeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _mosfet_catalog() -> pd.DataFrame:
        catalog = pd.DataFrame({
            "Manufacture": ["MCC", "MCC", "MCC"],
            "Product": ["TRADEOFF", "STRICT-PASS", "SAFETY-FAIL"],
            "Status": ["Active", "Active", "Active"],
            "Compliance": ["R H", "R H", "R H"],
            "Channel": ["N", "N", "N"],
            "Package Type": ["DFN", "DFN", "DFN"],
            "Drain-Source Voltage VDS (V)": [40.0, 40.0, 39.9],
            "Drain-Source On-Resistance RDS(ON) Max @VGS=10V (Ω)": [
                3.35, 3.30, 3.30,
            ],
            "Junction Temperature Tj [max] (°C)": [150.0, 30.0, 150.0],
        })
        catalog.attrs["source_name"] = "MCC_DataExport_power-mosfets.csv"
        return catalog

    @staticmethod
    def _query() -> dict[str, object]:
        return {
            "Package Type": "DFN",
            "Channel": "N",
            "Drain-Source Voltage VDS (V)": 40.0,
            "Drain-Source On-Resistance RDS(ON) Max @VGS=10V (Ω)": 3.30,
            "Junction Temperature Tj [max] (°C)": 150.0,
        }

    def test_default_remains_strict_and_strict_pass_is_first(self) -> None:
        default = recommend(self._mosfet_catalog(), self._query(), top_k=3)
        explicit = recommend(
            self._mosfet_catalog(), self._query(), top_k=3, mode=STRICT_MODE,
        )

        self.assertEqual(default["Product"].tolist(), explicit["Product"].tolist())
        self.assertEqual(default.iloc[0]["Product"], "STRICT-PASS")
        tradeoff = default.set_index("Product").loc["TRADEOFF"]
        self.assertEqual(tradeoff["关键参数检查"], "不满足")
        self.assertAlmostEqual(float(tradeoff["关键规则符合度"]), 60.0)

    def test_balanced_mode_allows_small_tradeoff_to_compete_by_closeness(self) -> None:
        results = recommend(
            self._mosfet_catalog(), self._query(), top_k=3, mode=BALANCED_MODE,
        )

        self.assertEqual(results.iloc[0]["Product"], "TRADEOFF")
        tradeoff = results.set_index("Product").loc["TRADEOFF"]
        self.assertEqual(tradeoff["关键参数检查"], "有条件推荐")
        self.assertIn("超出规则 1.5%", tradeoff["关键参数问题"])
        self.assertGreater(float(tradeoff["关键规则符合度"]), 95.0)
        self.assertLess(float(tradeoff["关键规则符合度"]), 100.0)

    def test_balanced_mode_does_not_relax_safety_rule(self) -> None:
        results = recommend(
            self._mosfet_catalog(), self._query(), top_k=3, mode=BALANCED_MODE,
        )
        safety = results.set_index("Product").loc["SAFETY-FAIL"]

        self.assertEqual(results.iloc[-1]["Product"], "SAFETY-FAIL")
        self.assertEqual(safety["关键参数检查"], "不满足")
        self.assertIn("不可权衡", safety["风险提示"])

    def test_balanced_mode_rejects_tradeoff_beyond_configured_limit(self) -> None:
        catalog = self._mosfet_catalog()
        severe = catalog.iloc[0].copy()
        severe["Product"] = "TRADEOFF-OVERLIMIT"
        severe[
            "Drain-Source On-Resistance RDS(ON) Max @VGS=10V (Ω)"
        ] = 6.60
        catalog = pd.concat([catalog, severe.to_frame().T], ignore_index=True)
        catalog.attrs["source_name"] = "MCC_DataExport_power-mosfets.csv"

        results = recommend(
            catalog, self._query(), top_k=4, mode=BALANCED_MODE,
        ).set_index("Product")
        overlimit = results.loc["TRADEOFF-OVERLIMIT"]

        self.assertEqual(overlimit["关键参数检查"], "不满足")
        self.assertIn("可权衡上限 20%", overlimit["关键参数问题"])
        self.assertIn("超过平衡模式上限", overlimit["风险提示"])

    def test_missing_tradeoff_parameter_remains_pending_in_balanced_mode(self) -> None:
        catalog = self._mosfet_catalog()
        missing = catalog.iloc[0].copy()
        missing["Product"] = "MISSING-RDS"
        missing[
            "Drain-Source On-Resistance RDS(ON) Max @VGS=10V (Ω)"
        ] = float("nan")
        catalog = pd.concat([catalog, missing.to_frame().T], ignore_index=True)
        catalog.attrs["source_name"] = "MCC_DataExport_power-mosfets.csv"

        results = recommend(
            catalog, self._query(), top_k=4, mode=BALANCED_MODE,
        ).set_index("Product")
        pending = results.loc["MISSING-RDS"]

        self.assertEqual(pending["关键参数检查"], "待人工确认")
        self.assertIn("RDS", pending["关键参数问题"])
        self.assertAlmostEqual(float(pending["关键规则符合度"]), 85.0)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知匹配模式"):
            recommend(
                self._mosfet_catalog(), self._query(), mode="anything-else",
            )

    def test_categories_without_direction_rules_are_mode_neutral(self) -> None:
        catalog = pd.DataFrame({
            "Manufacture": ["MCC", "MCC"],
            "Product": ["NEAR", "FAR"],
            "Status": ["Active", "Active"],
            "Package Type": ["PKG", "PKG"],
            "Test Parameter (V)": [10.0, 20.0],
        })
        query = {"Package Type": "PKG", "Test Parameter (V)": 10.0}

        strict = recommend(catalog, query, top_k=2, mode=STRICT_MODE)
        balanced = recommend(catalog, query, top_k=2, mode=BALANCED_MODE)
        self.assertEqual(strict["Product"].tolist(), balanced["Product"].tolist())
        self.assertEqual(
            strict["综合得分"].tolist(), balanced["综合得分"].tolist(),
        )

    def test_every_clean_catalog_runs_in_both_modes(self) -> None:
        catalog_paths = sorted(DATA_DIR.glob("*.csv"))
        self.assertTrue(catalog_paths)

        for path in catalog_paths:
            with self.subTest(catalog=path.name):
                catalog = load_catalog(path)
                numeric, categorical = infer_features(catalog)
                selected_query: dict[str, object] | None = None
                for _, row in catalog.iterrows():
                    active = [
                        column for column in numeric
                        if parse_numeric_values(row.get(column))
                    ]
                    package = row.get("Package Type")
                    if not active or pd.isna(package) or not str(package).strip():
                        continue
                    selected_query = {"Package Type": package}
                    selected_query.update({
                        column: row[column]
                        for column in categorical
                        if not pd.isna(row.get(column)) and str(row[column]).strip()
                    })
                    selected_query.update({column: row[column] for column in active[:5]})
                    break

                self.assertIsNotNone(selected_query)
                for mode in (STRICT_MODE, BALANCED_MODE):
                    results = recommend(
                        catalog,
                        selected_query or {},
                        top_k=3,
                        mode=mode,
                    )
                    self.assertFalse(results.empty)


if __name__ == "__main__":
    unittest.main()
