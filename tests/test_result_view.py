from __future__ import annotations

import unittest

import pandas as pd

from app import _gui_manufacturer, _result_column_width, build_result_view


class ResultViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            [
                {
                    "Product": "AUTO-1",
                    "Manufacture": "MCC",
                    "综合得分": 95.0,
                    "已知参数匹配度": 96.0,
                    "参数覆盖率": 100.0,
                    "方向偏好得分": 90.0,
                    "可信度": "高",
                    "关键参数检查": "通过",
                    "车规等级": "车规级",
                    "Package Type": "SOT-23",
                    "Channel": "N",
                    "Drain-Source Voltage VDS (V)": 60.0,
                    "Junction Temperature Tj [max] (°C)": 150.0,
                    "关键参数问题": "无",
                    "缺失参数": "无",
                    "风险提示": "无",
                    "核对提示": "核对 Pinout",
                },
                {
                    "Product": "NORMAL-1",
                    "Manufacture": "MCC",
                    "综合得分": 90.0,
                    "已知参数匹配度": 92.0,
                    "参数覆盖率": 90.0,
                    "方向偏好得分": 85.0,
                    "可信度": "中",
                    "关键参数检查": "待人工确认",
                    "车规等级": "非车规级",
                    "Package Type": "SOT-23",
                    "Channel": "N",
                    "Drain-Source Voltage VDS (V)": 60.0,
                    "Junction Temperature Tj [max] (°C)": 175.0,
                    "关键参数问题": "无",
                    "缺失参数": "EAS",
                    "风险提示": "有缺失",
                    "核对提示": "核对 Pinout",
                },
            ]
        )

    def test_automotive_filter_only_changes_rows(self) -> None:
        view = build_result_view(self.results, automotive_only=True)
        self.assertEqual(view["Product"].tolist(), ["AUTO-1"])
        self.assertEqual(view.columns.tolist(), self.results.columns.tolist())

    def test_compact_view_keeps_summary_constraints_and_critical_values(self) -> None:
        view = build_result_view(
            self.results,
            compact=True,
            candidate_columns=[
                "Package Type", "Channel", "Drain-Source Voltage VDS (V)",
                "Junction Temperature Tj [max] (°C)",
            ],
        )
        self.assertEqual(
            view.columns.tolist(),
            [
                "Product", "Manufacture", "综合得分", "可信度", "关键参数检查",
                "车规等级", "Package Type", "Channel", "Drain-Source Voltage VDS (V)",
                "Junction Temperature Tj [max] (°C)", "关键参数问题", "缺失参数", "风险提示",
            ],
        )
        self.assertNotIn("已知参数匹配度", view.columns)
        self.assertNotIn("参数覆盖率", view.columns)
        self.assertNotIn("方向偏好得分", view.columns)
        self.assertNotIn("核对提示", view.columns)

    def test_filters_can_be_combined_without_mutating_full_results(self) -> None:
        original = self.results.copy(deep=True)
        view = build_result_view(
            self.results,
            compact=True,
            automotive_only=True,
            candidate_columns=[
                "Package Type", "Channel", "Drain-Source Voltage VDS (V)",
                "Junction Temperature Tj [max] (°C)",
            ],
        )
        self.assertEqual(view["Product"].tolist(), ["AUTO-1"])
        pd.testing.assert_frame_equal(self.results, original)

    def test_narrative_columns_do_not_dominate_important_columns(self) -> None:
        self.assertLess(
            _result_column_width("关键参数问题"),
            _result_column_width("Product"),
        )
        self.assertLess(
            _result_column_width("缺失参数"),
            _result_column_width("Product"),
        )
        self.assertGreaterEqual(_result_column_width("综合得分"), 90)

    def test_mcc_is_abbreviated_only_for_gui_display(self) -> None:
        full_name = "MCC (Micro Commercial Components)"
        self.assertEqual(_gui_manufacturer(full_name), "MCC")
        self.assertEqual(_gui_manufacturer("Nexperia"), "Nexperia")

        results = self.results.copy()
        results.loc[0, "Manufacture"] = full_name
        view = build_result_view(results)
        self.assertEqual(view.loc[0, "Manufacture"], full_name)


if __name__ == "__main__":
    unittest.main()
