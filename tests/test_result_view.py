from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from app import (
    _gui_manufacturer,
    _match_mode_label,
    _match_mode_value,
    _result_column_width,
    MatchingApp,
    build_result_view,
)
from matching_model import BALANCED_MODE, STRICT_MODE


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
                    "关键规则符合度": 100.0,
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
                    "关键规则符合度": 85.0,
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
                "Product", "Manufacture", "综合得分", "关键规则符合度", "可信度",
                "关键参数检查", "车规等级", "Package Type", "Channel",
                "Drain-Source Voltage VDS (V)",
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

    def test_match_mode_display_mapping_is_safe(self) -> None:
        self.assertEqual(_match_mode_value("严格模式"), STRICT_MODE)
        self.assertEqual(_match_mode_value("平衡/整体接近模式"), BALANCED_MODE)
        self.assertEqual(_match_mode_value("未知模式"), STRICT_MODE)
        self.assertEqual(_match_mode_label(BALANCED_MODE), "平衡/整体接近模式")


class MatchModeGuiTests(unittest.TestCase):
    def test_run_match_passes_selected_mode_to_model(self) -> None:
        app = object.__new__(MatchingApp)
        app.match_mode_var = SimpleNamespace(get=lambda: "平衡/整体接近模式")
        app.result_mode_var = SimpleNamespace(set=Mock())
        app.top_var = SimpleNamespace(get=lambda: 7)
        app.df = pd.DataFrame({"Product": ["X"]})
        query = {"Package Type": "PKG", "Value": "1"}
        app._collect_query = Mock(return_value=query)
        app._refresh_results_view = Mock()

        expected = pd.DataFrame({"Product": ["MATCH"]})
        with patch("app.recommend", return_value=expected) as mocked_recommend:
            MatchingApp.run_match(app)

        mocked_recommend.assert_called_once_with(
            app.df,
            query,
            top_k=7,
            mode=BALANCED_MODE,
        )
        self.assertIs(app.results, expected)
        self.assertEqual(app.results_mode, BALANCED_MODE)
        app.result_mode_var.set.assert_called_once_with(
            "本表模式：平衡/整体接近模式"
        )
        self.assertTrue(app.has_run_match)
        app._refresh_results_view.assert_called_once_with()


class ResultCopyTests(unittest.TestCase):
    def test_multi_selection_is_synchronized_between_frozen_and_detail_tables(self) -> None:
        app = object.__new__(MatchingApp)
        app._syncing_result_selection = False
        app.product_tree = Mock()
        app.tree = Mock()
        app.product_tree.selection.return_value = ("result-0", "result-2")
        app.tree.selection.return_value = ("old-result",)

        MatchingApp._sync_result_selection(
            app,
            SimpleNamespace(widget=app.product_tree),
        )

        app.tree.selection_remove.assert_called_once_with("old-result")
        app.tree.selection_set.assert_called_once_with("result-0", "result-2")
        app.tree.focus.assert_called_once_with("result-0")
        app.tree.see.assert_called_once_with("result-0")
        self.assertFalse(app._syncing_result_selection)

    def test_selected_product_numbers_are_copied_one_per_line(self) -> None:
        app = object.__new__(MatchingApp)
        app.product_tree = Mock()
        app.tree = Mock()
        app.status_var = SimpleNamespace(set=Mock())
        app.clipboard_clear = Mock()
        app.clipboard_append = Mock()
        # Treeview 不承诺 selection() 的顺序；复制结果应保持当前排名顺序。
        app.product_tree.selection.return_value = ("result-2", "result-0")
        app.product_tree.get_children.return_value = (
            "result-0", "result-1", "result-2",
        )
        values_by_item = {
            "result-0": ("PART-A",),
            "result-2": ("PART-C",),
        }
        app.product_tree.item.side_effect = (
            lambda item_id, _option: values_by_item.get(item_id, ())
        )

        result = MatchingApp.copy_selected_product_numbers(app)

        self.assertEqual(result, "break")
        app.clipboard_clear.assert_called_once_with()
        app.clipboard_append.assert_called_once_with("PART-A\nPART-C")
        app.status_var.set.assert_called_once_with("已复制 2 个物料号（每行一个）")

    def test_copy_without_selection_does_not_clear_clipboard(self) -> None:
        app = object.__new__(MatchingApp)
        app.product_tree = Mock()
        app.tree = Mock()
        app.status_var = SimpleNamespace(set=Mock())
        app.clipboard_clear = Mock()
        app.clipboard_append = Mock()
        app.product_tree.selection.return_value = ()
        app.product_tree.get_children.return_value = ()
        app.tree.selection.return_value = ()

        result = MatchingApp.copy_selected_product_numbers(app)

        self.assertEqual(result, "break")
        app.clipboard_clear.assert_not_called()
        app.clipboard_append.assert_not_called()
        app.status_var.set.assert_called_once_with(
            "请先在匹配结果中选择要复制的物料"
        )


if __name__ == "__main__":
    unittest.main()
