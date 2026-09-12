"""Tests for the prediction-cleaner's table model and fuse/revert logic.

``CleanerHandler`` is a ``QObject`` subclass, but none of the logic
exercised here touches Qt's signal/event machinery, so instances are
built via ``__new__`` (bypassing ``__init__``/``QObject.__init__``) and
given only the plain-Python attributes each test needs -- mirroring the
``AnalysisResults.__new__`` pattern used elsewhere in this test suite.
Qt widgets (``self.w``) are replaced by lightweight fakes so most of
these tests do not require a running ``QApplication`` -- except the
``BreakpointNavWidget`` tests, since it is a real ``QWidget``; a single
module-level ``QApplication`` instance is created for those.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import zarr
from qtpy.QtCore import QEvent, QObject, QPointF, QRect, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication, QStyleOptionViewItem

from octron.analysis_octron.helpers.analysis_results import AnalysisResults
from octron.cleanup_octron.cleanup_handler import (
    CleanerHandler,
    _rgba_to_qcolor,
)
from octron.cleanup_octron.possible_joins_table import (
    BreakpointNavWidget,
    CheckBoxDelegate,
    PossibleJoinsTableModel,
    coverage_gain_color,
)

# BreakpointNavWidget is a real QWidget (QToolButton/QHBoxLayout), which
# requires a QApplication instance to exist before any is constructed --
# unlike the QObject-only classes (models/delegates) tested elsewhere in
# this file. Created once at import time so it's available regardless of
# test order.
_APP = QApplication.instance() or QApplication([])


def _mouse_release_event(pos):
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


# ---------------------------------------------------------------------------
# PossibleJoinsTableModel
# ---------------------------------------------------------------------------


def test_joins_table_set_candidates_defaults_unchecked():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a - id 1"}])
    assert model.rowCount() == 1
    assert model.checked_track_ids() == []
    assert model.data(model.index(0, 0), Qt.CheckStateRole) == Qt.Unchecked


def test_joins_table_checkbox_toggle_emits_signal_and_updates_state():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 5, "name": "mouse - id 5"}])
    received = []
    model.check_state_changed.connect(lambda: received.append(True))

    idx = model.index(0, 0)
    model.setData(idx, Qt.Checked, role=Qt.CheckStateRole)

    assert model.checked_track_ids() == [5]
    assert received == [True]


def test_joins_table_name_truncation_and_tooltip():
    model = PossibleJoinsTableModel()
    long_name = "a-very-long-layer-name - id 42"
    model.set_candidates([{"track_id": 42, "name": long_name}])
    display = model.data(model.index(0, 1), Qt.DisplayRole)
    tooltip = model.data(model.index(0, 1), Qt.ToolTipRole)
    assert tooltip == long_name
    assert display.endswith("...")
    assert len(display) < len(long_name)


def test_joins_table_set_increase_percentages():
    model = PossibleJoinsTableModel()
    model.set_candidates(
        [{"track_id": 1, "name": "a"}, {"track_id": 2, "name": "b"}]
    )
    model.set_increase_percentages({1: 12.5, 2: 3.0})
    assert model.data(model.index(0, 2)) == "+12.5%"
    assert model.data(model.index(1, 2)) == "+3.0%"


def test_joins_table_clear_resets_rows():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a"}])
    model.clear()
    assert model.rowCount() == 0
    assert model.checked_track_ids() == []


def test_joins_table_foreground_color_tracks_increase_pct():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a"}])
    model.set_increase_percentages({1: 0.0})
    zero_color = model.data(model.index(0, 2), Qt.ForegroundRole)
    model.set_increase_percentages({1: 100.0})
    full_color = model.data(model.index(0, 2), Qt.ForegroundRole)
    assert zero_color.name() == coverage_gain_color(0.0).name()
    assert full_color.name() == coverage_gain_color(100.0).name()
    assert zero_color.name() != full_color.name()


def test_joins_table_header_and_cell_alignment_left_for_layer_column():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a"}])
    header_align = model.headerData(1, Qt.Horizontal, Qt.TextAlignmentRole)
    cell_align = model.data(model.index(0, 1), Qt.TextAlignmentRole)
    left_vcenter = int(Qt.AlignLeft | Qt.AlignVCenter)
    assert int(header_align) == left_vcenter
    assert int(cell_align) == left_vcenter


# ---------------------------------------------------------------------------
# CheckBoxDelegate: click hit-testing must actually toggle the checkbox.
# ---------------------------------------------------------------------------


def test_checkbox_delegate_toggles_on_click_inside_box():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a"}])
    delegate = CheckBoxDelegate()
    index = model.index(0, 0)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 18, 20)

    center = delegate._box_rect(option.rect).center()
    handled = delegate.editorEvent(
        _mouse_release_event(center), model, option, index
    )

    assert handled is True
    assert model.checked_track_ids() == [1]


def test_checkbox_delegate_ignores_click_outside_box():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a"}])
    delegate = CheckBoxDelegate()
    index = model.index(0, 0)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 18, 20)

    outside = QPointF(100, 100)
    handled = delegate.editorEvent(
        _mouse_release_event(outside), model, option, index
    )

    assert handled is False
    assert model.checked_track_ids() == []


def test_checkbox_delegate_click_toggles_back_off():
    model = PossibleJoinsTableModel()
    model.set_candidates([{"track_id": 1, "name": "a", "checked": True}])
    delegate = CheckBoxDelegate()
    index = model.index(0, 0)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 18, 20)

    center = delegate._box_rect(option.rect).center()
    delegate.editorEvent(_mouse_release_event(center), model, option, index)

    assert model.checked_track_ids() == []


# ---------------------------------------------------------------------------
# BreakpointNavWidget: forward/backward target selection, auto-disable,
# dynamic tooltips, and click -> jump_requested.
# ---------------------------------------------------------------------------


def test_breakpoint_nav_widget_disabled_with_no_breakpoints():
    widget = BreakpointNavWidget()
    widget.set_current_frame(10)
    assert widget.backward_btn.isEnabled() is False
    assert widget.forward_btn.isEnabled() is False


def test_breakpoint_nav_widget_disabled_with_no_current_frame():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((10, 20))
    assert widget.backward_btn.isEnabled() is False
    assert widget.forward_btn.isEnabled() is False


def test_breakpoint_nav_widget_before_both_breakpoints():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((49, 55))
    widget.set_current_frame(0)
    assert widget.backward_btn.isEnabled() is False
    assert widget.backward_btn.toolTip() == "No earlier breakpoint"
    assert widget.forward_btn.isEnabled() is True
    assert widget.forward_btn.toolTip() == "Jump to frame 49"


def test_breakpoint_nav_widget_between_both_breakpoints():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((49, 55))
    widget.set_current_frame(52)
    assert widget.backward_btn.isEnabled() is True
    assert widget.backward_btn.toolTip() == "Jump to frame 49"
    assert widget.forward_btn.isEnabled() is True
    assert widget.forward_btn.toolTip() == "Jump to frame 55"


def test_breakpoint_nav_widget_after_both_breakpoints():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((49, 55))
    widget.set_current_frame(100)
    assert widget.backward_btn.isEnabled() is True
    assert widget.backward_btn.toolTip() == "Jump to frame 55"
    assert widget.forward_btn.isEnabled() is False
    assert widget.forward_btn.toolTip() == "No later breakpoint"


def test_breakpoint_nav_widget_click_emits_jump_requested():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((49, 55))
    widget.set_current_frame(0)
    received = []
    widget.jump_requested.connect(received.append)

    widget.forward_btn.click()

    assert received == [49]


def test_breakpoint_nav_widget_click_disabled_button_is_noop():
    widget = BreakpointNavWidget()
    widget.set_breakpoints((49, 55))
    widget.set_current_frame(0)  # backward has nothing to target
    received = []
    widget.jump_requested.connect(received.append)

    widget.backward_btn.click()

    assert received == []


# ---------------------------------------------------------------------------
# coverage_gain_color
# ---------------------------------------------------------------------------


def test_coverage_gain_color_endpoints_and_clamping():
    grey = coverage_gain_color(0.0)
    lemon = coverage_gain_color(100.0)
    assert (grey.red(), grey.green(), grey.blue()) == (136, 136, 136)
    assert (lemon.red(), lemon.green(), lemon.blue()) == (154, 205, 50)
    # Out-of-range values are clamped rather than over/undershooting.
    assert coverage_gain_color(-10.0).name() == grey.name()
    assert coverage_gain_color(150.0).name() == lemon.name()


def test_coverage_gain_color_interpolates_midpoint():
    mid = coverage_gain_color(50.0)
    grey = coverage_gain_color(0.0)
    lemon = coverage_gain_color(100.0)
    # Grey (136, 136, 136) -> lemon green (154, 205, 50): red/green rise,
    # blue falls.
    assert grey.red() < mid.red() < lemon.red()
    assert grey.green() < mid.green() < lemon.green()
    assert grey.blue() > mid.blue() > lemon.blue()


# ---------------------------------------------------------------------------
# _rgba_to_qcolor
# ---------------------------------------------------------------------------


def test_rgba_to_qcolor_converts_and_clamps():
    color = _rgba_to_qcolor((1.5, -0.2, 0.5, 1.0))
    # QColor stores components at 16-bit precision internally, so allow a
    # small absolute tolerance rather than requiring exact float equality.
    assert color.redF() == pytest.approx(1.0, abs=1e-4)
    assert color.greenF() == pytest.approx(0.0, abs=1e-4)
    assert color.blueF() == pytest.approx(0.5, abs=1e-4)
    assert color.alphaF() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# CleanerHandler pure-logic helpers (no Qt widgets/viewer required)
# ---------------------------------------------------------------------------


def _make_handler():
    """Build a bare CleanerHandler for pure-logic tests (no QObject init)."""
    handler = CleanerHandler.__new__(CleanerHandler)
    handler.results = None
    handler.w = None
    handler._track_positions = {}
    handler._candidates = []
    handler._source_track_id = None
    handler._num_frames = None
    return handler


def _make_results_stub(track_id_label, csv_frame_indices):
    """Build a minimal AnalysisResults-like stub for filtering tests."""
    obj = AnalysisResults.__new__(AnalysisResults)
    obj.track_id_label = track_id_label
    obj._csv_frame_indices = csv_frame_indices
    return obj


def test_csv_frames_returns_empty_set_when_no_results():
    handler = _make_handler()
    assert handler._csv_frames(1) == set()


def test_csv_frames_reads_from_results():
    handler = _make_handler()
    handler.results = _make_results_stub({1: "mouse"}, {1: {0, 1, 2}})
    assert handler._csv_frames(1) == {0, 1, 2}
    assert handler._csv_frames(99) == set()


def test_gap_and_distance_none_when_frame_ranges_overlap():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(0, 20)), 2: set(range(10, 30))},
    )
    handler._track_positions = {1: {}, 2: {}}
    assert handler._gap_and_distance(1, 2) is None


def test_gap_and_distance_computes_forward_gap_and_euclidean_distance():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(0, 10)), 2: set(range(15, 25))},
    )
    handler._track_positions = {
        1: {9: (0.0, 0.0)},
        2: {15: (3.0, 4.0)},
    }
    gap_frames, distance = handler._gap_and_distance(1, 2)
    assert gap_frames == 5  # frames 10..14 are the gap
    assert distance == pytest.approx(5.0)  # 3-4-5 triangle


def test_gap_and_distance_computes_backward_gap():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(20, 30)), 2: set(range(0, 10))},
    )
    handler._track_positions = {
        1: {20: (10.0, 10.0)},
        2: {9: (10.0, 10.0)},
    }
    gap_frames, distance = handler._gap_and_distance(1, 2)
    assert gap_frames == 10  # frames 10..19 (10 frames)
    assert distance == pytest.approx(0.0)


def test_gap_and_distance_none_when_position_missing():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(0, 10)), 2: set(range(15, 25))},
    )
    handler._track_positions = {1: {}, 2: {}}  # no position data
    assert handler._gap_and_distance(1, 2) is None


def test_breakpoints_for_matches_gap_and_distance_endpoints():
    """_breakpoints_for() must return the same two frames _gap_and_distance()
    used internally, just sorted ascending instead of source/candidate-order.
    """
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(0, 10)), 2: set(range(15, 25))},
    )
    handler._track_positions = {
        1: {9: (0.0, 0.0)},
        2: {15: (3.0, 4.0)},
    }
    assert handler._breakpoints_for(1, 2) == (9, 15)
    # Reversed roles (candidate comes first): still sorted ascending.
    assert handler._breakpoints_for(2, 1) == (9, 15)


def test_breakpoints_for_none_when_overlapping():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: set(range(0, 20)), 2: set(range(10, 30))},
    )
    assert handler._breakpoints_for(1, 2) is None


def test_next_fused_id_is_max_plus_one():
    handler = _make_handler()
    handler.results = _make_results_stub({1: "a", 5: "b", 3: "c"}, {})
    assert handler._next_fused_id() == 6


def test_next_fused_id_defaults_to_one_when_empty():
    handler = _make_handler()
    handler.results = _make_results_stub({}, {})
    assert handler._next_fused_id() == 1


# ---------------------------------------------------------------------------
# Layer visibility: isolate/show-all must never touch non-owned layers
# (e.g. the source video), only OCTRON's own track/mask layers.
# ---------------------------------------------------------------------------


class _FakeNapariLayer:
    def __init__(self, name, head_length=0, tail_length=30):
        self.name = name
        self.visible = True
        self.color_by = None
        self.head_length = head_length
        self.tail_length = tail_length


class _FakeNapariLayers(list):
    """Minimal stand-in for napari's LayerList (name-keyed access)."""

    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if layer.name == key:
                    return layer
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, str):
            return any(layer.name == key for layer in self)
        return super().__contains__(key)


def test_owned_layer_names_excludes_non_track_layers():
    handler = _make_handler()
    handler.results = _make_results_stub({1: "mouse", 2: "mouse"}, {})
    owned = handler._owned_layer_names()
    assert owned == {
        "mouse - id 1",
        "mouse - MASKS - id 1",
        "mouse - id 2",
        "mouse - MASKS - id 2",
    }


def test_isolate_tracks_leaves_video_layer_untouched():
    handler = _make_handler()
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse"},
        {1: {0}, 2: {1}},
    )
    video_layer = _FakeNapariLayer("my_video.mp4")
    track1 = _FakeNapariLayer("mouse - id 1")
    track2 = _FakeNapariLayer("mouse - id 2")
    viewer = type("FakeViewer", (), {})()  # simple attribute container
    viewer.layers = _FakeNapariLayers([video_layer, track1, track2])

    class _FakeWidget:
        _viewer = viewer

    handler.w = _FakeWidget()

    handler._isolate_tracks([1])
    assert video_layer.visible is True
    assert track1.visible is True
    assert track2.visible is False

    handler._isolate_tracks([2])
    assert video_layer.visible is True
    assert track1.visible is False
    assert track2.visible is True


# ---------------------------------------------------------------------------
# Full-trace head/tail extension: checked rows (and the source) show the
# entire track; unchecking restores each layer's original head/tail.
# ---------------------------------------------------------------------------


def _make_full_trace_handler():
    handler = _make_handler()
    handler.results = _make_results_stub({1: "mouse", 2: "mouse"}, {})
    handler._num_frames = 200
    handler._full_traced_ids = set()
    handler._original_trace_lengths = {}

    track1 = _FakeNapariLayer("mouse - id 1", head_length=0, tail_length=250)
    track2 = _FakeNapariLayer("mouse - id 2", head_length=0, tail_length=250)
    viewer = type("FakeViewer", (), {})()
    viewer.layers = _FakeNapariLayers([track1, track2])

    class _FakeWidget:
        _viewer = viewer

    handler.w = _FakeWidget()
    return handler, track1, track2


def test_apply_full_trace_state_extends_head_and_tail_to_num_frames():
    handler, track1, _track2 = _make_full_trace_handler()
    handler._apply_full_trace_state([1])
    assert (track1.head_length, track1.tail_length) == (200, 200)


def test_apply_full_trace_state_captures_and_restores_original_lengths():
    handler, _track1, track2 = _make_full_trace_handler()
    handler._apply_full_trace_state([1, 2])
    assert (track2.head_length, track2.tail_length) == (200, 200)
    assert handler._original_trace_lengths[2] == (0, 250)

    # Unchecking candidate 2 (source 1 stays selected) restores track2
    # only, leaving track1 untouched.
    handler._apply_full_trace_state([1])
    assert (track2.head_length, track2.tail_length) == (0, 250)


def test_apply_full_trace_state_restores_previous_source_on_switch():
    handler, track1, track2 = _make_full_trace_handler()
    handler._apply_full_trace_state([1])
    assert (track1.head_length, track1.tail_length) == (200, 200)

    # Switching source to track 2 restores track1 and full-traces track2.
    handler._apply_full_trace_state([2])
    assert (track1.head_length, track1.tail_length) == (0, 250)
    assert (track2.head_length, track2.tail_length) == (200, 200)


def test_apply_full_trace_state_recheck_reuses_true_original():
    """Re-checking after an uncheck must not treat the full-trace value
    as the new "original" -- the true pre-override value stays cached.
    """
    handler, _track1, track2 = _make_full_trace_handler()
    handler._apply_full_trace_state([1, 2])
    handler._apply_full_trace_state([1])  # uncheck 2 -> restored
    handler._apply_full_trace_state([1, 2])  # re-check 2
    assert (track2.head_length, track2.tail_length) == (200, 200)
    assert handler._original_trace_lengths[2] == (0, 250)


def test_show_all_layers_makes_every_layer_visible():
    handler, track1, track2 = _make_full_trace_handler()
    track1.visible = False
    track2.visible = False

    handler._show_all_layers()

    assert track1.visible is True
    assert track2.visible is True


def test_show_all_layers_restores_full_traced_head_and_tail():
    """Show all must undo the full-trace override, not just visibility."""
    handler, track1, track2 = _make_full_trace_handler()
    handler._apply_full_trace_state([1, 2])
    assert (track1.head_length, track1.tail_length) == (200, 200)
    assert (track2.head_length, track2.tail_length) == (200, 200)

    handler._show_all_layers()

    assert (track1.head_length, track1.tail_length) == (0, 250)
    assert (track2.head_length, track2.tail_length) == (0, 250)
    assert handler._full_traced_ids == set()


# ---------------------------------------------------------------------------
# Breakpoint-nav button lifecycle: setup after candidate refresh, jump
# wiring, and viewer-frame-change propagation to every row's widget.
# ---------------------------------------------------------------------------


class _FakeQtIndex:
    def __init__(self, row, col):
        self.row = row
        self.col = col


class _FakeJoinsTableModel:
    def index(self, row, col):
        return _FakeQtIndex(row, col)


class _FakeQTableView:
    """Minimal stand-in for QTableView.set/indexWidget (row/col-keyed)."""

    def __init__(self):
        self._widgets = {}

    def setIndexWidget(self, index, widget):
        self._widgets[(index.row, index.col)] = widget

    def indexWidget(self, index):
        return self._widgets.get((index.row, index.col))


class _FakeDimsEvents:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self):
        for callback in list(self._callbacks):
            callback(None)


class _FakeDims:
    def __init__(self):
        self.current_step = (0,)
        self.events = type("Events", (), {"current_step": _FakeDimsEvents()})()

    def set_point(self, axis, value):
        step = list(self.current_step)
        step[axis] = value
        self.current_step = tuple(step)
        self.events.current_step.emit()


class _FakeViewerWithDims:
    def __init__(self):
        self.dims = _FakeDims()


def _make_breakpoint_handler():
    # Unlike _make_handler()'s other pure-attribute uses, these tests
    # rely on real Qt signal/slot delivery (BreakpointNavWidget.
    # jump_requested -> handler._jump_to_frame), which silently never
    # fires if the QObject's C++ side was never constructed -- so the
    # underlying QObject.__init__() is called explicitly here (still
    # skipping CleanerHandler.__init__() itself, which needs a real
    # widget/model).
    handler = CleanerHandler.__new__(CleanerHandler)
    QObject.__init__(handler)
    handler.results = _make_results_stub({1: "mouse", 2: "mouse"}, {})
    handler.joins_table = _FakeJoinsTableModel()
    handler._breakpoint_widgets = []
    handler._candidates = [
        {"track_id": 2, "name": "mouse - id 2", "breakpoints": (49, 55)}
    ]

    class _FakeWidget:
        _viewer = _FakeViewerWithDims()
        possible_joins_table = _FakeQTableView()

    handler.w = _FakeWidget()
    return handler


def test_setup_breakpoint_buttons_creates_one_widget_per_candidate():
    handler = _make_breakpoint_handler()
    handler._setup_breakpoint_buttons()

    assert len(handler._breakpoint_widgets) == 1
    widget = handler.w.possible_joins_table.indexWidget(
        handler.joins_table.index(0, 3)
    )
    assert widget is handler._breakpoint_widgets[0]
    assert widget.forward_btn.toolTip() == "Jump to frame 49"


def test_clear_breakpoint_widgets_empties_the_list():
    handler = _make_breakpoint_handler()
    handler._setup_breakpoint_buttons()
    handler._clear_breakpoint_widgets()
    assert handler._breakpoint_widgets == []


def test_jump_to_frame_calls_viewer_dims_set_point():
    handler = _make_breakpoint_handler()
    handler._jump_to_frame(49)
    assert handler.w._viewer.dims.current_step == (49,)


def test_breakpoint_button_click_jumps_viewer_and_updates_all_rows():
    """Clicking a row's forward button jumps the viewer AND -- via the
    connected dims.events.current_step callback -- refreshes every row's
    button state for the new frame, including other rows.
    """
    handler = _make_breakpoint_handler()
    # row1's breakpoints (30, 40) straddle the frame row0's forward
    # click will jump to (49), so the transition is observable: before
    # the jump (viewer at frame 0) both are ahead of "current"; after
    # the jump (viewer at frame 49) both are behind it.
    handler._candidates.append(
        {"track_id": 3, "name": "mouse - id 3", "breakpoints": (30, 40)}
    )
    handler._setup_breakpoint_buttons()
    viewer = handler.w._viewer
    viewer.dims.events.current_step.connect(handler._on_viewer_frame_changed)

    row0_widget = handler._breakpoint_widgets[0]  # breakpoints (49, 55)
    row1_widget = handler._breakpoint_widgets[1]  # breakpoints (30, 40)
    assert row1_widget.backward_btn.isEnabled() is False  # nothing < 0
    assert row1_widget.forward_btn.toolTip() == "Jump to frame 30"

    row0_widget.forward_btn.click()  # jumps viewer to frame 49

    assert viewer.dims.current_step == (49,)
    # row1 didn't move itself, but its cached current_frame must have
    # followed the viewer via the dims.events.current_step callback:
    # both of its breakpoints (30, 40) are now behind frame 49.
    assert row1_widget.backward_btn.toolTip() == "Jump to frame 40"
    assert row1_widget.forward_btn.isEnabled() is False


def test_recompute_increase_values_handles_overlapping_candidates():
    """Marginal +coverage% accounts for overlap between checked candidates."""

    class _FakeLabel:
        def setText(self, text):
            self.value = text

        def setStyleSheet(self, _style):
            pass

    class _FakeJoinsTable:
        def __init__(self, checked):
            self._checked = checked
            self.percentages = None

        def checked_track_ids(self):
            return list(self._checked)

        def set_increase_percentages(self, mapping):
            self.percentages = mapping

    class _FakeWidget:
        def __init__(self):
            self.increase_percent_label = _FakeLabel()

    handler = _make_handler()
    # source: frames 0-9 (10 frames); candidate A: 20-29 (10 new frames);
    # candidate B: 25-39 (15 frames, 5 overlap with A -> marginal 10 if A
    # already checked).
    handler.results = _make_results_stub(
        {1: "mouse", 2: "mouse", 3: "mouse"},
        {
            1: set(range(0, 10)),
            2: set(range(20, 30)),
            3: set(range(25, 40)),
        },
    )
    handler._source_track_id = 1
    handler._num_frames = 100
    handler._candidates = [
        {"track_id": 2, "name": "mouse - id 2"},
        {"track_id": 3, "name": "mouse - id 3"},
    ]
    handler.w = _FakeWidget()
    handler.joins_table = _FakeJoinsTable(checked=[2, 3])

    handler._recompute_increase_values()

    # Candidate 2's marginal contribution (given 3 also checked): frames
    # 25-29 are already covered by candidate 3, leaving only 20-24 (5
    # frames) as this row's own contribution.
    assert handler.joins_table.percentages[2] == pytest.approx(5.0)
    # Candidate 3's marginal contribution (given 2 also checked): only
    # frames 30-39 are new (25-29 already covered by candidate 2).
    assert handler.joins_table.percentages[3] == pytest.approx(10.0)
    # Total union increase: source ∪ 2 ∪ 3 = {0-9} ∪ {20-39} (30 frames);
    # minus source's own 10 frames -> 20 new frames out of 100 -> 20%.
    assert handler.w.increase_percent_label.value == "20.0%"


# ---------------------------------------------------------------------------
# Disk-level fuse / archive / revert round trip
# ---------------------------------------------------------------------------

NUM_FRAMES = 60
HEIGHT, WIDTH = 8, 8


def _write_tracking_csv(path, track_id, label, frames, num_frames=NUM_FRAMES):
    """Write a tracking CSV in the on-disk format AnalysisResults expects.

    Mirrors AnalysisOctron.predict_batch()'s writer: 6 metadata lines + one
    blank separator line, then a MultiIndex (frame_counter, frame_idx,
    track_id) DataFrame written with the index included.
    """
    df = pd.DataFrame(
        {
            "frame_counter": range(len(frames)),
            "frame_idx": frames,
            "track_id": track_id,
            "label": label,
            "confidence": 0.9,
            "pos_x": 10.0 + np.arange(len(frames)),
            "pos_y": 10.0 + np.arange(len(frames)),
            "bbox_x_min": 0.0,
            "bbox_x_max": 5.0,
            "bbox_y_min": 0.0,
            "bbox_y_max": 5.0,
            "bbox_area": 25.0,
            "bbox_aspect_ratio": 1.0,
        }
    ).set_index(["frame_counter", "frame_idx", "track_id"])
    header = [
        "video_name: test.mp4",
        f"frame_count: {num_frames}",
        f"frame_count_analyzed: {num_frames}",
        f"video_height: {HEIGHT}",
        f"video_width: {WIDTH}",
        f"created_at: {datetime.now()}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, na_rep="NaN", lineterminator="\n")


@pytest.fixture
def two_track_results_dir(tmp_path):
    """Two same-label, temporally-exclusive tracks with mask arrays on disk."""
    label = "mouse"
    frames_a = list(range(0, 20))
    frames_b = list(range(25, 45))
    _write_tracking_csv(tmp_path / f"{label}_track_1.csv", 1, label, frames_a)
    _write_tracking_csv(tmp_path / f"{label}_track_2.csv", 2, label, frames_b)

    zarr_path = tmp_path / "predictions.zarr"
    store = zarr.storage.LocalStore(zarr_path, read_only=False)
    zarr.open_group(store=store, mode="a")
    for track_id, frames in [(1, frames_a), (2, frames_b)]:
        arr = zarr.create_array(
            store=store,
            name=f"{track_id}_masks",
            shape=(NUM_FRAMES, HEIGHT, WIDTH),
            chunks=(1, HEIGHT, WIDTH),
            fill_value=-1,
            dtype="int8",
            overwrite=True,
        )
        arr.attrs["label"] = label
        arr.attrs["classes"] = {"0": label}
        for f in frames:
            arr[f, :, :] = 1
        arr.attrs["annotated_frames"] = list(frames)

    meta = {
        "model_classes": {"0": label},
        "video_info": {
            "num_frames_original": NUM_FRAMES,
            "height": HEIGHT,
            "width": WIDTH,
        },
    }
    with open(tmp_path / "prediction_metadata.json", "w") as f:
        json.dump(meta, f)

    return tmp_path, frames_a, frames_b


class _FakeWidgetNoViewer:
    """Stand-in for octron_prediction_cleaner_widget with no napari viewer.

    ``_reload()`` bails out early when ``_viewer`` is None, so save/revert
    can be exercised purely at the filesystem level without napari/Qt.
    """

    _viewer = None


def _make_disk_handler(results_dir):
    handler = CleanerHandler.__new__(CleanerHandler)
    handler.w = _FakeWidgetNoViewer()
    handler.results = AnalysisResults(results_dir, verbose=False)
    handler.save_dir = results_dir
    handler._track_positions = {}
    handler._candidates = []
    handler._source_track_id = None
    handler._num_frames = handler.results.num_frames
    return handler


def test_csv_path_for_track_locates_by_track_id(two_track_results_dir):
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    path = handler._csv_path_for_track(2)
    assert path is not None
    assert path.name == "mouse_track_2.csv"
    assert handler._csv_path_for_track(999) is None


def test_save_fuses_csv_and_masks_then_revert_restores_originals(
    two_track_results_dir,
):
    results_dir, frames_a, frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)

    # save() needs a joins_table with checked_track_ids(); use the real
    # model since it needs no QApplication.
    handler.joins_table = PossibleJoinsTableModel()
    handler.joins_table.set_candidates([{"track_id": 2, "name": "x"}])
    handler.joins_table.setData(
        handler.joins_table.index(0, 0), Qt.Checked, role=Qt.CheckStateRole
    )
    handler._source_track_id = 1

    handler.save()

    # Fused CSV/zarr array exist; originals archived, not deleted.
    fused_csv = results_dir / "mouse_track_3.csv"
    assert fused_csv.exists()
    assert not (results_dir / "mouse_track_1.csv").exists()
    assert not (results_dir / "mouse_track_2.csv").exists()

    archives = list((results_dir / "fused_originals").iterdir())
    assert len(archives) == 1
    manifest = json.loads((archives[0] / "manifest.json").read_text())
    assert manifest["fused_id"] == 3
    assert sorted(manifest["member_track_ids"]) == [1, 2]
    assert (archives[0] / "mouse_track_1.csv.bak").exists()
    assert (archives[0] / "mouse_track_2.csv.bak").exists()

    fused_results = AnalysisResults(results_dir, verbose=False)
    assert fused_results.track_id_label == {3: "mouse"}
    mask_data = fused_results.get_mask_data()
    assert set(mask_data[3]["frame_indices"]) == set(frames_a) | set(frames_b)

    # Fusion did not touch handler.results itself (no _reload wiring in
    # this fake-widget setup) -- reload it manually the way _reload()
    # would, then revert.
    handler.results = fused_results
    handler.revert_last()

    assert not fused_csv.exists()
    assert (results_dir / "mouse_track_1.csv").exists()
    assert (results_dir / "mouse_track_2.csv").exists()
    assert not list((results_dir / "fused_originals").glob("*/manifest.json"))

    reverted_results = AnalysisResults(results_dir, verbose=False)
    assert reverted_results.track_id_label == {1: "mouse", 2: "mouse"}
    mask_data_reverted = reverted_results.get_mask_data()
    assert set(mask_data_reverted[1]["frame_indices"]) == set(frames_a)
    assert set(mask_data_reverted[2]["frame_indices"]) == set(frames_b)


def test_save_noop_without_checked_candidates(two_track_results_dir):
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler.joins_table = PossibleJoinsTableModel()
    handler._source_track_id = 1

    handler.save()  # no checked rows -> should be a no-op

    assert (results_dir / "mouse_track_1.csv").exists()
    assert (results_dir / "mouse_track_2.csv").exists()
    assert not (results_dir / "fused_originals").exists()


def test_reset_all_with_no_archives_is_noop(two_track_results_dir):
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler.reset_all()  # should not raise
    assert (results_dir / "mouse_track_1.csv").exists()


def test_set_default_thresholds_uses_fov_and_frame_count(
    two_track_results_dir,
):
    """max_gap_space = FOV-size/3; max_gap_time = 10% of frame count."""

    class _FakeSpinbox:
        def __init__(self):
            self.value_set = None

        def setValue(self, value):
            self.value_set = value

    class _FakeWidget:
        def __init__(self):
            self.max_gap_time_spinbox = _FakeSpinbox()
            self.max_gap_space_spinbox = _FakeSpinbox()

    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler.w = _FakeWidget()

    handler._set_default_thresholds()

    assert handler.w.max_gap_time_spinbox.value_set == round(NUM_FRAMES * 0.1)
    assert handler.w.max_gap_space_spinbox.value_set == round(
        max(WIDTH, HEIGHT) / 3
    )


# ---------------------------------------------------------------------------
# Layer-removal-triggered delete: removing a track's layer in the viewer
# archives (not deletes) its CSV/mask, undoable via the same revert log.
# ---------------------------------------------------------------------------


def test_track_id_for_layer_name_matches_tracks_and_mask_names():
    handler = _make_handler()
    handler.results = _make_results_stub({1: "mouse", 2: "mouse"}, {})
    assert handler._track_id_for_layer_name("mouse - id 1") == 1
    assert handler._track_id_for_layer_name("mouse - MASKS - id 2") == 2
    assert handler._track_id_for_layer_name("some_video.mp4") is None
    # Well-formed suffix, but not a track that currently exists.
    assert handler._track_id_for_layer_name("mouse - id 999") is None


class _FakeRemovedEvent:
    def __init__(self, layer):
        self.value = layer


class _FakeLayerForRemoval:
    def __init__(self, name):
        self.name = name


def test_on_layer_removed_ignores_non_owned_layer(two_track_results_dir):
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler._reloading = False

    handler._on_layer_removed(
        _FakeRemovedEvent(_FakeLayerForRemoval("some_video.mp4"))
    )

    assert (results_dir / "mouse_track_1.csv").exists()
    assert (results_dir / "mouse_track_2.csv").exists()
    assert not (results_dir / "fused_originals").exists()


def test_on_layer_removed_ignores_events_while_reloading(
    two_track_results_dir,
):
    """_reload()'s own viewer.layers.clear() must not trigger archival."""
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler._reloading = True

    handler._on_layer_removed(
        _FakeRemovedEvent(_FakeLayerForRemoval("mouse - id 2"))
    )

    assert (results_dir / "mouse_track_2.csv").exists()
    assert not (results_dir / "fused_originals").exists()


def test_on_layer_removed_archives_track_for_tracks_layer_name(
    two_track_results_dir,
):
    results_dir, _frames_a, frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler._reloading = False
    handler._schedule_reload = lambda: None  # no real viewer/event loop here

    handler._on_layer_removed(
        _FakeRemovedEvent(_FakeLayerForRemoval("mouse - id 2"))
    )

    assert not (results_dir / "mouse_track_2.csv").exists()
    assert (results_dir / "mouse_track_1.csv").exists()
    archives = list((results_dir / "fused_originals").iterdir())
    assert len(archives) == 1
    manifest = json.loads((archives[0] / "manifest.json").read_text())
    assert manifest["kind"] == "delete"
    assert manifest["deleted_track_id"] == 2

    fresh = AnalysisResults(results_dir, verbose=False)
    assert fresh.track_id_label == {1: "mouse"}
    mask_data = fresh.get_mask_data()
    assert 2 not in mask_data


def test_on_layer_removed_archives_track_for_mask_layer_name(
    two_track_results_dir,
):
    """Deleting the Labels/MASKS layer (not Tracks) also deletes the track."""
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler._reloading = False
    handler._schedule_reload = lambda: None

    handler._on_layer_removed(
        _FakeRemovedEvent(_FakeLayerForRemoval("mouse - MASKS - id 2"))
    )

    assert not (results_dir / "mouse_track_2.csv").exists()


def test_delete_track_is_idempotent_once_archived(two_track_results_dir):
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)

    handler._delete_track(2)
    archives_after_first = list((results_dir / "fused_originals").iterdir())
    handler._delete_track(2)  # CSV already moved away -> no-op
    archives_after_second = list((results_dir / "fused_originals").iterdir())

    assert len(archives_after_first) == 1
    assert len(archives_after_second) == 1


def test_revert_last_restores_deleted_track(two_track_results_dir):
    results_dir, _frames_a, frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)

    handler._delete_track(2)
    handler.results = AnalysisResults(results_dir, verbose=False)

    handler.revert_last()

    assert (results_dir / "mouse_track_2.csv").exists()
    assert not list((results_dir / "fused_originals").glob("*/manifest.json"))
    reverted = AnalysisResults(results_dir, verbose=False)
    assert reverted.track_id_label == {1: "mouse", 2: "mouse"}
    mask_data = reverted.get_mask_data()
    assert set(mask_data[2]["frame_indices"]) == set(frames_b)


def test_schedule_reload_defers_call_to_next_event_loop_tick():
    import time

    handler = _make_handler()
    called = []
    handler._reload = lambda: called.append(True)

    handler._schedule_reload()
    assert called == []  # not yet -- deferred, not synchronous

    for _ in range(10):
        _APP.processEvents()
        if called:
            break
        time.sleep(0.01)
    assert called == [True]


def test_revert_last_treats_missing_kind_key_as_fuse(two_track_results_dir):
    """Archives written before the "kind" field existed must still revert."""
    results_dir, _frames_a, _frames_b = two_track_results_dir
    handler = _make_disk_handler(results_dir)
    handler.joins_table = PossibleJoinsTableModel()
    handler.joins_table.set_candidates([{"track_id": 2, "name": "x"}])
    handler.joins_table.setData(
        handler.joins_table.index(0, 0), Qt.Checked, role=Qt.CheckStateRole
    )
    handler._source_track_id = 1
    handler.save()

    fused_results = AnalysisResults(results_dir, verbose=False)
    archive_dir = next((results_dir / "fused_originals").iterdir())
    manifest_path = archive_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["kind"]
    manifest_path.write_text(json.dumps(manifest))

    handler.results = fused_results
    handler.revert_last()

    assert (results_dir / "mouse_track_1.csv").exists()
    assert (results_dir / "mouse_track_2.csv").exists()
