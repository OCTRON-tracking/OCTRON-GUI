"""Qt handler wiring the prediction-cleaner GUI to disk fuse/revert logic.

Mirrors ``octron/analysis_octron/gui/analysis_handler.py::AnalysisHandler``'s
relationship to the main widget: ``self.w`` is the parent widget owning
the Qt controls this handler wires up, and business logic lives here,
separate from the generated ``setupUi()`` code in ``cleanup_gui_elements.py``.

See the "Track Fusion widget" plan for the full functional spec this
implements (candidate filtering, coverage math, on-disk fuse/archive,
revert/reset).
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import zarr
from loguru import logger
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import QObject
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QHeaderView

from octron.analysis_octron.helpers.analysis_results import AnalysisResults
from octron.analysis_octron.helpers.analysis_zarr import create_prediction_zarr
from octron.cleanup_octron.possible_joins_table import (
    BreakpointNavWidget,
    CheckBoxDelegate,
    PossibleJoinsTableModel,
    coverage_gain_color,
)
from octron.sam_octron.helpers.sam_zarr import (
    get_annotated_frames,
    mark_frames_annotated,
)
from octron.tracking.helpers.tracker_vis import create_color_icon


def _rgba_to_qcolor(rgba):
    """Convert an RGBA float tuple ([0, 1] range) into a QColor."""
    r, g, b, a = (float(c) for c in rgba)
    clamp = max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b))
    return QColor.fromRgbF(*clamp, max(0.0, min(1.0, a)))


class CleanerHandler(QObject):
    """Wire the prediction-cleaner GUI to on-disk track-fusion logic."""

    def __init__(self, parent_widget, analysis_results=None, save_dir=None):
        """Initialize the handler, storing widget/results refs.

        Parameters
        ----------
        parent_widget : QWidget
            The ``octron_prediction_cleaner_widget`` instance that owns
            the GUI controls this handler wires up.
        analysis_results : AnalysisResults, optional
            Already-loaded results for the prediction folder currently
            shown in the viewer. None when the widget was opened
            manually from the Plugins menu without a prediction folder
            in view yet.
        save_dir : str or Path, optional
            Path to the prediction folder ``analysis_results`` was
            loaded from. Required (together with ``analysis_results``)
            for any join/fuse/revert/reset operation.

        """
        super().__init__()
        # cleanup_widget.py -> octron_prediction_cleaner_widget
        self.w = parent_widget
        self.results = analysis_results
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.joins_table = PossibleJoinsTableModel()
        self.w.possible_joins_table.setModel(self.joins_table)
        self._configure_joins_table_view()

        self._label_colors = {}
        self._track_positions = {}
        self._candidates = []
        self._source_track_id = None
        self._num_frames = (
            analysis_results.num_frames if analysis_results else None
        )
        # Track IDs currently displayed with an extended ("full") trace,
        # and the head/tail lengths they had before that override, so
        # unchecking a candidate (or switching source) can put them
        # back exactly as they were. See _apply_full_trace_state().
        self._full_traced_ids = set()
        self._original_trace_lengths = {}

        # One BreakpointNavWidget per candidate row (column 3), recreated
        # on every _refresh_candidates(). Kept here (rather than only
        # relying on QTableView.setIndexWidget()'s own bookkeeping) so
        # they can be explicitly torn down and so _on_viewer_frame_changed
        # can update all of them without needing the view at all.
        self._breakpoint_widgets = []
        viewer = getattr(self.w, "_viewer", None)
        if viewer is not None:
            viewer.dims.events.current_step.connect(
                self._on_viewer_frame_changed
            )

    def _configure_joins_table_view(self):
        """Fix up column sizing/checkbox styling that setupUi() can't express.

        The generated ``setupUi()`` (Designer/uic output; not to be
        hand-edited) applies one uniform 85px min/default section size
        to all 3 columns, which made the checkbox column far wider than
        the checkbox itself and pushed the table into horizontal
        overflow. This is pure view configuration (column widths +
        stylesheet), not structural UI, so it is set here at runtime
        instead.
        """
        table = self.w.possible_joins_table
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(18)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        table.setColumnWidth(0, 18)
        table.setColumnWidth(2, 58)
        table.setColumnWidth(3, 2 * BreakpointNavWidget.BUTTON_SIZE + 6)
        # CheckBoxDelegate paints AND hit-tests column 0's checkbox
        # itself (a Qt Style Sheet ``::indicator`` rule can paint one
        # geometry while the style's native click hit-test rect
        # disagrees, especially in a column this narrow -- the exact
        # "checkbox renders but can't be clicked" symptom). Keep a
        # strong reference so the delegate isn't garbage-collected out
        # from under the view.
        self._checkbox_delegate = CheckBoxDelegate(table)
        table.setItemDelegateForColumn(0, self._checkbox_delegate)

    def connect_signals(self):
        """Wire parent-widget buttons/spinboxes to handler entrypoints."""
        self.w.source_layers_comboBox.currentIndexChanged.connect(
            self._on_source_changed
        )
        self.w.showall_layers_pushButton.clicked.connect(self._show_all_layers)
        self.w.max_gap_time_spinbox.valueChanged.connect(
            self._on_thresholds_changed
        )
        self.w.max_gap_space_spinbox.valueChanged.connect(
            self._on_thresholds_changed
        )
        self.joins_table.check_state_changed.connect(
            self._on_check_state_changed
        )
        self.w.save_btn.clicked.connect(self.save)
        self.w.revert1_btn.clicked.connect(self.revert_last)
        self.w.reset_btn.clicked.connect(self.reset_all)

    #######################################################################
    # REFRESH / POPULATION
    #######################################################################

    def refresh_from_results(self):
        """(Re)populate the combobox/labels/table from ``self.results``.

        Called once at init, and again after every save/revert/reset
        (following a full reload of ``self.results``).
        """
        self._label_colors = {}
        self._track_positions = {}
        self._candidates = []
        self._source_track_id = None
        self._num_frames = self.results.num_frames if self.results else None
        # A fresh load/reload recreates the napari layers from scratch
        # (with their normal default head/tail lengths), so there is
        # nothing left to restore -- just drop the bookkeeping.
        self._full_traced_ids = set()
        self._original_trace_lengths = {}
        self._set_default_thresholds()

        combobox = self.w.source_layers_comboBox
        combobox.blockSignals(True)
        combobox.clear()
        if self.results is not None and self.results.track_id_label:
            self._label_colors = self._label_swatch_colors()
            self._track_positions = self._build_track_positions()
            entries = sorted(
                self.results.track_id_label.items(),
                key=lambda kv: (kv[1], kv[0]),
            )
            for track_id, label in entries:
                text = f"{label} (id {track_id})"
                color = self._label_colors.get(label)
                if color is not None:
                    icon = create_color_icon(_rgba_to_qcolor(color))
                    combobox.addItem(icon, text, track_id)
                else:
                    combobox.addItem(text, track_id)
        combobox.blockSignals(False)
        combobox.setCurrentIndex(-1)

        self.w.total_no_tracks_label.setText(str(combobox.count()))
        self.joins_table.clear()
        self._clear_breakpoint_widgets()
        self.w.coverage_percent_label.setText("%")
        self.w.increase_percent_label.setText("%")
        self.w.increase_percent_label.setStyleSheet("")
        self.w.save_btn.setEnabled(False)

        archives_exist = bool(self._list_archives())
        self.w.revert1_btn.setEnabled(archives_exist)
        self.w.reset_btn.setEnabled(archives_exist)

    def _set_default_thresholds(self):
        """Auto-set the gap thresholds from the current results' dimensions.

        Space: one third of the largest field-of-view dimension
        (width or height). Time: 10% of the total frame count. Both
        are plain integers, recomputed on every refresh (init and
        after every save/revert/reset) so they always reflect the
        currently loaded video.
        """
        if self.results is None:
            return
        if self._num_frames:
            default_time = max(1, round(self._num_frames * 0.1))
            self.w.max_gap_time_spinbox.setValue(int(default_time))
        width, height = self.results.width, self.results.height
        if width and height:
            default_space = max(1, round(max(width, height) / 3))
            self.w.max_gap_space_spinbox.setValue(int(default_space))

    def _label_swatch_colors(self):
        """Return ``{label: rgba_float_tuple}``, one color per label.

        Mirrors ``AnalysisResults.get_color_for_track_id()``'s color
        derivation, but always uses the first sub-color of a label's
        slice (occurrence index 0) instead of a per-track occurrence
        index, so every track sharing a label shows the same swatch.
        """
        results = self.results
        labels = sorted(set(results.track_id_label.values()))
        (
            all_labels_submaps,
            indices_max_diff_labels,
            indices_max_diff_subcolors,
        ) = results.define_colors()
        swatches = {}
        for i, label in enumerate(labels):
            original_class_id = None
            if results.classes:
                keys = [k for k, v in results.classes.items() if v == label]
                if keys:
                    original_class_id = int(keys[0])
            if original_class_id is None:
                original_class_id = i
            label_color_index = indices_max_diff_labels[
                original_class_id % len(indices_max_diff_labels)
            ]
            subcolor_index = indices_max_diff_subcolors[0]
            swatches[label] = tuple(
                all_labels_submaps[label_color_index][subcolor_index]
            )
        return swatches

    def _build_track_positions(self):
        """Return ``{track_id: {frame_idx: (pos_x, pos_y)}}`` (real data only).

        Used for nearest-endpoint distance checks in candidate
        filtering. Uses ``interpolate=False`` since only actually
        observed positions at track boundaries are meaningful here.
        """
        positions = {}
        tracking_data = self.results.get_tracking_data(interpolate=False)
        for track_id, info in tracking_data.items():
            df = info["data"]
            positions[track_id] = dict(
                zip(
                    df["frame_idx"].astype(int),
                    zip(df["pos_x"], df["pos_y"], strict=False),
                    strict=False,
                )
            )
        return positions

    #######################################################################
    # LAYER NAMING / VISIBILITY (mirrors AnalysisOctron.load_predictions())
    #######################################################################

    @staticmethod
    def _track_layer_name(label, track_id):
        return f"{label} - id {track_id}"

    @staticmethod
    def _mask_layer_name(label, track_id):
        return f"{label} - MASKS - id {track_id}"

    def _label_for(self, track_id):
        """Return the label for a track ID, or None if unknown/absent."""
        if track_id is None or self.results is None:
            return None
        return self.results.track_id_label.get(track_id)

    def _owned_layer_names(self):
        """Return every track/mask layer name load_predictions() can create.

        Layers outside this set (the source video, a "dummy mask"
        placeholder, or anything else the user added) are never touched
        by isolate/show-all -- only layers that OCTRON itself created
        for a track are toggled.
        """
        if self.results is None:
            return set()
        names = set()
        for track_id, label in self.results.track_id_label.items():
            names.add(self._track_layer_name(label, track_id))
            names.add(self._mask_layer_name(label, track_id))
        return names

    def _isolate_tracks(self, track_ids):
        """Show only the given tracks' own layers; leave other layers alone.

        Only layers OCTRON created for a track (Tracks/Labels layers,
        see :meth:`_owned_layer_names`) are hidden/shown here -- e.g.
        the source video layer is left untouched so it stays visible
        regardless of which source track is selected.
        """
        viewer = self.w._viewer
        if viewer is None or self.results is None:
            return
        owned_names = self._owned_layer_names()
        target_names = set()
        for tid in track_ids:
            label = self._label_for(tid)
            if label is None:
                continue
            target_names.add(self._track_layer_name(label, tid))
            target_names.add(self._mask_layer_name(label, tid))
        for layer in viewer.layers:
            if layer.name not in owned_names:
                continue
            layer.visible = layer.name in target_names
        for tid in track_ids:
            label = self._label_for(tid)
            if label is None:
                continue
            name = self._track_layer_name(label, tid)
            if name in viewer.layers:
                viewer.layers[name].color_by = "frame_idx"

    def _track_layer(self, track_id):
        """Return the napari Tracks layer for a track ID, or None."""
        viewer = self.w._viewer
        if viewer is None:
            return None
        label = self._label_for(track_id)
        if label is None:
            return None
        name = self._track_layer_name(label, track_id)
        if name not in viewer.layers:
            # napari's LayerList has no .get(): it's a name-keyed
            # sequence, not a dict.
            return None
        return viewer.layers[name]

    def _set_full_trace(self, track_ids):
        """Extend head/tail length to the full video for the given tracks.

        napari has no "infinite" trail option, so the closest equivalent
        is extending both ``head_length`` (frames ahead of the current
        one) and ``tail_length`` (frames behind it) to the video's full
        frame count -- long enough that the whole track is always drawn
        regardless of where the time slider currently sits.
        """
        if self.results is None or not self._num_frames:
            return
        for tid in track_ids:
            layer = self._track_layer(tid)
            if layer is None:
                continue
            layer.head_length = self._num_frames
            layer.tail_length = self._num_frames

    def _capture_original_trace(self, track_id):
        """Remember a track's current head/tail lengths, once, before override.

        A no-op if already captured (e.g. re-checking a candidate that
        was previously full-traced and reverted in this same session)
        so the stored value always reflects the true pre-override state.
        """
        if track_id in self._original_trace_lengths:
            return
        layer = self._track_layer(track_id)
        if layer is None:
            return
        self._original_trace_lengths[track_id] = (
            layer.head_length,
            layer.tail_length,
        )

    def _restore_original_trace(self, track_id):
        """Reset a track's head/tail lengths back to their captured values."""
        original = self._original_trace_lengths.get(track_id)
        if original is None:
            return
        layer = self._track_layer(track_id)
        if layer is None:
            return
        layer.head_length, layer.tail_length = original

    def _apply_full_trace_state(self, desired_ids):
        """Full-trace exactly ``desired_ids``; restore everyone else.

        ``desired_ids`` is the source track (once selected) plus every
        currently-checked candidate. Track IDs leaving that set have
        their head/tail lengths restored to what they were before this
        review session touched them; track IDs entering it have their
        current head/tail lengths captured first so they can be
        restored later.
        """
        desired = {tid for tid in desired_ids if tid is not None}
        for tid in self._full_traced_ids - desired:
            self._restore_original_trace(tid)
        for tid in desired - self._full_traced_ids:
            self._capture_original_trace(tid)
        self._set_full_trace(desired)
        self._full_traced_ids = desired

    def _show_all_layers(self):
        """Set every napari layer's visibility back to True."""
        viewer = self.w._viewer
        if viewer is None:
            return
        for layer in viewer.layers:
            layer.visible = True

    #######################################################################
    # SOURCE SELECTION / CANDIDATE FILTERING
    #######################################################################

    def _on_source_changed(self, index):
        """Handle a new source-track selection in the combobox."""
        if index < 0:
            return
        track_id = self.w.source_layers_comboBox.itemData(index)
        if track_id is None:
            return
        self._source_track_id = int(track_id)
        self._isolate_tracks([self._source_track_id])
        self._apply_full_trace_state([self._source_track_id])
        self._refresh_candidates()

    def _on_thresholds_changed(self, _value):
        """Recompute candidates when the gap/distance thresholds change."""
        self._refresh_candidates()

    def _csv_frames(self, track_id):
        """Return the set of frame indices this track has real data for."""
        if self.results is None:
            return set()
        return self.results._csv_frame_indices.get(track_id, set())

    def _track_endpoints(self, source_id, cand_id):
        """Return ``(src_endpoint, cand_endpoint)`` frames nearest each other.

        Whichever track's data ends first contributes its last frame;
        the other (which starts after the gap) contributes its first
        frame. None when the two tracks' frame ranges overlap (no
        single gap to bridge, so not eligible for a join).
        """
        source_frames = sorted(self._csv_frames(source_id))
        cand_frames = sorted(self._csv_frames(cand_id))
        if not source_frames or not cand_frames:
            return None
        if source_frames[-1] < cand_frames[0]:
            return source_frames[-1], cand_frames[0]
        if cand_frames[-1] < source_frames[0]:
            return source_frames[0], cand_frames[-1]
        return None  # Overlapping frame ranges -> not eligible

    def _gap_and_distance(self, source_id, cand_id):
        """Return ``(gap_frames, distance)`` between nearest endpoints.

        Returns None when the two tracks' frame ranges overlap (not
        eligible for a join) or when position data is missing.
        """
        endpoints = self._track_endpoints(source_id, cand_id)
        if endpoints is None:
            return None
        src_endpoint, cand_endpoint = endpoints
        gap_frames = abs(cand_endpoint - src_endpoint) - 1

        src_pos = self._track_positions.get(source_id, {}).get(src_endpoint)
        cand_pos = self._track_positions.get(cand_id, {}).get(cand_endpoint)
        if src_pos is None or cand_pos is None:
            return None
        distance = (
            (src_pos[0] - cand_pos[0]) ** 2 + (src_pos[1] - cand_pos[1]) ** 2
        ) ** 0.5
        return gap_frames, distance

    def _breakpoints_for(self, source_id, cand_id):
        """Return the two frames framing the join gap, sorted ascending.

        These are the "fusion points" the breakpoint-nav buttons jump
        between: where the earlier of the two tracks ends, and where
        the later one starts. None when the tracks overlap.
        """
        endpoints = self._track_endpoints(source_id, cand_id)
        return tuple(sorted(endpoints)) if endpoints is not None else None

    #######################################################################
    # BREAKPOINT NAVIGATION (column 3: jump the viewer to a join's gap)
    #######################################################################

    def _current_viewer_frame(self):
        """Return the viewer's current frame index, or None if unavailable."""
        viewer = self.w._viewer
        if viewer is None:
            return None
        try:
            return int(viewer.dims.current_step[0])
        except (IndexError, TypeError, AttributeError):
            return None

    def _jump_to_frame(self, frame):
        """Move the napari viewer's timeline to ``frame``."""
        viewer = self.w._viewer
        if viewer is None:
            return
        viewer.dims.set_point(0, frame)

    def _on_viewer_frame_changed(self, event=None):
        """Refresh every row's breakpoint-nav buttons for the new frame.

        Connected once (in ``__init__``) to the viewer's
        ``dims.events.current_step``, so scrubbing the timeline --
        including the jump triggered by clicking a nav button itself --
        keeps every row's enabled state/tooltip in sync automatically.
        """
        current_frame = self._current_viewer_frame()
        for widget in self._breakpoint_widgets:
            widget.set_current_frame(current_frame)

    def _clear_breakpoint_widgets(self):
        """Tear down all breakpoint-nav button widgets from the table.

        ``QTableView.setIndexWidget()`` is keyed by row/column, not by
        candidate identity, so the previous batch is torn down
        explicitly before a new one is created rather than relying on
        undocumented model-reset cleanup behavior.
        """
        for widget in self._breakpoint_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._breakpoint_widgets = []

    def _setup_breakpoint_buttons(self):
        """Create the forward/backward nav widget for each candidate row.

        Must run right after ``self.joins_table.set_candidates()`` --
        row indices only become valid once the model has that many rows.
        """
        table = self.w.possible_joins_table
        current_frame = self._current_viewer_frame()
        for row, candidate in enumerate(self._candidates):
            widget = BreakpointNavWidget()
            widget.set_breakpoints(candidate.get("breakpoints"))
            widget.set_current_frame(current_frame)
            widget.jump_requested.connect(self._jump_to_frame)
            table.setIndexWidget(self.joins_table.index(row, 3), widget)
            self._breakpoint_widgets.append(widget)

    def _refresh_candidates(self):
        """Repopulate the candidate table for the current source track."""
        self.joins_table.clear()
        self._clear_breakpoint_widgets()
        self.w.save_btn.setEnabled(False)
        if self._source_track_id is None or self.results is None:
            self.w.coverage_percent_label.setText("%")
            self.w.increase_percent_label.setText("%")
            self.w.increase_percent_label.setStyleSheet("")
            return

        source_id = self._source_track_id
        label = self.results.track_id_label.get(source_id)
        source_frames = self._csv_frames(source_id)
        coverage = (
            100.0 * len(source_frames) / self._num_frames
            if self._num_frames
            else 0.0
        )
        self.w.coverage_percent_label.setText(f"{coverage:.1f}%")
        self.w.increase_percent_label.setText("0.0%")
        self.w.increase_percent_label.setStyleSheet(
            f"color: {coverage_gain_color(0.0).name()};"
        )

        max_gap_time = self.w.max_gap_time_spinbox.value()
        max_gap_space = self.w.max_gap_space_spinbox.value()

        candidates = []
        for track_id, tid_label in self.results.track_id_label.items():
            if track_id == source_id or tid_label != label:
                continue
            gap_info = self._gap_and_distance(source_id, track_id)
            if gap_info is None:
                continue
            gap_frames, distance = gap_info
            if gap_frames > max_gap_time or distance > max_gap_space:
                continue
            candidates.append(
                {
                    "track_id": track_id,
                    "name": self._track_layer_name(tid_label, track_id),
                    "breakpoints": self._breakpoints_for(source_id, track_id),
                }
            )
        self._candidates = candidates
        self.joins_table.set_candidates(candidates)
        self._setup_breakpoint_buttons()
        self._recompute_increase_values()

    def _on_check_state_changed(self):
        """React to a checkbox toggle: update visibility/labels/save_btn."""
        checked_ids = self.joins_table.checked_track_ids()
        self._recompute_increase_values()
        visible_ids = [self._source_track_id, *checked_ids]
        self._isolate_tracks(visible_ids)
        self._apply_full_trace_state(visible_ids)
        self.w.save_btn.setEnabled(bool(checked_ids))

    def _recompute_increase_values(self):
        """Update per-row +coverage% and the total increase_percent_label."""
        if self._source_track_id is None:
            return
        source_frames = self._csv_frames(self._source_track_id)
        checked_ids = set(self.joins_table.checked_track_ids())

        increase_by_id = {}
        for cand in self._candidates:
            tid = cand["track_id"]
            others = set(source_frames)
            for other_id in checked_ids:
                if other_id == tid:
                    continue
                others |= self._csv_frames(other_id)
            marginal = self._csv_frames(tid) - others
            increase_by_id[tid] = (
                100.0 * len(marginal) / self._num_frames
                if self._num_frames
                else 0.0
            )
        self.joins_table.set_increase_percentages(increase_by_id)

        union_frames = set(source_frames)
        for cid in checked_ids:
            union_frames |= self._csv_frames(cid)
        total_increase = (
            100.0 * len(union_frames - source_frames) / self._num_frames
            if self._num_frames
            else 0.0
        )
        self.w.increase_percent_label.setText(f"{total_increase:.1f}%")
        self.w.increase_percent_label.setStyleSheet(
            f"color: {coverage_gain_color(total_increase).name()};"
        )

    #######################################################################
    # SAVE (immediate on-disk fusion + archival)
    #######################################################################

    def _next_fused_id(self):
        """Return a new, never-before-used track ID from live on-disk state."""
        ids = list(self.results.track_id_label.keys())
        return (max(ids) + 1) if ids else 1

    def _archive_root(self):
        return self.save_dir / "fused_originals"

    def _csv_path_for_track(self, track_id):
        """Locate the on-disk CSV for a track ID via AnalysisResults.csvs."""
        if not self.results.csvs:
            return None
        for p in self.results.csvs:
            m = re.search(r"_track_(\d+)\.csv$", p.name)
            if m and int(m.group(1)) == track_id:
                return p
        return None

    def save(self):
        """Fuse the source + checked candidates into a new track, on disk.

        This is an immediate, atomic disk operation: no separate "bake"
        step. Originals are archived (not deleted) under
        ``fused_originals/<timestamp>_id<fused_id>/`` together with a
        ``manifest.json`` that doubles as the undo log for
        :meth:`revert_last`/:meth:`reset_all`.
        """
        if self.results is None or self.save_dir is None:
            return
        checked_ids = self.joins_table.checked_track_ids()
        if self._source_track_id is None or not checked_ids:
            return

        source_id = self._source_track_id
        member_ids = [source_id, *checked_ids]
        label = self.results.track_id_label.get(source_id)
        if label is None:
            show_error("Could not determine label for the source track.")
            return

        member_csv_paths = {}
        for tid in member_ids:
            csv_path = self._csv_path_for_track(tid)
            if csv_path is None or not csv_path.exists():
                show_error(f"Could not locate CSV for track id {tid}.")
                return
            member_csv_paths[tid] = csv_path

        fused_id = self._next_fused_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_dir = self._archive_root() / f"{timestamp}_id{fused_id}"

        try:
            csv_header_lines = self.results.csv_header_lines
            dfs = []
            raw_header_ref = None
            for tid in member_ids:
                csv_path = member_csv_paths[tid]
                with open(csv_path) as f:
                    raw_header = [next(f) for _ in range(csv_header_lines)]
                if raw_header_ref is None:
                    raw_header_ref = raw_header
                df = pd.read_csv(csv_path, skiprows=csv_header_lines)
                df["track_id"] = fused_id
                dfs.append(df)

            fused_df = pd.concat(dfs, ignore_index=True)
            fused_df = fused_df.sort_values("frame_idx").reset_index(drop=True)
            fused_df["frame_counter"] = range(len(fused_df))

            fused_csv_path = self.save_dir / f"{label}_track_{fused_id}.csv"
            first_five = [line.rstrip("\n") for line in raw_header_ref[:5]]
            header_out = [*first_five, f"created_at: {datetime.now()}", ""]
            with open(fused_csv_path, "w") as f:
                f.write("\n".join(header_out) + "\n")
                fused_df.to_csv(
                    f, index=False, na_rep="NaN", lineterminator="\n"
                )

            mask_arrays_renamed = self._fuse_masks(member_ids, fused_id, label)

            # Archive originals AFTER the fused CSV/masks were written
            # successfully, so a failure above leaves the originals in
            # place rather than orphaned in the archive.
            archive_dir.mkdir(parents=True, exist_ok=True)
            for tid in member_ids:
                csv_path = member_csv_paths[tid]
                shutil.move(
                    str(csv_path), str(archive_dir / f"{csv_path.name}.bak")
                )

            manifest = {
                "fused_id": fused_id,
                "label": label,
                "member_track_ids": member_ids,
                "mask_arrays_renamed": mask_arrays_renamed,
                "timestamp": timestamp,
            }
            with open(archive_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Fusing tracks {member_ids} into {fused_id}: {e}")
            show_error(f"Could not fuse tracks: {e}")
            return

        show_info(
            f"Fused {len(member_ids)} tracks into '{label}' (id {fused_id})."
        )
        self._reload()

    def _fuse_masks(self, member_ids, fused_id, label):
        """Create the fused mask array and rename originals in-place.

        Returns the ``mask_arrays_renamed`` list for the manifest (empty
        when this results dir has no masks / is detection-only).
        """
        if not self.results.has_masks or self.results.zarr_root is None:
            return []

        zarr_root_path = Path(self.results.zarr)
        store = zarr.storage.LocalStore(zarr_root_path, read_only=False)
        shape = (
            self.results.num_frames,
            self.results.height,
            self.results.width,
        )
        fused_array = create_prediction_zarr(
            store,
            f"{fused_id}_masks",
            shape=shape,
            chunk_size=1,
            fill_value=-1,
            dtype="int8",
            video_hash="",
        )
        fused_array.attrs["label"] = label

        all_frame_indices = []
        classes_attr = None
        for tid in member_ids:
            member_key = f"{tid}_masks"
            if member_key not in self.results.zarr_root.array_keys():
                continue
            member_array = self.results.zarr_root[member_key]
            if classes_attr is None:
                classes_attr = member_array.attrs.get("classes")
            frame_indices = get_annotated_frames(member_array)
            for f in frame_indices:
                fused_array[int(f), :, :] = member_array[int(f), :, :]
            all_frame_indices.extend(int(f) for f in frame_indices)
        if classes_attr:
            fused_array.attrs["classes"] = classes_attr
        mark_frames_annotated(fused_array, sorted(set(all_frame_indices)))

        # Rename originals in place. get_mask_data() only looks up
        # f"{track_id}_masks" for track IDs still present in the
        # current CSV set, so a renamed array becomes invisible without
        # needing an extension trick (unlike the CSV .bak rename).
        # NOTE: the new name must NOT start with "<digits>_", since
        # AnalysisResults._track_ids_zarr() treats ANY zarr array key
        # of that shape as a live track ID (it only looks at the token
        # before the first underscore) -- a suffix-style rename such as
        # f"{tid}_masks_fused_orig" would still be picked up as track
        # id `tid`. A prefix keeps it invisible to that scan.
        mask_arrays_renamed = []
        for tid in member_ids:
            old_dir = zarr_root_path / f"{tid}_masks"
            if not old_dir.exists():
                continue
            new_name = f"fused_orig_{tid}_masks"
            shutil.move(str(old_dir), str(zarr_root_path / new_name))
            mask_arrays_renamed.append(
                {"from": f"{tid}_masks", "to": new_name}
            )
        return mask_arrays_renamed

    #######################################################################
    # REVERT / RESET
    #######################################################################

    def _list_archives(self):
        """Return archive folders sorted oldest-first (timestamp-prefixed)."""
        if self.save_dir is None:
            return []
        root = self._archive_root()
        if not root.exists():
            return []
        return sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )

    def revert_last(self):
        """Undo the most recent save (restore originals, drop fused track)."""
        archives = self._list_archives()
        if not archives:
            return
        archive_dir = archives[-1]
        try:
            with open(archive_dir / "manifest.json") as f:
                manifest = json.load(f)
            fused_id = manifest["fused_id"]
            label = manifest["label"]

            for bak_file in archive_dir.glob("*.csv.bak"):
                original_name = bak_file.name[: -len(".bak")]
                shutil.move(str(bak_file), str(self.save_dir / original_name))

            fused_csv = self.save_dir / f"{label}_track_{fused_id}.csv"
            if fused_csv.exists():
                fused_csv.unlink()

            has_renamed = manifest.get("mask_arrays_renamed")
            if has_renamed and self.results is not None:
                zarr_root_path = (
                    Path(self.results.zarr)
                    if self.results.zarr is not None
                    else None
                )
                if zarr_root_path is not None:
                    for entry in manifest["mask_arrays_renamed"]:
                        renamed_dir = zarr_root_path / entry["to"]
                        original_dir = zarr_root_path / entry["from"]
                        if renamed_dir.exists():
                            shutil.move(str(renamed_dir), str(original_dir))
                    fused_mask_dir = zarr_root_path / f"{fused_id}_masks"
                    if fused_mask_dir.exists():
                        shutil.rmtree(fused_mask_dir)

            shutil.rmtree(archive_dir)
        except Exception as e:
            logger.error(f"Reverting archive '{archive_dir.name}': {e}")
            show_error(f"Could not revert last fuse operation: {e}")
            return

        show_info(f"Reverted fused track (id {manifest.get('fused_id')}).")
        self._reload()

    def reset_all(self):
        """Undo every save this session, back to the pristine on-disk state."""
        if not self._list_archives():
            return
        while self._list_archives():
            before = len(self._list_archives())
            self.revert_last()
            # Guard against an infinite loop if revert_last() failed to
            # remove the archive it was processing (it already reported
            # the error to the user).
            if len(self._list_archives()) >= before:
                break
        show_warning("Reset complete.")

    #######################################################################
    # RELOAD
    #######################################################################

    def _reload(self):
        """Clear and repopulate the viewer in place after a fuse/revert."""
        viewer = self.w._viewer
        if viewer is None or self.save_dir is None:
            return
        from octron.analysis_octron.analysis_octron import AnalysisOctron

        viewer.layers.clear()
        for _ in AnalysisOctron().load_predictions(
            self.save_dir, viewer=viewer, show_cleaner_widget=False
        ):
            pass
        self.results = AnalysisResults(self.save_dir)
        self.refresh_from_results()
