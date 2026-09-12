"""Table model for the prediction cleaner's candidate-join list.

Mirrors ``octron/gui_tables.py::ExistingDataTable``'s structure/pattern
(headers list, ``_data`` list of rows, ``beginResetModel``/
``endResetModel`` for bulk updates), but adds a checkable first column
since candidates here are selected for a join rather than just listed.
"""

from qtpy.QtCore import QAbstractTableModel, QEvent, QRect, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QHBoxLayout,
    QStyledItemDelegate,
    QToolButton,
    QWidget,
)

# Coverage-gain color scale: neutral grey (0%, no benefit) fading to
# lemon green (100%, maximal benefit). Grey is chosen because it reads
# clearly on both light and dark napari themes; used by both the +Cov%
# table column and CleanerHandler's increase_percent_label.
_NEUTRAL_GREY = (136, 136, 136)
_LEMON_GREEN = (154, 205, 50)
LEMON_GREEN_HEX = "#{:02x}{:02x}{:02x}".format(*_LEMON_GREEN)


def coverage_gain_color(percent):
    """Interpolate a QColor between neutral grey (0%) and lemon green (100%).

    ``percent`` is clamped to [0, 100] first, so out-of-range values
    (e.g. floating point noise) still produce a valid color.
    """
    t = max(0.0, min(1.0, percent / 100.0))
    channels = (
        round(grey + (lemon - grey) * t)
        for grey, lemon in zip(_NEUTRAL_GREY, _LEMON_GREEN, strict=True)
    )
    return QColor(*channels)


def _event_pos(event):
    """Return a mouse event's position as a QPoint across Qt bindings.

    ``QMouseEvent.position()`` (QPointF) is the current API; ``pos()``
    is the older one some bindings/versions still rely on.
    """
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


class CheckBoxDelegate(QStyledItemDelegate):
    """Paint and hit-test a checkbox indicator explicitly (column 0).

    A Qt Style Sheet ``::indicator`` rule can paint a checkbox at one
    geometry while the style's internal click hit-test rectangle (used
    by the default delegate's ``editorEvent()``) is computed
    independently and can disagree -- especially once a column is
    narrower than the style's usual metrics, as it is here. The
    reported symptom is exactly that: a checkbox that renders but never
    responds to clicks. Painting and hit-testing the same explicit
    rectangle here removes that ambiguity entirely.
    """

    BOX_SIZE = 13

    def _box_rect(self, cell_rect):
        x = cell_rect.x() + (cell_rect.width() - self.BOX_SIZE) // 2
        y = cell_rect.y() + (cell_rect.height() - self.BOX_SIZE) // 2
        return QRect(x, y, self.BOX_SIZE, self.BOX_SIZE)

    def paint(self, painter, option, index):
        """Draw a small rounded checkbox instead of the native indicator."""
        if index.column() != 0:
            super().paint(painter, option, index)
            return
        checked = index.data(Qt.CheckStateRole) == Qt.Checked
        rect = self._box_rect(option.rect)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if checked:
            color = QColor(LEMON_GREEN_HEX)
            painter.setBrush(color)
            painter.setPen(color)
        else:
            painter.setBrush(QColor(128, 128, 128, 40))
            painter.setPen(QColor(128, 128, 128, 160))
        painter.drawRoundedRect(rect, 3, 3)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        """Toggle the checkbox on a left-click release inside its box."""
        if index.column() != 0 or not (index.flags() & Qt.ItemIsUserCheckable):
            return super().editorEvent(event, model, option, index)
        if event.type() not in (
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        ):
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self._box_rect(option.rect).contains(_event_pos(event)):
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            current = index.data(Qt.CheckStateRole)
            new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            model.setData(index, new_state, Qt.CheckStateRole)
        return True  # Swallow so a double-click doesn't toggle it twice.

    def createEditor(self, parent, option, index):
        """No editor widget -- the checkbox is fully handled in paint/click."""
        return None


class BreakpointNavWidget(QWidget):
    """Two tiny buttons that jump the viewer to a candidate's join breakpoints.

    A "breakpoint" is one of the two frames framing the temporal gap
    between the source track and this candidate: where the earlier of
    the two tracks ends, and where the later one starts. The backward
    (``<``) / forward (``>``) buttons step the napari viewer's current
    frame to the nearest breakpoint in that direction and auto-disable
    (with a tooltip naming the target frame) once nothing is left to
    jump to in that direction. This widget knows nothing about napari
    or ``CleanerHandler`` -- it just tracks breakpoints/current frame
    and emits :attr:`jump_requested` with the target frame to jump to.
    """

    jump_requested = Signal(int)

    BUTTON_SIZE = 16

    def __init__(self, parent=None):
        """Build the two nav buttons in a tight horizontal layout."""
        super().__init__(parent)
        self._breakpoints = ()
        self._current_frame = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)
        self.backward_btn = QToolButton(self)
        self.backward_btn.setText("<")
        self.backward_btn.setAutoRaise(True)
        self.backward_btn.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
        self.forward_btn = QToolButton(self)
        self.forward_btn.setText(">")
        self.forward_btn.setAutoRaise(True)
        self.forward_btn.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
        layout.addWidget(self.backward_btn)
        layout.addWidget(self.forward_btn)

        self.backward_btn.clicked.connect(lambda: self._on_clicked(-1))
        self.forward_btn.clicked.connect(lambda: self._on_clicked(1))
        self._refresh()

    def set_breakpoints(self, breakpoints):
        """Replace the (up to two) frame indices this row can jump to."""
        self._breakpoints = tuple(breakpoints or ())
        self._refresh()

    def set_current_frame(self, frame):
        """Update the viewer's current frame and re-evaluate button state."""
        self._current_frame = frame
        self._refresh()

    def _target(self, direction):
        """Return the nearest breakpoint in ``direction`` (>0 fwd, <0 back)."""
        if self._current_frame is None or not self._breakpoints:
            return None
        if direction > 0:
            ahead = [b for b in self._breakpoints if b > self._current_frame]
            return min(ahead) if ahead else None
        behind = [b for b in self._breakpoints if b < self._current_frame]
        return max(behind) if behind else None

    def _refresh(self):
        back_target = self._target(-1)
        fwd_target = self._target(1)
        self.backward_btn.setEnabled(back_target is not None)
        self.forward_btn.setEnabled(fwd_target is not None)
        self.backward_btn.setToolTip(
            f"Jump to frame {back_target}"
            if back_target is not None
            else "No earlier breakpoint"
        )
        self.forward_btn.setToolTip(
            f"Jump to frame {fwd_target}"
            if fwd_target is not None
            else "No later breakpoint"
        )

    def _on_clicked(self, direction):
        target = self._target(direction)
        if target is not None:
            self.jump_requested.emit(target)


class PossibleJoinsTableModel(QAbstractTableModel):
    """List candidate tracks that could be joined onto a source track.

    Columns
    -------
    0 : checkbox (no header text) -- whether this candidate is
        currently selected for inclusion in the join.
    1 : layer name, elided for display; the full name is shown as a
        tooltip on hover.
    2 : incremental ("+coverage") percentage this row would add to the
        join, given whichever other rows are already checked. This is
        marginal/new coverage only (candidates can overlap each other
        even though each is independently exclusive with the source),
        so it is recomputed by the handler on every checkbox toggle via
        :meth:`set_increase_percentages`.

    The model itself has no notion of frame coverage -- that requires
    per-track frame-index bookkeeping the handler already owns via
    ``AnalysisResults``. Toggling a checkbox emits
    ``check_state_changed`` so the handler can recompute every row's
    incremental percentage (plus the total union gain) and push new
    values back in.
    """

    check_state_changed = Signal()

    def __init__(self):
        """Initialize an empty candidate table."""
        super().__init__()
        # Column 3 (breakpoint nav) has no model-driven cell content --
        # it's fully covered by a BreakpointNavWidget the handler places
        # via QTableView.setIndexWidget() after each set_candidates().
        self.headers = ["", "Layer", "+Cov%", ""]
        self._data = []  # list of dicts: track_id, name, checked, increase_pct

    def set_candidates(self, candidates):
        """Replace the candidate list.

        Parameters
        ----------
        candidates : list of dict
            Each dict needs at least ``track_id`` and ``name``.
            ``checked`` defaults to False and ``increase_pct`` to 0.0
            when not given.

        """
        self.beginResetModel()
        self._data = [
            {
                "track_id": c["track_id"],
                "name": c["name"],
                "checked": c.get("checked", False),
                "increase_pct": c.get("increase_pct", 0.0),
            }
            for c in candidates
        ]
        self.endResetModel()

    def clear(self):
        """Remove all rows."""
        self.set_candidates([])

    def checked_track_ids(self):
        """Return the track IDs of all currently checked rows."""
        return [row["track_id"] for row in self._data if row["checked"]]

    def set_increase_percentages(self, increase_by_track_id):
        """Update column 2 for every row from a ``{track_id: pct}`` mapping.

        Rows whose track ID is missing from the mapping keep their
        current value.
        """
        if not self._data:
            return
        for row in self._data:
            if row["track_id"] in increase_by_track_id:
                row["increase_pct"] = increase_by_track_id[row["track_id"]]
        top_left = self.index(0, 2)
        bottom_right = self.index(len(self._data) - 1, 2)
        self.dataChanged.emit(top_left, bottom_right, [Qt.DisplayRole])

    def rowCount(self, parent=None):
        """Return the number of candidate rows."""
        return len(self._data)

    def columnCount(self, parent=None):
        """Return the number of columns (checkbox, name, +coverage)."""
        return len(self.headers)

    def flags(self, index):
        """Return item flags, making column 0 checkable."""
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemIsUserCheckable
        return base

    def data(self, index, role=Qt.DisplayRole):
        """Return the data for the given index and role."""
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None
        row = self._data[index.row()]
        col = index.column()

        if role == Qt.CheckStateRole and col == 0:
            return Qt.Checked if row["checked"] else Qt.Unchecked

        if role == Qt.DisplayRole:
            if col == 1:
                name = row["name"]
                return name if len(name) <= 22 else f"{name[:19]}..."
            if col == 2:
                return f"{row['increase_pct']:+.1f}%"
            return None

        if role == Qt.ToolTipRole and col == 1:
            return row["name"]

        if role == Qt.TextAlignmentRole and col == 1:
            return Qt.AlignLeft | Qt.AlignVCenter

        if role == Qt.TextAlignmentRole and col == 2:
            return Qt.AlignRight | Qt.AlignVCenter

        if role == Qt.ForegroundRole and col == 2:
            return coverage_gain_color(row["increase_pct"])

        return None

    def setData(self, index, value, role=Qt.EditRole):
        """Handle checkbox toggles for column 0."""
        if role == Qt.CheckStateRole and index.column() == 0:
            row = self._data[index.row()]
            row["checked"] = value == Qt.Checked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            self.check_state_changed.emit()
            return True
        return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Return the header label (and its alignment) for a section."""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        if (
            orientation == Qt.Horizontal
            and role == Qt.TextAlignmentRole
            and section == 1
        ):
            return Qt.AlignLeft | Qt.AlignVCenter
        return None
