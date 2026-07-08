# Add colored squares with QIcon
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap


def create_color_icon(color, size=16):
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background
    painter = QPainter(pixmap)
    painter.setPen(QColor(0, 0, 0, 0))  # Transparent outline
    painter.setBrush(QColor(color))  # Fill color
    painter.drawRect(0, 0, size - 1, size - 1)  # Draw square
    painter.end()
    return QIcon(pixmap)
