"""A PDF page canvas with selectable, movable and resizable line boxes."""
from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


def pil_pixmap(image) -> QPixmap:
    image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    return QPixmap.fromImage(QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy())


def bounds(polygon) -> QRectF:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def rectangle_polygon(rect: QRectF) -> list[list[float]]:
    return [[rect.left(), rect.top()], [rect.right(), rect.top()], [rect.right(), rect.bottom()], [rect.left(), rect.bottom()]]


class LineBox(QGraphicsRectItem):
    def __init__(self, line, canvas: "PageCanvas"):
        box = bounds(line.polygon)
        super().__init__(QRectF(0, 0, max(4, box.width()), max(4, box.height())))
        self.line_id = line.id
        self.line_status = line.status
        self.canvas = canvas
        self.setPos(box.topLeft())
        self.setZValue(2)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self._resize_edges = ""
        self._start = QRectF()
        self._press = QPointF()
        self.setToolTip("单击选择；拖动移动；拖动边缘调整大小。按 Ctrl 可多选。")

    def paint(self, painter, option, widget=None):
        selected = self.isSelected()
        color = QColor("#236fd1") if selected else QColor("#b97820" if self.line_status in {"stale", "pending", "error", "review"} else "#278978")
        painter.setPen(QPen(color, 2.5 if selected else 1.5, Qt.PenStyle.SolidLine))
        fill = QColor(color)
        fill.setAlpha(32 if selected else 14)
        painter.setBrush(fill)
        painter.drawRect(self.rect())
        if selected:
            size = 7 / max(self.canvas.transform().m11(), 0.1)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for corner in [self.rect().topLeft(), self.rect().topRight(), self.rect().bottomLeft(), self.rect().bottomRight()]:
                painter.drawRect(QRectF(corner.x() - size / 2, corner.y() - size / 2, size, size))

    def _edges(self, point: QPointF) -> str:
        margin = 9 / max(self.canvas.transform().m11(), 0.1)
        rect = self.rect()
        margin_x = min(margin, rect.width() * 0.2)
        margin_y = min(margin, rect.height() * 0.23)
        result = ""
        if abs(point.x() - rect.left()) < margin_x:
            result += "l"
        elif abs(point.x() - rect.right()) < margin_x:
            result += "r"
        if abs(point.y() - rect.top()) < margin_y:
            result += "t"
        elif abs(point.y() - rect.bottom()) < margin_y:
            result += "b"
        return result

    def hoverMoveEvent(self, event):
        if not self.canvas.editable:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        edges = self._edges(event.pos())
        if len(edges) == 2:
            cursor = Qt.CursorShape.SizeFDiagCursor if edges in {"lt", "rb"} else Qt.CursorShape.SizeBDiagCursor
        elif edges in {"l", "r"}:
            cursor = Qt.CursorShape.SizeHorCursor
        elif edges in {"t", "b"}:
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if not self.canvas.editable:
            if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.scene().clearSelection()
            self.setSelected(True)
            event.accept()
            return
        self._start = self.mapRectToScene(self.rect())
        self._press = event.scenePos()
        self._resize_edges = self._edges(event.pos()) if self.isSelected() else ""
        self.canvas._drag_starts = {line_id: box.mapRectToScene(box.rect()) for line_id, box in self.canvas._boxes.items()}
        if self._resize_edges:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.canvas.editable:
            return
        if self._resize_edges:
            delta = event.scenePos() - self._press
            rect = QRectF(self._start)
            if "l" in self._resize_edges:
                rect.setLeft(min(rect.right() - 8, rect.left() + delta.x()))
            if "r" in self._resize_edges:
                rect.setRight(max(rect.left() + 8, rect.right() + delta.x()))
            if "t" in self._resize_edges:
                rect.setTop(min(rect.bottom() - 8, rect.top() + delta.y()))
            if "b" in self._resize_edges:
                rect.setBottom(max(rect.top() + 8, rect.bottom() + delta.y()))
            rect = rect.intersected(self.canvas.page_rect)
            self.setPos(rect.topLeft())
            self.setRect(0, 0, rect.width(), rect.height())
            event.accept()
        else:
            super().mouseMoveEvent(event)
            rect = self.mapRectToScene(self.rect())
            x = min(max(0, rect.left()), max(0, self.canvas.page_rect.width() - rect.width()))
            y = min(max(0, rect.top()), max(0, self.canvas.page_rect.height() - rect.height()))
            self.setPos(x, y)

    def mouseReleaseEvent(self, event):
        if not self.canvas.editable:
            event.accept()
            return
        if not self._resize_edges:
            super().mouseReleaseEvent(event)
        self._resize_edges = ""
        changes = []
        for line_id, start in self.canvas._drag_starts.items():
            box = self.canvas._boxes.get(line_id)
            if box is None:
                continue
            rect = box.mapRectToScene(box.rect())
            if start.isValid() and (abs(rect.x() - start.x()) + abs(rect.y() - start.y()) + abs(rect.width() - start.width()) + abs(rect.height() - start.height()) > 1):
                rect = rect.intersected(self.canvas.page_rect)
                changes.append((line_id, rectangle_polygon(rect)))
        self.canvas._drag_starts = {}
        if len(changes) == 1:
            line_id, polygon = changes[0]
            QTimer.singleShot(0, lambda: self.canvas.line_rect_changed.emit(line_id, polygon))
        elif changes:
            QTimer.singleShot(0, lambda: self.canvas.line_rects_changed.emit(changes))


class PageCanvas(QGraphicsView):
    lines_selected = Signal(list)
    line_rect_changed = Signal(str, list)
    line_rects_changed = Signal(list)
    new_rect_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageCanvas")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#e2e8f0"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.page_rect = QRectF()
        self.editable = True
        self._boxes: dict[str, LineBox] = {}
        self._pixmap_item = None
        self._suppress_selection = False
        self._add_mode = False
        self._draw_start = None
        self._drag_starts = {}
        self._rubber_item = None
        self._scene.selectionChanged.connect(self._selection_changed)

    def _selection_changed(self):
        if not self._suppress_selection:
            self.lines_selected.emit(self.selected_ids())

    def selected_ids(self) -> list[str]:
        return [item.line_id for item in self._scene.selectedItems() if isinstance(item, LineBox)]

    def set_page(self, image, lines, selected_ids: Iterable[str] = (), reset_view=True):
        self._suppress_selection = True
        self._scene.clear()
        self._boxes = {}
        self._rubber_item = None
        self._draw_start = None
        self._pixmap_item = self._scene.addPixmap(pil_pixmap(image))
        self._pixmap_item.setZValue(0)
        self.page_rect = QRectF(0, 0, image.width, image.height)
        self._scene.setSceneRect(self.page_rect.adjusted(-35, -35, 35, 35))
        for line in lines:
            box = LineBox(line, self)
            self._scene.addItem(box)
            self._boxes[line.id] = box
        for line_id in selected_ids:
            if line_id in self._boxes:
                self._boxes[line_id].setSelected(True)
        self._suppress_selection = False
        if reset_view:
            self.fit_page()

    def update_lines(self, lines, selected_ids: Iterable[str] = ()):
        self._suppress_selection = True
        for box in self._boxes.values():
            self._scene.removeItem(box)
        self._boxes = {}
        for line in lines:
            box = LineBox(line, self)
            self._scene.addItem(box)
            self._boxes[line.id] = box
        for line_id in selected_ids:
            if line_id in self._boxes:
                self._boxes[line_id].setSelected(True)
        self._suppress_selection = False

    def select_lines(self, ids: Iterable[str], center=True):
        ids = set(ids)
        self._suppress_selection = True
        for line_id, box in self._boxes.items():
            box.setSelected(line_id in ids)
        self._suppress_selection = False
        if center and ids:
            selected = [box.sceneBoundingRect() for line_id, box in self._boxes.items() if line_id in ids]
            if selected:
                rect = selected[0]
                for extra in selected[1:]:
                    rect = rect.united(extra)
                self.ensureVisible(rect.adjusted(-35, -35, 35, 35), 35, 35)

    def set_add_mode(self, enabled: bool):
        self._add_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.RubberBandDrag)

    def fit_page(self):
        if self.page_rect.isValid():
            self.fitInView(self.page_rect.adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float):
        target = self.transform().m11() * factor
        if 0.04 <= target <= 5:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.18 if event.angleDelta().y() > 0 else 1 / 1.18)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self._add_mode and self.editable and event.button() == Qt.MouseButton.LeftButton and self.page_rect.isValid():
            self._draw_start = self.mapToScene(event.position().toPoint())
            self._rubber_item = self._scene.addRect(QRectF(self._draw_start, self._draw_start), QPen(QColor("#236fd1"), 2))
            self._rubber_item.setZValue(5)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_start is not None and self._rubber_item is not None:
            rect = QRectF(self._draw_start, self.mapToScene(event.position().toPoint())).normalized().intersected(self.page_rect)
            self._rubber_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_start is not None and self._rubber_item is not None:
            rect = self._rubber_item.rect()
            self._scene.removeItem(self._rubber_item)
            self._rubber_item = None
            self._draw_start = None
            self.set_add_mode(False)
            if rect.width() >= 12 and rect.height() >= 8:
                self.new_rect_requested.emit(rectangle_polygon(rect))
            event.accept()
        else:
            super().mouseReleaseEvent(event)
