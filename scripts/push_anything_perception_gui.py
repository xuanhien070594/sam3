import sys
import os
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QMessageBox,
    QDialog,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen
import numpy as np
import trimesh
import psutil
import subprocess

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from shapely.geometry import Point, Polygon
import math
import matplotlib.patches as patches
from loguru import logger
from PIL import Image

from object_detection_and_segmentation import scan_objects
from object_state_subscriber import ObjectStateSubscriber
from target_poses_publisher import TargetPosesPublisher
from controller_command_sender import ControllerCommandSenser


class WaitingSpinnerWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, size: int = 72) -> None:
        super().__init__(parent)
        self._rotation_deg = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)
        self.setFixedSize(size, size)

    def _tick(self) -> None:
        self._rotation_deg = (self._rotation_deg + 15.0) % 360.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx = self.width() * 0.5
        cy = self.height() * 0.5
        outer_r = min(self.width(), self.height()) * 0.5 - 4
        inner_r = outer_r * 0.55
        n = 12
        for i in range(n):
            t = (i / float(n)) * 2.0 * math.pi
            angle = t + math.radians(self._rotation_deg)
            opacity = 0.25 + 0.75 * (1.0 - (i / float(max(n - 1, 1))))
            painter.setOpacity(opacity)
            pen = QPen(QColor(33, 150, 243))
            pen.setWidth(4)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            x1 = cx + inner_r * math.cos(angle)
            y1 = cy + inner_r * math.sin(angle)
            x2 = cx + outer_r * math.cos(angle)
            y2 = cy + outer_r * math.sin(angle)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class ScanningDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scanning and Tracking")
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        layout = QVBoxLayout(self)
        self._msg_label = QLabel("Scanning the scene and detecting objects")
        self._msg_label.setAlignment(Qt.AlignCenter)
        f = self._msg_label.font()
        f.setPointSize(max(f.pointSize(), 18))
        self._msg_label.setFont(f)
        layout.addWidget(self._msg_label)
        layout.addSpacing(12)
        self._spinner = WaitingSpinnerWidget(self)
        layout.addWidget(self._spinner, alignment=Qt.AlignCenter)
        layout.addSpacing(8)
        self.setMinimumWidth(360)

    def set_message(self, text: str) -> None:
        self._msg_label.setText(text)


class ScanThread(QThread):
    finished_ok = pyqtSignal(object, object, object)
    finished_err = pyqtSignal(str)

    def __init__(self, gui: "PushAnythingPerceptionGUI") -> None:
        super().__init__(None)
        self._gui = gui

    def run(self) -> None:
        try:
            img_rgb, names, boxes = self._gui._perform_scan_computation()
            self.finished_ok.emit(img_rgb, names, boxes)
        except Exception as e:
            logger.exception("Scan thread failed: {}", e)
            self.finished_err.emit(str(e))


# One color per object (by order in the scene); used for both current and goal boxes.
OBJECT_EDGE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
)

WORKSPACE_X_LIMIT = (0.35, 0.65)
WORKSPACE_Y_LIMIT = (-0.4, 0.4)

# Plot: robot +Y → +plot x; robot +X → downward on screen via invert_yaxis (py = rx > 0).
PLOT_XLIM = (0.3, 0.7)  # robot X → plot y (positive ticks)
PLOT_YLIM = (-0.5, 0.5)  # robot Y → plot x

GOAL_X_RANGE = (-0.5, 1.0)
GOAL_Y_RANGE = (-0.75, 0.75)


def _robot_xy_to_plot_xy(rx: float, ry: float) -> Tuple[float, float]:
    """Map robot (x, y) to plot data coords; call invert_yaxis on the axes so +robot X is down."""
    return ry, rx


OBJECT_NAME_MAPPING = {
    "I_shape_video": "Letter I",
    "R_shape_video": "Letter R",
    "A_shape_video": "Letter A",
    "E_shape_video": "Letter E",
    "S_shape_video": "Letter S",
    "3_shape_video": "Number 3",
    "baby_toy": "Baby Toy",
    "book": "Book",
    "tape": "Tape",
}


def _display_object_name(codename: str) -> str:
    return OBJECT_NAME_MAPPING.get(codename, codename)


def _goal_angle_deg_to_quat_wxyz(angle_deg: float) -> np.ndarray:
    """Planar rotation about robot +Z; quaternion [w, x, y, z]."""
    half = np.deg2rad(float(angle_deg)) * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


# Matplotlib plot typography (data coordinates / axis labels)
PLOT_FONTSIZE_TITLE = 13
PLOT_FONTSIZE_OBJECT_LABEL = 12
PLOT_FONTSIZE_AXES = 11


class PushAnythingPerceptionGUI(QWidget):
    def __init__(self):
        super().__init__()
        remote_host = "anything@192.168.1.2"
        remote_workdir = "/home/anything/workspace/dairlib"
        controller_cmd = (
            "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller "
            "--is_simulation=false --demo_name=anything --lcm_url=udpm://239.255.76.67:7667?ttl=1"
        )
        visualizer_cmd = (
            "bazel-bin/examples/sampling_c3/franka_visualizer "
            "--is_simulation=false --demo_name=anything"
        )

        self.controller_command_sender = ControllerCommandSenser(
            remote_host=remote_host, remote_workdir=remote_workdir, remote_exec=controller_cmd
        )
        self.visualizer_command_sender = ControllerCommandSenser(
            remote_host=remote_host, remote_workdir=remote_workdir, remote_exec=visualizer_cmd
        )

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.captured_image_filename = "realsense_capture.jpg"
        self.captured_image_path = os.path.join(
            self.base_dir, self.captured_image_filename
        )
        self.bundle_sdf_dir = "/home/yufeiyang/Documents/BundleSDF"
        self.auto_tracking_gui_path = os.path.join(
            self.bundle_sdf_dir, "auto_tracking_gui.py"
        )
        self.mesh_assets_dir = os.path.join(self.bundle_sdf_dir, "assets_textured")
        self.foundation_pose_dir = os.path.join(self.bundle_sdf_dir, "foundationPose")
        self.masks_dir = os.path.join(self.bundle_sdf_dir, "assets")

        # # TODO: will be removed once the testings on MacOS are done
        # self.mesh_assets_dir = "/Users/hienbui/Downloads/assets_textured"
        # self.foundation_pose_dir = "/Users/hienbui/Downloads"
        # self.masks_dir = "/Users/hienbui/Downloads"

        self.setWindowTitle("Push Anything Perception GUI")
        self.resize(2000, 1500)

        # --- Main layout ---
        main_layout = QVBoxLayout()

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.btn1 = QPushButton("Scan")
        self.btn2 = QPushButton("Select Goals")
        self.btn3 = QPushButton("Send Goals and Start Controller")
        _primary_font = QFont()
        _primary_font.setPointSize(15)
        self._primary_font = _primary_font
        for _b in (self.btn1, self.btn2, self.btn3):
            _b.setFont(_primary_font)
            _b.setMinimumHeight(46)
            _b.setMinimumWidth(175)
        # Select Goals: only after a completed Scan. Send Goals: after Select Goals + valid goals.
        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)

        self.label_object = QLabel("Object: ")
        self.label_object.setFont(_primary_font)
        self.label_object.setMinimumHeight(46)
        self.label_object.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.combo_object = QComboBox()
        self.combo_object.setMinimumWidth(200)
        self.combo_object.setFont(_primary_font)
        self.combo_object.setMinimumHeight(46)
        self.btn_default_goals = QPushButton("Default Goals")
        self.btn_default_goals.setFont(_primary_font)
        self.btn_default_goals.setMinimumHeight(46)
        self.btn_default_goals.clicked.connect(self.on_default_goals)
        self.btn_randomize_goals = QPushButton("Randomize Goals")
        self.btn_randomize_goals.setFont(_primary_font)
        self.btn_randomize_goals.setMinimumHeight(46)
        self.btn_randomize_goals.clicked.connect(self.on_randomize_goals)

        self.checkbox_single_goal_mode = QCheckBox("Single Goal Mode")
        self.checkbox_single_goal_mode.setChecked(True)
        self.checkbox_single_goal_mode.setFont(_primary_font)
        self.checkbox_single_goal_mode.setMinimumHeight(46)

        self.label_tracking_status = QLabel("Tracking Status")
        self.label_tracking_status.setFont(_primary_font)
        self.label_tracking_status.setMinimumHeight(46)
        self.label_tracking_status.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.dot_tracking_status = QLabel()
        self.dot_tracking_status.setFixedSize(18, 18)

        self.label_controller_status = QLabel("Controller Status")
        self.label_controller_status.setFont(_primary_font)
        self.label_controller_status.setMinimumHeight(46)
        self.label_controller_status.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.dot_controller_status = QLabel()
        self.dot_controller_status.setFixedSize(18, 18)

        self._set_tracking_status_running(False)
        self._set_controller_status_running(False)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red; font-size: 17pt;")
        self.warning_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.warning_label.hide()
        self.valid_goals_label = QLabel("")
        self.valid_goals_label.setStyleSheet("color: green; font-size: 17pt;")
        self.valid_goals_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.valid_goals_label.hide()

        self.btn1.clicked.connect(self.on_scan_clicked)
        self.btn2.clicked.connect(self.on_select)
        self.btn3.clicked.connect(self.on_send_to_controller)
        self.combo_object.currentIndexChanged.connect(self.on_object_combo_changed)

        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)
        button_layout.addWidget(self.checkbox_single_goal_mode)
        button_layout.addStretch()
        button_layout.addSpacing(12)
        button_layout.addWidget(self.label_tracking_status)
        button_layout.addWidget(self.dot_tracking_status)
        button_layout.addSpacing(16)
        button_layout.addWidget(self.label_controller_status)
        button_layout.addWidget(self.dot_controller_status)

        object_row_layout = QHBoxLayout()
        object_row_layout.addWidget(self.label_object)
        object_row_layout.addWidget(self.combo_object)
        object_row_layout.addWidget(self.btn_default_goals)
        object_row_layout.addWidget(self.btn_randomize_goals)
        object_row_layout.addWidget(self.warning_label)
        object_row_layout.addWidget(self.valid_goals_label)
        object_row_layout.addStretch()

        sliders_layout = QVBoxLayout()
        sliders_layout.setSpacing(16)
        sliders_layout.setContentsMargins(0, 8, 0, 8)
        self.label_slider_x = QLabel("Goal X (m)")
        self.label_slider_y = QLabel("Goal Y (m)")
        self.label_slider_rot = QLabel("Goal Rot (°)")
        self.value_slider_x = QLabel("")
        self.value_slider_y = QLabel("")
        self.value_slider_rot = QLabel("")
        for w in (
            self.label_slider_x,
            self.label_slider_y,
            self.label_slider_rot,
            self.value_slider_x,
            self.value_slider_y,
            self.value_slider_rot,
        ):
            w.setFont(_primary_font)
        for w in (self.label_slider_x, self.label_slider_y, self.label_slider_rot):
            w.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        for w in (self.value_slider_x, self.value_slider_y, self.value_slider_rot):
            w.setMinimumWidth(88)
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row_x = QHBoxLayout()
        row_x.addWidget(self.label_slider_x)
        row_x.addWidget(self.value_slider_x)
        row_x.addStretch(1)
        row_y = QHBoxLayout()
        row_y.addWidget(self.label_slider_y)
        row_y.addWidget(self.value_slider_y)
        row_y.addStretch(1)
        row_rot = QHBoxLayout()
        row_rot.addWidget(self.label_slider_rot)
        row_rot.addWidget(self.value_slider_rot)
        row_rot.addStretch(1)
        for _row in (row_x, row_y, row_rot):
            _row.setContentsMargins(0, 4, 0, 4)

        sliders_layout.addLayout(row_x)
        sliders_layout.addLayout(row_y)
        sliders_layout.addLayout(row_rot)

        # Initially hide adjust controls (labels + sliders until Select Goals)
        self.label_slider_x.hide()
        self.label_slider_y.hide()
        self.label_slider_rot.hide()
        self.value_slider_x.hide()
        self.value_slider_y.hide()
        self.value_slider_rot.hide()
        self.label_object.hide()
        self.combo_object.hide()
        self.btn_default_goals.hide()
        self.btn_randomize_goals.hide()
        self.warning_label.hide()
        self.valid_goals_label.hide()

        # --- Image label ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)  # needed for mouse events
        _instruction_font = QFont()
        _instruction_font.setPointSize(18)
        self.image_label.setFont(_instruction_font)
        self.image_label.setWordWrap(True)

        # Matplotlib canvas (margins applied after each draw; stretch so plot isn't clipped)
        self.canvas = FigureCanvas(Figure(figsize=(12, 5.5), dpi=100))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(260)
        self.canvas.hide()

        # Cursor coordinates label (always visible once hover starts)
        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: green;")

        # Scan image is drawn on self.canvas in on_scan (after on_scan_clicked defers work)
        self.image_label.setText("Please press Scan to identify and track objects.")

        # Add widgets to layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(object_row_layout)
        main_layout.addLayout(sliders_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(self.coord_label)
        self.setLayout(main_layout)

        self.object_states = []
        self.current_object_index = 0
        self.current_detected_objects: List[str] = []
        # Last scan (left panel): RGB image + optional boxes/names for annotation
        self._scan_view_rgb: Optional[np.ndarray] = None
        self._scan_view_names: List[str] = []
        self._scan_view_boxes: List[List[int]] = []
        self._scanning_ui_active = False
        self._scan_chrome_visibility_backup: Dict[Any, bool] = {}
        self._scan_stretch_backup: Optional[Tuple[int, int]] = None
        self._scan_dialog: Optional[ScanningDialog] = None
        self._scan_thread: Optional[ScanThread] = None
        self._object_state_subscribers: List[ObjectStateSubscriber] = []
        self._tracking_poll_timer: Optional[QTimer] = None
        self._tracking_poll_started_at: Optional[float] = None
        self.target_poses_publisher = TargetPosesPublisher()
        self._goal_ax = None
        self._dragging_goal_index: Optional[int] = None
        # Plot-space (px, py) offset from goal center to grab point while dragging (avoids snapping).
        self._drag_plot_offset: Optional[Tuple[float, float]] = None
        self._rotating_goal_index: Optional[int] = None
        self._rotate_prev_pointer_rad: Optional[float] = None

        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_button_release)

    def _set_status_dot_color(self, dot_label: QLabel, running: bool) -> None:
        color = "#2e7d32" if running else "#c62828"
        dot_label.setStyleSheet(
            f"border-radius: 9px; background-color: {color}; border: 1px solid #424242;"
        )

    def _set_tracking_status_running(self, running: bool) -> None:
        self._set_status_dot_color(self.dot_tracking_status, running)

    def _set_controller_status_running(self, running: bool) -> None:
        self._set_status_dot_color(self.dot_controller_status, running)

    def _scan_chrome_widgets(self):
        """Secondary controls hidden during scan (main buttons + Single Goal Mode stay visible)."""
        return (
            self.label_object,
            self.combo_object,
            self.btn_default_goals,
            self.btn_randomize_goals,
            self.warning_label,
            self.valid_goals_label,
            self.label_slider_x,
            self.label_slider_y,
            self.label_slider_rot,
            self.value_slider_x,
            self.value_slider_y,
            self.value_slider_rot,
            self.coord_label,
        )

    def _enter_scanning_only_message_ui(self) -> None:
        """Hide secondary controls; keep main buttons, Single Goal Mode, and the scanning message visible."""
        self._scanning_ui_active = True
        self._scan_chrome_visibility_backup = {
            w: w.isVisible() for w in self._scan_chrome_widgets()
        }
        for w in self._scan_chrome_widgets():
            w.hide()

        ly = self.layout()
        idx_img = ly.indexOf(self.image_label)
        idx_canvas = ly.indexOf(self.canvas)
        self._scan_stretch_backup = None
        if idx_img >= 0 and idx_canvas >= 0:
            try:
                self._scan_stretch_backup = (
                    ly.stretch(idx_img),
                    ly.stretch(idx_canvas),
                )
            except AttributeError:
                self._scan_stretch_backup = (0, 1)
            ly.setStretch(idx_img, 1)
            ly.setStretch(idx_canvas, 0)

    def _restore_scan_chrome_after_scan(self) -> None:
        """Restore hidden widgets and layout stretch after scanning (or on error)."""
        if not self._scanning_ui_active:
            return
        self._scanning_ui_active = False
        for w, vis in self._scan_chrome_visibility_backup.items():
            w.setVisible(vis)
        self._scan_chrome_visibility_backup = {}

        ly = self.layout()
        idx_img = ly.indexOf(self.image_label)
        idx_canvas = ly.indexOf(self.canvas)
        if self._scan_stretch_backup is not None and idx_img >= 0 and idx_canvas >= 0:
            ly.setStretch(idx_img, self._scan_stretch_backup[0])
            ly.setStretch(idx_canvas, self._scan_stretch_backup[1])
        self._scan_stretch_backup = None

    def _apply_default_goals(self) -> None:
        """Center of each workspace partition, 0° angle, cyclic object assignment."""
        n = len(self.object_states)
        if n == 0:
            return

        wx0, wx1 = WORKSPACE_X_LIMIT
        wy0, wy1 = WORKSPACE_Y_LIMIT
        cx_center = 0.5 * (wx0 + wx1)
        region_edges = np.linspace(wy0, wy1, n + 1)
        region_centers = 0.5 * (region_edges[:-1] + region_edges[1:])

        # Cyclic shift by one partition: A->B, B->C, C->A.
        for i, state in enumerate(self.object_states):
            target_region = (i + 1) % n
            state["goal_cx"] = float(cx_center)
            state["goal_cy"] = float(region_centers[target_region])
            state["goal_angle"] = 0.0

    def on_default_goals(self) -> None:
        logger.info("User pressed Default Goals")
        if not self.object_states:
            return
        self._apply_default_goals()
        self._sync_sliders_from_state()
        self.update_plot()

    def _apply_randomized_goals(self) -> None:
        """Split workspace into N regions on Y; assign one object per region and randomize pose."""
        n = len(self.object_states)
        if n == 0:
            return
        rng = np.random.default_rng()
        wx0, wx1 = WORKSPACE_X_LIMIT
        wy0, wy1 = WORKSPACE_Y_LIMIT

        # Shuffle object-to-region mapping so each object is assigned to a random region.
        object_indices = list(range(n))
        rng.shuffle(object_indices)
        region_edges = np.linspace(wy0, wy1, n + 1)
        placed_polygons = []
        max_attempts = 200

        for region_idx, obj_idx in enumerate(object_indices):
            state = self.object_states[obj_idx]
            dx = float(state["dims"][0])
            dy = float(state["dims"][1])
            # Keep full rotated box inside workspace via circumscribed-circle margin.
            margin = 0.5 * math.hypot(dx, dy)
            x_min = wx0 + margin
            x_max = wx1 - margin
            y_min = float(region_edges[region_idx]) + margin
            y_max = float(region_edges[region_idx + 1]) - margin

            # Fallback if object is too large for strict margin in region/workspace.
            if x_min > x_max:
                x_min, x_max = wx0, wx1
            if y_min > y_max:
                y_min = float(region_edges[region_idx])
                y_max = float(region_edges[region_idx + 1])
            if x_min > x_max or y_min > y_max:
                logger.warning(
                    "Random goal region invalid for '{}'; keeping existing goal.",
                    _display_object_name(state["name"]),
                )
                placed_polygons.append(
                    Polygon(
                        self._rect_corners(
                            state["goal_cx"],
                            state["goal_cy"],
                            dx,
                            dy,
                            state["goal_angle"],
                        )
                    )
                )
                continue

            accepted = False
            for _ in range(max_attempts):
                cand_cx = float(rng.uniform(x_min, x_max))
                cand_cy = float(rng.uniform(y_min, y_max))
                cand_angle = float(rng.uniform(-180.0, 180.0))
                corners = self._rect_corners(cand_cx, cand_cy, dx, dy, cand_angle)

                inside_workspace = True
                for x, y in corners:
                    if not (wx0 <= x <= wx1 and wy0 <= y <= wy1):
                        inside_workspace = False
                        break
                if not inside_workspace:
                    continue

                cand_poly = Polygon(corners)
                overlaps_existing = False
                for placed_poly in placed_polygons:
                    if (
                        cand_poly.intersects(placed_poly)
                        or cand_poly.distance(placed_poly) < 1e-6
                    ):
                        overlaps_existing = True
                        break
                if overlaps_existing:
                    continue

                state["goal_cx"] = cand_cx
                state["goal_cy"] = cand_cy
                state["goal_angle"] = cand_angle
                placed_polygons.append(cand_poly)
                accepted = True
                break

            if not accepted:
                logger.warning(
                    "Could not find non-overlapping in-workspace random goal for '{}' after {} attempts; keeping existing goal.",
                    _display_object_name(state["name"]),
                    max_attempts,
                )
                placed_polygons.append(
                    Polygon(
                        self._rect_corners(
                            state["goal_cx"],
                            state["goal_cy"],
                            dx,
                            dy,
                            state["goal_angle"],
                        )
                    )
                )

    def on_randomize_goals(self) -> None:
        logger.info("User pressed Randomize Goals")
        if not self.object_states:
            return
        self._apply_randomized_goals()
        self._sync_sliders_from_state()
        self.update_plot()

    def _capture_realsense_frame(self) -> Optional[np.ndarray]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        started = False
        try:
            pipeline.start(config)
            started = True
            for _ in range(10):
                frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("RealSense did not return a color frame")
            color_bgr = np.asanyarray(color_frame.get_data())
            color_rgb = np.ascontiguousarray(color_bgr[..., ::-1])
            logger.info("Captured RealSense frame, shape {}", color_rgb.shape)
            return color_rgb
        except Exception as e:
            logger.warning("RealSense capture failed: {}", e)
            return None
        finally:
            if started:
                pipeline.stop()

    def _rect_corners(self, cx, cy, w, h, angle):
        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            angle=angle,
            rotation_point="center",
        )
        path = rect.get_path()
        tr = rect.get_transform()
        corners = tr.transform(path.vertices)[:4]
        return corners

    def _rect_polygon_plot_xy(
        self, cx: float, cy: float, w: float, h: float, angle_deg: float
    ) -> np.ndarray:
        """Closed bounding box in plot coords; same geometry as overlap checks (robot frame → plot)."""
        corners = self._rect_corners(cx, cy, w, h, angle_deg)
        return np.array(
            [_robot_xy_to_plot_xy(float(p[0]), float(p[1])) for p in corners],
            dtype=float,
        )

    def check_overlap(self):
        for i in range(len(self.object_states)):
            state1 = self.object_states[i]
            for j in range(i + 1, len(self.object_states)):
                state2 = self.object_states[j]

                corners1 = self._rect_corners(
                    state1["goal_cx"],
                    state1["goal_cy"],
                    state1["dims"][0],
                    state1["dims"][1],
                    state1["goal_angle"],
                )
                corners2 = self._rect_corners(
                    state2["goal_cx"],
                    state2["goal_cy"],
                    state2["dims"][0],
                    state2["dims"][1],
                    state2["goal_angle"],
                )

                poly1 = Polygon(corners1)
                poly2 = Polygon(corners2)

                if poly1.intersects(poly2) or poly1.distance(poly2) < 1e-6:
                    logger.warning(
                        "Overlap detected between {} and {}",
                        _display_object_name(state1["name"]),
                        _display_object_name(state2["name"]),
                    )
                    return True

        return False

    def _goals_outside_workspace(self) -> bool:
        wx0, wx1 = WORKSPACE_X_LIMIT
        wy0, wy1 = WORKSPACE_Y_LIMIT
        for state in self.object_states:
            corners = self._rect_corners(
                state["goal_cx"],
                state["goal_cy"],
                state["dims"][0],
                state["dims"][1],
                state["goal_angle"],
            )
            for x, y in corners:
                if not (wx0 <= x <= wx1 and wy0 <= y <= wy1):
                    logger.warning(
                        "Goal for '{}' outside workspace (corner {:.4f}, {:.4f})",
                        _display_object_name(state["name"]),
                        x,
                        y,
                    )
                    return True
        return False

    def _draw_scan_image_with_detections(
        self,
        ax,
        img_rgb: np.ndarray,
        names: List[str],
        boxes: List[List[int]],
    ) -> None:
        ax.imshow(img_rgb)
        ax.axis("off")
        for i, (name, box) in enumerate(zip(names, boxes)):
            if len(box) != 4:
                continue
            x, y, bw, bh = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            color = OBJECT_EDGE_COLORS[i % len(OBJECT_EDGE_COLORS)]
            rect = patches.Rectangle(
                (x, y),
                bw,
                bh,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x,
                max(2.0, float(y) - 6.0),
                _display_object_name(name),
                fontsize=11,
                color=color,
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

    def _figure_dual_axes(self):
        """Left: camera / detections. Right: goal planning (world frame)."""
        fig = self.canvas.figure
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        return ax_left, ax_right

    def _draw_left_scan_panel(self, ax) -> None:
        ax.set_title("Scan (detections)", fontsize=PLOT_FONTSIZE_TITLE)
        if self._scan_view_rgb is None:
            ax.text(
                0.5,
                0.5,
                "Run Scan",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=PLOT_FONTSIZE_OBJECT_LABEL,
            )
            ax.axis("off")
            return
        if (
            self._scan_view_names
            and self._scan_view_boxes
            and len(self._scan_view_names) == len(self._scan_view_boxes)
        ):
            self._draw_scan_image_with_detections(
                ax,
                self._scan_view_rgb,
                self._scan_view_names,
                self._scan_view_boxes,
            )
        else:
            ax.imshow(self._scan_view_rgb)
            ax.axis("off")

    def _show_scanning_label(self) -> None:
        self._enter_scanning_only_message_ui()
        self.image_label.setText("Scanning the scene and detecting objects")
        self.image_label.show()
        self.canvas.hide()
        self.image_label.repaint()
        QApplication.processEvents()

    def on_scan_clicked(self) -> None:
        """Update UI, then defer heavy work so the scanning label can paint first."""
        self.btn1.setEnabled(False)
        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)
        self._show_scanning_label()
        QTimer.singleShot(0, self._on_scan_after_ui_ready)

    def _on_scan_after_ui_ready(self) -> None:
        self._kill_existing_tracking_processes()
        self._set_tracking_status_running(False)
        self._set_controller_status_running(False)
        if self._tracking_poll_timer is not None:
            self._tracking_poll_timer.stop()
            self._tracking_poll_timer.deleteLater()
            self._tracking_poll_timer = None
        self._object_state_subscribers = []
        self._tracking_poll_started_at = None
        self._scan_dialog = ScanningDialog(self)
        self._scan_dialog.show()
        self._scan_dialog.raise_()
        self._scan_dialog.activateWindow()
        QApplication.processEvents()

        self._scan_thread = ScanThread(self)
        self._scan_thread.finished_ok.connect(self._on_scan_thread_finished_ok)
        self._scan_thread.finished_err.connect(self._on_scan_thread_finished_err)
        self._scan_thread.finished.connect(self._clear_scan_thread_ref)
        self._scan_thread.start()

    def _clear_scan_thread_ref(self) -> None:
        self._scan_thread = None

    def _on_scan_thread_finished_ok(
        self,
        img_rgb: np.ndarray,
        names: List[str],
        boxes: List[List[int]],
    ) -> None:
        self._apply_scan_ui(img_rgb, names, boxes)
        self._start_tracking_and_poll_messages()

    def _on_scan_thread_finished_err(self, message: str) -> None:
        self._set_tracking_status_running(False)
        if self._scan_dialog is not None:
            self._scan_dialog.close()
            self._scan_dialog = None
        self._restore_scan_chrome_after_scan()
        QMessageBox.warning(self, "Scan failed", message)
        self.btn2.setEnabled(True)
        self.btn3.setEnabled(False)
        self.btn1.setEnabled(True)

    def _perform_scan_computation(
        self,
    ) -> Tuple[np.ndarray, List[str], List[List[int]]]:
        logger.info("User pressed Scan button")
        img_rgb: Optional[np.ndarray] = None
        if rs is not None:
            img_rgb = self._capture_realsense_frame()
        if img_rgb is None:
            if rs is not None:
                logger.warning(
                    "RealSense capture failed; using fallback image: {}",
                    self.captured_image_path,
                )
            else:
                logger.info(
                    "pyrealsense2 not available; using image path for scanning: {}",
                    self.captured_image_path,
                )
            pil_fallback = Image.open(self.captured_image_path).convert("RGB")
            img_rgb = np.asarray(pil_fallback)

        pil_img = Image.fromarray(img_rgb)
        scan_boxes: List[List[int]] = []
        names: List[str] = []
        try:
            names, scan_boxes = scan_objects(pil_img, self.masks_dir)
            if names and scan_boxes:
                paired = [
                    (name, box) for name, box in zip(names, scan_boxes) if len(box) >= 4
                ]
                paired.sort(key=lambda p: float(p[1][0] + p[1][2] * 0.5))
                names = [p[0] for p in paired]
                scan_boxes = [list(p[1]) for p in paired]
            with open(os.path.join(self.masks_dir, "object_names.txt"), "w") as f:
                for name in names:
                    f.write(name + "\n")
            logger.info(
                "mask scanning is done, detected objects: {}",
                names,
            )
        except Exception as e:
            logger.exception("scan_objects failed: {}", e)
            names = []
            scan_boxes = []

        return img_rgb, names, scan_boxes

    def _apply_scan_ui(
        self,
        img_rgb: np.ndarray,
        names: List[str],
        boxes: List[List[int]],
    ) -> None:
        self.current_detected_objects = list(names)
        self._scan_view_rgb = img_rgb
        self._scan_view_names = list(names)
        self._scan_view_boxes = [list(b) for b in boxes]

        ax_left, ax_right = self._figure_dual_axes()
        self._draw_left_scan_panel(ax_left)
        ax_right.text(
            0.5,
            0.5,
            "Press Select Goals",
            transform=ax_right.transAxes,
            ha="center",
            va="center",
            fontsize=PLOT_FONTSIZE_OBJECT_LABEL,
            color="gray",
        )
        ax_right.axis("off")
        self._apply_figure_margins()
        self.canvas.draw()
        self.canvas.show()
        self.image_label.hide()

    def _start_tracking_and_poll_messages(self) -> None:
        logger.info("Starting to track objects: {}", self.current_detected_objects)
        subprocess.Popen(
            [sys.executable, self.auto_tracking_gui_path],
            cwd=self.bundle_sdf_dir,
        )

        # Send only the object names to the controller
        # target poses are set to default values and will be ignored by the controller
        self.target_poses_publisher.publish_target(
            self.current_detected_objects,
            [np.array([0.0, 0.0])] * len(self.current_detected_objects),
            [np.array([1.0, 0.0, 0.0, 0.0])] * len(self.current_detected_objects),
            2,
        )
        logger.info("Published target poses to controller")
        if self._scan_dialog is not None:
            self._scan_dialog.set_message("Start tracking objects")
            QApplication.processEvents()

        self._object_state_subscribers = [
            ObjectStateSubscriber(
                channel=f"OBJECT_{name}_STATE_SIMULATION",
            )
            for name in self.current_detected_objects
        ]
        self._tracking_poll_started_at = time.monotonic()
        self._tracking_poll_timer = QTimer(self)
        self._tracking_poll_timer.timeout.connect(self._poll_tracking_messages)
        self._tracking_poll_timer.start(50)

    def _finish_tracking_poll_wait(self) -> None:
        if self._tracking_poll_timer is not None:
            self._tracking_poll_timer.stop()
            self._tracking_poll_timer.deleteLater()
            self._tracking_poll_timer = None
        self._object_state_subscribers = []
        self._tracking_poll_started_at = None

        if self._scan_dialog is not None:
            self._scan_dialog.close()
            self._scan_dialog = None

        self._restore_scan_chrome_after_scan()
        self._set_default_goals_state_after_new_scan()
        self.btn1.setEnabled(True)
        self._set_tracking_status_running(True)
        self.visualizer_command_sender.stop_remote()
        self.visualizer_command_sender.start_remote()

    def _poll_tracking_messages(self) -> None:
        if not self._object_state_subscribers:
            return

        timeout_sec = 180.0
        if self._tracking_poll_started_at is not None:
            if (time.monotonic() - self._tracking_poll_started_at) >= timeout_sec:
                self._set_tracking_status_running(False)
                self._finish_tracking_poll_wait()
                QMessageBox.warning(
                    self,
                    "Failed to track objects! Please try scanning and tracking again.",
                )
                return

        for subscriber in self._object_state_subscribers:
            if not subscriber.has_received:
                subscriber.poll(0)

        if all(
            subscriber.has_received for subscriber in self._object_state_subscribers
        ):
            self._finish_tracking_poll_wait()

    def _set_default_goals_state_after_new_scan(self) -> None:
        """A new scan invalidates prior goal selection (same as UI before Select Goals)."""
        self.object_states = []
        self.current_object_index = -1
        self.combo_object.blockSignals(True)
        self.combo_object.clear()
        self.combo_object.blockSignals(False)
        self.label_object.hide()
        self.combo_object.hide()
        self.btn_default_goals.hide()
        self.btn_randomize_goals.hide()
        self.warning_label.hide()
        self.valid_goals_label.hide()
        self.label_slider_x.hide()
        self.label_slider_y.hide()
        self.label_slider_rot.hide()
        self.value_slider_x.hide()
        self.value_slider_y.hide()
        self.value_slider_rot.hide()
        self.btn2.setEnabled(True)
        self.btn3.setEnabled(False)

    def _kill_existing_tracking_processes(self) -> None:
        targets = ["auto_tracking_gui.py", "fpTracking_share3.py", "camera_memory.py"]

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if not proc.info["cmdline"]:
                    continue
                cmdline = " ".join(proc.info["cmdline"])
                for target in targets:
                    if target in cmdline:
                        print(f"Killing PID {proc.pid}: {cmdline}")
                        proc.kill()
                        break  # stop checking other targets for this process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def on_select(self):
        logger.info("User pressed Select Goals button")
        # Load the mesh files and get the bounding box extents (x, y, z size)
        object_dims = []
        for name in self.current_detected_objects:
            mesh_path = os.path.join(self.mesh_assets_dir, f"{name}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                dimensions = mesh.bounding_box.extents
                logger.info("{}", dimensions)
                object_dims.append((name, dimensions))

            else:
                logger.error("Mesh file not found for {}: {}", name, mesh_path)

        # Populate object states (initialize goals to current poses).
        self.object_states = []
        few_objects = object_dims[:3]

        for name, dims in few_objects:
            initial_pose_path = os.path.join(
                self.foundation_pose_dir, name, "obj_pose_in_world", "00001.txt"
            )
            logger.info("Looking for initial pose at: {}", initial_pose_path)
            cx = 0.5
            cy = 0.5
            angle = 0.0
            if os.path.exists(initial_pose_path):
                try:
                    pose = np.loadtxt(initial_pose_path)
                    logger.info(
                        "Loaded pose for {} from {}:\n{}",
                        name,
                        initial_pose_path,
                        pose,
                    )
                    if pose.shape == (4, 4):
                        cx = float(pose[0, 3])
                        cy = float(pose[1, 3])
                        angle = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
                    else:
                        logger.error(
                            "Invalid pose matrix shape for {}: {}",
                            name,
                            pose.shape,
                        )
                except Exception as e:
                    logger.error("Failed to load pose matrix for {}: {}", name, e)
            else:
                logger.error(
                    "Pose matrix not found for {}: {}", name, initial_pose_path
                )

            self.object_states.append(
                {
                    "name": name,
                    "cx": cx,
                    "cy": cy,
                    "angle": angle,
                    "goal_cx": cx,
                    "goal_cy": cy,
                    "goal_angle": angle,
                    "dims": dims,
                }
            )

        self.current_object_index = 0 if self.object_states else -1

        # Hide image and show plot and sliders
        self.image_label.hide()
        self.canvas.show()
        self.label_slider_x.show()
        self.label_slider_y.show()
        self.label_slider_rot.show()
        self.value_slider_x.show()
        self.value_slider_y.show()
        self.value_slider_rot.show()
        self.label_object.show()
        self.combo_object.show()
        self.btn_default_goals.show()
        self.btn_randomize_goals.show()

        self._populate_object_combo()
        self._sync_sliders_from_state()
        self.update_plot()

    def _publish_target_poses_gui(self, goal_mode: int) -> None:
        obj_names = [s["name"] for s in self.object_states]
        obj_positions = [
            np.array([s["goal_cx"], s["goal_cy"]], dtype=np.float64)
            for s in self.object_states
        ]
        obj_orientations = [
            _goal_angle_deg_to_quat_wxyz(s["goal_angle"]) for s in self.object_states
        ]
        self.target_poses_publisher.publish_target(
            obj_names, obj_positions, obj_orientations, goal_mode
        )
        logger.info(
            "Published TARGET_POSES_GUI for {} object(s), goal_mode={}.",
            len(obj_names),
            goal_mode,
        )
        self._set_controller_status_running(False)
        self.controller_command_sender.stop_remote()
        time.sleep(3)
        self.controller_command_sender.start_remote()
        self._set_controller_status_running(True)

    def on_send_to_controller(self):
        logger.info("Send Goals button pressed")
        if not self.object_states:
            self._set_controller_status_running(False)
            logger.warning("No object states available to push.")
            return

        confirm = QMessageBox(self)
        confirm.setFont(self._primary_font)
        confirm.setWindowTitle("Confirm send to controller")
        confirm.setText(
            "Send detections and goal poses to the controller computer? "
            "Confirm only when Franka is not running."
        )
        confirm.setIcon(QMessageBox.Question)
        btn_send = confirm.addButton("✓", QMessageBox.AcceptRole)
        confirm.addButton("✗", QMessageBox.RejectRole)
        confirm.setDefaultButton(btn_send)
        confirm.exec_()
        if confirm.clickedButton() != btn_send:
            logger.info("Send to controller cancelled by user")
            return

        goal_mode = 2 if self.checkbox_single_goal_mode.isChecked() else 0

        # Print the goal states for debugging
        for state in self.object_states:
            logger.info(
                "Object '{}' goal center: ({:.4f}, {:.4f}), goal angle: {:.2f}°, goal_mode: {}",
                _display_object_name(state["name"]),
                state["goal_cx"],
                state["goal_cy"],
                state["goal_angle"],
                goal_mode,
            )

        self._publish_target_poses_gui(goal_mode)
        self._set_controller_status_running(True)

    def update_plot(self):
        ax_left, ax = self._figure_dual_axes()
        self._goal_ax = ax
        self._draw_left_scan_panel(ax_left)

        if self.object_states:
            ax.tick_params(axis="both", which="major", labelsize=PLOT_FONTSIZE_AXES)
            wx0, wx1 = WORKSPACE_X_LIMIT
            wy0, wy1 = WORKSPACE_Y_LIMIT
            # Workspace in plot: (plot_x, plot_y) = (robot_y, robot_x); y-axis inverted for display.
            workspace_rect = patches.Rectangle(
                (wy0, wx0),
                wy1 - wy0,
                wx1 - wx0,
                linewidth=1.5,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(workspace_rect)
            for i, state in enumerate(self.object_states):
                color = OBJECT_EDGE_COLORS[i % len(OBJECT_EDGE_COLORS)]
                dims = state["dims"]
                ccx, ccy, cang = state["cx"], state["cy"], state["angle"]
                gcx, gcy, gang = (
                    state["goal_cx"],
                    state["goal_cy"],
                    state["goal_angle"],
                )
                pgcx, pgcy = _robot_xy_to_plot_xy(gcx, gcy)
                # Polygon from robot-frame corners: swap (y,x) is orientation-reversing, so a single
                # matplotlib Rectangle angle cannot match both edges; reuse _rect_corners geometry.
                rect_current = patches.Polygon(
                    self._rect_polygon_plot_xy(ccx, ccy, dims[0], dims[1], cang),
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor="none",
                    linestyle="-",
                    closed=True,
                    zorder=3,
                )
                ax.add_patch(rect_current)
                rect_goal = patches.Polygon(
                    self._rect_polygon_plot_xy(gcx, gcy, dims[0], dims[1], gang),
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    linestyle=":",
                    closed=True,
                    zorder=4,
                )
                ax.add_patch(rect_goal)
                ax.text(
                    pgcx,
                    pgcy,
                    _display_object_name(state["name"]),
                    ha="center",
                    va="center",
                    fontsize=PLOT_FONTSIZE_OBJECT_LABEL,
                    color=color,
                )
            ax.set_xlim(PLOT_YLIM[0], PLOT_YLIM[1])
            ax.set_ylim(PLOT_XLIM[0], PLOT_XLIM[1])
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title(
                "Top-down View: Solid = Current Pose, Dotted = Goal Pose",
                fontsize=PLOT_FONTSIZE_TITLE,
            )
            # Green only when goals are inside workspace AND do not overlap each other.
            goals_overlap = self.check_overlap()
            goals_outside_ws = self._goals_outside_workspace()
            if goals_overlap:
                self.warning_label.setText("Warning: Bounding boxes overlap!")
                self.warning_label.show()
                self.valid_goals_label.hide()
            elif goals_outside_ws:
                self.warning_label.setText(
                    "Warning: Goal(s) outside of desired regions!"
                )
                self.warning_label.show()
                self.valid_goals_label.hide()
            else:
                self.warning_label.hide()
                self.valid_goals_label.setText("Selected goals are valid.")
                self.valid_goals_label.show()
            # Send to Controller: only when goals exist, in workspace, and not overlapping.
            self.btn3.setEnabled(not goals_overlap and not goals_outside_ws)
        else:
            ax.tick_params(axis="both", which="major", labelsize=PLOT_FONTSIZE_AXES)
            ax.text(
                0.5,
                0.5,
                "No objects loaded",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PLOT_FONTSIZE_OBJECT_LABEL,
            )
            ax.set_title("Goals (plan)", fontsize=PLOT_FONTSIZE_TITLE)
            self.warning_label.hide()
            self.valid_goals_label.hide()
            self.btn3.setEnabled(False)
        self._apply_figure_margins()
        self.canvas.draw()

    def _pick_goal_in_rotated_bbox(self, plot_x: float, plot_y: float) -> Optional[int]:
        """Pick goal when (plot_x, plot_y) lies inside its rotated goal bbox; tie-break by center distance."""
        if not self.object_states:
            return None
        clicked_point = Point(float(plot_x), float(plot_y))
        best_idx = None
        best_dist = float("inf")
        for i, state in enumerate(self.object_states):
            poly_plot = Polygon(
                self._rect_polygon_plot_xy(
                    float(state["goal_cx"]),
                    float(state["goal_cy"]),
                    float(state["dims"][0]),
                    float(state["dims"][1]),
                    float(state["goal_angle"]),
                )
            )
            if not (
                poly_plot.contains(clicked_point) or poly_plot.touches(clicked_point)
            ):
                continue
            gx, gy = _robot_xy_to_plot_xy(state["goal_cx"], state["goal_cy"])
            d = math.hypot(plot_x - gx, plot_y - gy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _pick_goal_for_drag(self, plot_x: float, plot_y: float) -> Optional[int]:
        return self._pick_goal_in_rotated_bbox(plot_x, plot_y)

    def _pick_goal_for_rotation(self, plot_x: float, plot_y: float) -> Optional[int]:
        """Pick goal when click lies inside its rotated goal bbox in plot coordinates."""
        return self._pick_goal_in_rotated_bbox(plot_x, plot_y)

    def _set_goal_from_plot_xy(self, index: int, plot_x: float, plot_y: float) -> None:
        if index < 0 or index >= len(self.object_states):
            return
        rx = float(np.clip(plot_y, GOAL_X_RANGE[0], GOAL_X_RANGE[1]))
        ry = float(np.clip(plot_x, GOAL_Y_RANGE[0], GOAL_Y_RANGE[1]))
        self.object_states[index]["goal_cx"] = rx
        self.object_states[index]["goal_cy"] = ry

    def _robot_delta_from_plot_to_goal(
        self, gcx: float, gcy: float, plot_x: float, plot_y: float
    ) -> Tuple[float, float]:
        """Plot (x,y) is (robot_y, robot_x); vector from goal center to point in robot frame."""
        rx_m, ry_m = plot_y, plot_x
        return rx_m - gcx, ry_m - gcy

    @staticmethod
    def _wrap_deg_principal(deg: float) -> float:
        """Map angle to (-180, 180] for stable display / slider (same pose mod 360)."""
        x = float(deg) % 360.0
        if x > 180.0:
            x -= 360.0
        return x

    def _on_canvas_button_press(self, event) -> None:
        if event.inaxes is None:
            return
        if self._goal_ax is None or event.inaxes != self._goal_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            goal_idx = self._pick_goal_for_drag(float(event.xdata), float(event.ydata))
            if goal_idx is None:
                return
            self.current_object_index = goal_idx
            self.combo_object.blockSignals(True)
            self.combo_object.setCurrentIndex(goal_idx)
            self.combo_object.blockSignals(False)
            state = self.object_states[goal_idx]
            gpx, gpy = _robot_xy_to_plot_xy(
                float(state["goal_cx"]), float(state["goal_cy"])
            )
            self._drag_plot_offset = (
                float(event.xdata) - gpx,
                float(event.ydata) - gpy,
            )
            self._dragging_goal_index = goal_idx
            self._set_goal_from_plot_xy(
                goal_idx,
                float(event.xdata) - self._drag_plot_offset[0],
                float(event.ydata) - self._drag_plot_offset[1],
            )
        elif event.button == 3:
            goal_idx = self._pick_goal_for_rotation(
                float(event.xdata), float(event.ydata)
            )
            if goal_idx is None:
                return
            self.current_object_index = goal_idx
            self.combo_object.blockSignals(True)
            self.combo_object.setCurrentIndex(goal_idx)
            self.combo_object.blockSignals(False)
            state = self.object_states[goal_idx]
            drx, dry = self._robot_delta_from_plot_to_goal(
                state["goal_cx"],
                state["goal_cy"],
                float(event.xdata),
                float(event.ydata),
            )
            if abs(drx) < 1e-9 and abs(dry) < 1e-9:
                return
            self._rotating_goal_index = goal_idx
            self._rotate_prev_pointer_rad = math.atan2(dry, drx)
        else:
            return

        self._sync_sliders_from_state()
        self.update_plot()

    def _on_canvas_motion(self, event) -> None:
        if self._dragging_goal_index is None and self._rotating_goal_index is None:
            return
        if (
            event.inaxes is None
            or self._goal_ax is None
            or event.inaxes != self._goal_ax
        ):
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._dragging_goal_index is not None:
            ox, oy = self._drag_plot_offset or (0.0, 0.0)
            self._set_goal_from_plot_xy(
                self._dragging_goal_index,
                float(event.xdata) - ox,
                float(event.ydata) - oy,
            )
        elif self._rotating_goal_index is not None:
            ri = self._rotating_goal_index
            state = self.object_states[ri]
            drx, dry = self._robot_delta_from_plot_to_goal(
                state["goal_cx"],
                state["goal_cy"],
                float(event.xdata),
                float(event.ydata),
            )
            if abs(drx) < 1e-9 and abs(dry) < 1e-9:
                return
            pointer_rad = math.atan2(dry, drx)
            prev = self._rotate_prev_pointer_rad
            if prev is not None:
                delta_rad = math.atan2(
                    math.sin(pointer_rad - prev),
                    math.cos(pointer_rad - prev),
                )
                self.object_states[ri]["goal_angle"] += math.degrees(delta_rad)
            self._rotate_prev_pointer_rad = pointer_rad

        self._sync_sliders_from_state()
        self.update_plot()

    def _on_canvas_button_release(self, event) -> None:
        if event.button == 1:
            self._dragging_goal_index = None
            self._drag_plot_offset = None
        elif event.button == 3:
            if self._rotating_goal_index is not None:
                ri = self._rotating_goal_index
                if 0 <= ri < len(self.object_states):
                    self.object_states[ri]["goal_angle"] = self._wrap_deg_principal(
                        float(self.object_states[ri]["goal_angle"])
                    )
            self._rotating_goal_index = None
            self._rotate_prev_pointer_rad = None

    def _apply_figure_margins(self) -> None:
        # Dual-panel layout: room for two titles and axis labels.
        self.canvas.figure.subplots_adjust(
            left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.28
        )

    def _update_slider_value_labels(self) -> None:
        if not self.object_states or self.current_object_index < 0:
            for w in (self.value_slider_x, self.value_slider_y, self.value_slider_rot):
                w.setText("—")
            return
        st = self.object_states[self.current_object_index]
        self.value_slider_x.setText(f"{st['goal_cx']:.3f} m")
        self.value_slider_y.setText(f"{st['goal_cy']:.3f} m")
        self.value_slider_rot.setText(f"{st['goal_angle']:.1f}°")

    def _sync_sliders_from_state(self) -> None:
        # Sliders were removed; keep this helper as the single entry point for value-label refresh.
        self._update_slider_value_labels()

    def _populate_object_combo(self) -> None:
        self.combo_object.blockSignals(True)
        self.combo_object.clear()
        for state in self.object_states:
            self.combo_object.addItem(_display_object_name(state["name"]))
        if self.object_states:
            idx = max(0, min(self.current_object_index, len(self.object_states) - 1))
            self.current_object_index = idx
            self.combo_object.setCurrentIndex(idx)
        self.combo_object.blockSignals(False)

    def on_object_combo_changed(self, index: int) -> None:
        if index < 0 or not self.object_states:
            return
        self.current_object_index = index
        self._sync_sliders_from_state()
        self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PushAnythingPerceptionGUI()
    window.show()
    sys.exit(app.exec_())
