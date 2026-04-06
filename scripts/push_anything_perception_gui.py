import sys
import os
from typing import Any, Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import numpy as np
import trimesh

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from shapely.geometry import Polygon
import math
import matplotlib.patches as patches
from loguru import logger
from PIL import Image

from object_detection_and_segmentation import scan_objects

# One color per object (by order in the scene); used for both current and goal boxes.
OBJECT_EDGE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
)

WORKSPACE_X_LIMIT = (0.372, 0.625)
WORKSPACE_Y_LIMIT = (-0.38, 0.38)

# Plot: robot +Y → +plot x; robot +X → downward on screen via invert_yaxis (py = rx > 0).
PLOT_XLIM = (0.0, 0.8)  # robot X → plot y (positive ticks)
PLOT_YLIM = (-0.5, 0.5)  # robot Y → plot x


def _robot_xy_to_plot_xy(rx: float, ry: float) -> Tuple[float, float]:
    """Map robot (x, y) to plot data coords; call invert_yaxis on the axes so +robot X is down."""
    return ry, rx


OBJECT_NAME_MAPPING = {
    "I_shape_video": "Letter I",
    "R_shape_video": "Letter R",
    "D_shape_video": "Letter D",
    "A_shape_video": "Letter A",
}


def _display_object_name(codename: str) -> str:
    return OBJECT_NAME_MAPPING.get(codename, codename)


# Matplotlib plot typography (data coordinates / axis labels)
PLOT_FONTSIZE_TITLE = 13
PLOT_FONTSIZE_OBJECT_LABEL = 12
PLOT_FONTSIZE_AXES = 11
PLOT_FONTSIZE_ROBOT_AXIS = 12
PLOT_FONTSIZE_ROBOT_CAPTION = 11


class InteractiveImageGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.captured_image_filename = "realsense_capture.jpg"
        self.captured_image_path = os.path.join(
            self.base_dir, self.captured_image_filename
        )
        self.bundle_sdf_dir = "/home/yufeiyang/Documents/BundleSDF"
        self.auto_tracking_gui_path = os.path.join(
            self.bundle_sdf_dir, "auto_tracking_gui.py"
        )
        # self.mesh_assets_dir = os.path.join(self.bundle_sdf_dir, "assets_textured")
        # self.foundation_pose_dir = os.path.join(self.bundle_sdf_dir, "foundationPose")
        # self.masks_dir = os.path.join(self.bundle_sdf_dir, "assets")

        # TODO: will be removed once the testings on MacOS are done
        self.mesh_assets_dir = "/Users/hienbui/Downloads/assets_textured"
        self.foundation_pose_dir = "/Users/hienbui/Downloads"
        self.masks_dir = "/Users/hienbui/Downloads"

        self.setWindowTitle("Push Anything Perception GUI")
        self.resize(2000, 1500)

        # --- Main layout ---
        main_layout = QVBoxLayout()

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.btn1 = QPushButton("Scan")
        self.btn2 = QPushButton("Select Goals")
        self.btn3 = QPushButton("Send Goals to Controller")
        _primary_font = QFont()
        _primary_font.setPointSize(15)
        self._primary_font = _primary_font
        for _b in (self.btn1, self.btn2, self.btn3):
            _b.setFont(_primary_font)
            _b.setMinimumHeight(46)
            _b.setMinimumWidth(175)
        # Select Goals: only after a completed Scan. Send Goals to Controller: after Select Goals + valid goals.
        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)

        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_rot = QSlider(Qt.Horizontal)
        self.slider_x.setRange(-500, 1000)
        self.slider_y.setRange(-750, 750)
        self.slider_rot.setRange(-3600, 3600)
        self.slider_x.setSingleStep(10)
        self.slider_y.setSingleStep(10)
        self.slider_rot.setSingleStep(10)
        for _s in (self.slider_x, self.slider_y, self.slider_rot):
            # Enough height so the handle is not clipped in tight rows (esp. macOS)
            _s.setMinimumHeight(36)
            _s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.label_object = QLabel("Object: ")
        self.label_object.setFont(_primary_font)
        self.label_object.setMinimumHeight(46)
        self.label_object.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.combo_object = QComboBox()
        self.combo_object.setMinimumWidth(200)
        self.combo_object.setFont(_primary_font)
        self.combo_object.setMinimumHeight(46)
        self.btn_reset_goals = QPushButton("Reset Goals")
        self.btn_reset_goals.setFont(_primary_font)
        self.btn_reset_goals.setMinimumHeight(46)
        self.btn_reset_goals.clicked.connect(self.on_reset_goals)

        self.checkbox_single_goal_mode = QCheckBox("Single Goal Mode")
        self.checkbox_single_goal_mode.setChecked(True)
        self.checkbox_single_goal_mode.setFont(_primary_font)
        self.checkbox_single_goal_mode.setMinimumHeight(46)

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
        self.slider_x.valueChanged.connect(self.on_slider_x_changed)
        self.slider_y.valueChanged.connect(self.on_slider_y_changed)
        self.slider_rot.valueChanged.connect(self.on_slider_rot_changed)
        self.combo_object.currentIndexChanged.connect(self.on_object_combo_changed)

        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)
        button_layout.addWidget(self.checkbox_single_goal_mode)
        button_layout.addStretch()

        object_row_layout = QHBoxLayout()
        object_row_layout.addWidget(self.label_object)
        object_row_layout.addWidget(self.combo_object)
        object_row_layout.addWidget(self.btn_reset_goals)
        object_row_layout.addWidget(self.warning_label)
        object_row_layout.addWidget(self.valid_goals_label)
        object_row_layout.addStretch()

        sliders_layout = QVBoxLayout()
        sliders_layout.setSpacing(16)
        sliders_layout.setContentsMargins(0, 8, 0, 8)
        self.label_slider_x = QLabel("X (m)")
        self.label_slider_y = QLabel("Y (m)")
        self.label_slider_rot = QLabel("Rot (°)")
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
        row_x.addWidget(self.slider_x, 1)
        row_x.addWidget(self.value_slider_x)
        row_y = QHBoxLayout()
        row_y.addWidget(self.label_slider_y)
        row_y.addWidget(self.slider_y, 1)
        row_y.addWidget(self.value_slider_y)
        row_rot = QHBoxLayout()
        row_rot.addWidget(self.label_slider_rot)
        row_rot.addWidget(self.slider_rot, 1)
        row_rot.addWidget(self.value_slider_rot)
        for _row in (row_x, row_y, row_rot):
            _row.setContentsMargins(0, 4, 0, 4)

        sliders_layout.addLayout(row_x)
        sliders_layout.addLayout(row_y)
        sliders_layout.addLayout(row_rot)

        # Initially hide adjust controls (labels + sliders until Select Goals)
        self.label_slider_x.hide()
        self.label_slider_y.hide()
        self.label_slider_rot.hide()
        self.slider_x.hide()
        self.slider_y.hide()
        self.slider_rot.hide()
        self.value_slider_x.hide()
        self.value_slider_y.hide()
        self.value_slider_rot.hide()
        self.label_object.hide()
        self.combo_object.hide()
        self.btn_reset_goals.hide()
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

    def _scan_chrome_widgets(self):
        """Secondary controls hidden during scan (main buttons + Single Goal Mode stay visible)."""
        return (
            self.label_object,
            self.combo_object,
            self.btn_reset_goals,
            self.warning_label,
            self.valid_goals_label,
            self.label_slider_x,
            self.label_slider_y,
            self.label_slider_rot,
            self.slider_x,
            self.slider_y,
            self.slider_rot,
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
        """Same layout as initial goals in on_select: center X, spread Y, 0° rotation."""
        num_objects = len(self.object_states)
        if num_objects == 0:
            return
        for i, state in enumerate(self.object_states):
            state["goal_cx"] = 0.5
            state["goal_cy"] = 0.2 * (i % num_objects) - (0.2 * (num_objects - 1)) / 2
            state["goal_angle"] = 0.0

    def on_reset_goals(self) -> None:
        logger.info("User pressed Reset goals")
        if not self.object_states:
            return
        self._apply_default_goals()
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
        self.image_label.setText("Scanning the scene and detecting objects...")
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
        scan_ok = False
        try:
            self.on_scan()
            scan_ok = True
        finally:
            self._restore_scan_chrome_after_scan()
            if scan_ok:
                self._reset_goals_state_after_new_scan()
            else:
                self.btn2.setEnabled(True)
                self.btn3.setEnabled(False)
            self.btn1.setEnabled(True)

    def _reset_goals_state_after_new_scan(self) -> None:
        """A new scan invalidates prior goal selection (same as UI before Select Goals)."""
        self.object_states = []
        self.current_object_index = -1
        self.combo_object.blockSignals(True)
        self.combo_object.clear()
        self.combo_object.blockSignals(False)
        self.label_object.hide()
        self.combo_object.hide()
        self.btn_reset_goals.hide()
        self.warning_label.hide()
        self.valid_goals_label.hide()
        self.label_slider_x.hide()
        self.label_slider_y.hide()
        self.label_slider_rot.hide()
        self.slider_x.hide()
        self.slider_y.hide()
        self.slider_rot.hide()
        self.value_slider_x.hide()
        self.value_slider_y.hide()
        self.value_slider_rot.hide()
        self.btn2.setEnabled(True)
        self.btn3.setEnabled(False)

    def on_scan(self) -> None:
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
        try:
            self.current_detected_objects, scan_boxes = scan_objects(
                pil_img, self.masks_dir
            )
            logger.info(
                "mask scanning is done, detected objects: {}",
                self.current_detected_objects,
            )
        except Exception as e:
            logger.exception("scan_objects failed: {}", e)
            self.current_detected_objects = []
            scan_boxes = []

        self._scan_view_rgb = img_rgb
        self._scan_view_names = list(self.current_detected_objects)
        self._scan_view_boxes = [list(b) for b in scan_boxes]

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

        # subprocess.Popen(
        #     [sys.executable, self.auto_tracking_gui_path],
        #     cwd=self.bundle_sdf_dir,
        #     env=os.environ.copy(),
        # )
        # # TODO clear existing running foundationpose instances if any

    def on_select(self):
        logger.info("User pressed Select Goals button")
        # Load the mesh files and get the bounding box extents (x, y, z size)
        object_dims = []
        self.current_detected_objects = ["I_shape_video", "R_shape_video"]
        for name in self.current_detected_objects:
            mesh_path = os.path.join(self.mesh_assets_dir, f"{name}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                dimensions = mesh.bounding_box.extents
                logger.info("{}", dimensions)
                object_dims.append((name, dimensions))

            else:
                logger.error("Mesh file not found for {}: {}", name, mesh_path)

        # Populate object states (current pose from file; default goals from layout, 0° rotation)
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
                    "dims": dims,
                }
            )

        self._apply_default_goals()

        self.current_object_index = 0 if self.object_states else -1

        # Hide image and show plot and sliders
        self.image_label.hide()
        self.canvas.show()
        self.label_slider_x.show()
        self.label_slider_y.show()
        self.label_slider_rot.show()
        self.slider_x.show()
        self.slider_y.show()
        self.slider_rot.show()
        self.value_slider_x.show()
        self.value_slider_y.show()
        self.value_slider_rot.show()
        self.label_object.show()
        self.combo_object.show()
        self.btn_reset_goals.show()

        self._populate_object_combo()
        self._sync_sliders_from_state()
        self.update_plot()

    def on_send_to_controller(self):
        logger.info("Send to Controller button pressed")
        if not self.object_states:
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
        for state in self.object_states:
            state["goal_mode"] = goal_mode
            logger.info(
                "Object '{}' goal center: ({:.4f}, {:.4f}), goal angle: {:.2f}°, goal_mode: {}",
                _display_object_name(state["name"]),
                state["goal_cx"],
                state["goal_cy"],
                state["goal_angle"],
                goal_mode,
            )
            logger.info("{}", state)

    def update_plot(self):
        ax_left, ax = self._figure_dual_axes()
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
                "Goals: solid = current, dotted = goal",
                fontsize=PLOT_FONTSIZE_TITLE,
            )
            self._annotate_robot_frame(ax)

            # Green only when goals are inside workspace AND do not overlap each other.
            goals_overlap = self.check_overlap()
            goals_outside_ws = self._goals_outside_workspace()
            if goals_overlap:
                self.warning_label.setText("Warning: Bounding boxes overlap!")
                self.warning_label.show()
                self.valid_goals_label.hide()
            elif goals_outside_ws:
                self.warning_label.setText("Warning: Goal(s) outside workspace!")
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

    def _annotate_robot_frame(self, ax) -> None:
        """Draw XY triad at origin: robot +Y right, robot +X down (Z omitted)."""
        L = 0.12
        z = 10
        # clip_on=False: arrows/labels may extend past spines.
        kw = dict(
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            zorder=z,
            clip_on=False,
        )
        # Robot +Y → plot +x (to the right)
        ax.add_patch(
            patches.FancyArrowPatch(
                (0.0, 0.0),
                (L, 0.0),
                color="darkgreen",
                **kw,
            )
        )
        # Robot +X → +plot y (displays downward after invert_yaxis)
        ax.add_patch(
            patches.FancyArrowPatch(
                (0.0, 0.0),
                (0.0, L),
                color="darkred",
                **kw,
            )
        )
        ax.text(
            L,
            0.02,
            "y",
            fontsize=PLOT_FONTSIZE_ROBOT_AXIS,
            color="darkgreen",
            zorder=z,
            va="center",
            clip_on=False,
        )
        ax.text(
            0.02,
            L + 0.02,
            "x",
            fontsize=PLOT_FONTSIZE_ROBOT_AXIS,
            color="darkred",
            zorder=z,
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.text(
            0.02,
            0.06,
            "Robot Frame",
            fontsize=PLOT_FONTSIZE_ROBOT_CAPTION,
            color="black",
            zorder=z,
            ha="left",
            va="top",
            clip_on=False,
        )

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
        if not self.object_states or self.current_object_index < 0:
            self._update_slider_value_labels()
            return
        state = self.object_states[self.current_object_index]
        cx = int(round(max(-0.5, min(1.0, state["goal_cx"])) * 1000))
        cy = int(round(max(-0.75, min(0.75, state["goal_cy"])) * 1000))
        angle = max(-360.0, min(360.0, float(state["goal_angle"])))
        rot = int(round(angle * 10))
        for s, v in (
            (self.slider_x, cx),
            (self.slider_y, cy),
            (self.slider_rot, rot),
        ):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._update_slider_value_labels()

    def on_slider_x_changed(self, value: int) -> None:
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["goal_cx"] = value / 1000.0
        self._update_slider_value_labels()
        self.update_plot()

    def on_slider_y_changed(self, value: int) -> None:
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["goal_cy"] = value / 1000.0
        self._update_slider_value_labels()
        self.update_plot()

    def on_slider_rot_changed(self, value: int) -> None:
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["goal_angle"] = value / 10.0
        self._update_slider_value_labels()
        self.update_plot()

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
    window = InteractiveImageGUI()
    window.show()
    sys.exit(app.exec_())
