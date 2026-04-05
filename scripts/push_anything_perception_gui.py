import sys
import os
from typing import List, Optional
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont
import subprocess
import numpy as np
import datetime
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


class InteractiveImageGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.captured_image_filename = "realsense_capture.jpg"
        self.captured_image_path = os.path.join(
            self.base_dir, self.captured_image_filename
        )
        self.object_names_filename = "object_names.txt"
        self.object_names_path = os.path.join(self.base_dir, self.object_names_filename)

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

        self.setWindowTitle("Interactive Image GUI")
        self.resize(1200, 1500)

        # Store clicked points
        self.points = []

        # --- Main layout ---
        main_layout = QVBoxLayout()

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.btn1 = QPushButton("Scan")
        self.btn2 = QPushButton("Select Goals")
        self.btn3 = QPushButton("Start Pushing")
        _primary_font = QFont()
        _primary_font.setPointSize(15)
        for _b in (self.btn1, self.btn2, self.btn3):
            _b.setFont(_primary_font)
            _b.setMinimumHeight(46)
            _b.setMinimumWidth(175)

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
        self.combo_object = QComboBox()
        self.combo_object.setMinimumWidth(200)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red; font-size: 17pt;")
        self.warning_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.warning_label.hide()
        self.valid_goals_label = QLabel("")
        self.valid_goals_label.setStyleSheet("color: green; font-size: 17pt;")
        self.valid_goals_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.valid_goals_label.hide()

        self.btn1.clicked.connect(self.on_scan)
        self.btn2.clicked.connect(self.on_select)
        self.btn3.clicked.connect(self.on_start_pushing)
        self.slider_x.valueChanged.connect(self.on_slider_x_changed)
        self.slider_y.valueChanged.connect(self.on_slider_y_changed)
        self.slider_rot.valueChanged.connect(self.on_slider_rot_changed)
        self.combo_object.currentIndexChanged.connect(self.on_object_combo_changed)

        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)

        object_row_layout = QHBoxLayout()
        object_row_layout.addWidget(self.label_object)
        object_row_layout.addWidget(self.combo_object)
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
        self.warning_label.hide()
        self.valid_goals_label.hide()

        # --- Image label ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)  # needed for mouse events

        # Matplotlib canvas (margins applied after each draw; stretch so plot isn't clipped)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6), dpi=100))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(260)
        self.canvas.hide()

        # Cursor coordinates label (always visible once hover starts)
        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: green;")

        # Scan image is drawn on self.canvas in on_scan
        self.image_label.setText("Press Scan to identify and start tracking objects")

        # Add widgets to layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(object_row_layout)
        main_layout.addLayout(sliders_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(self.coord_label)
        self.setLayout(main_layout)

        self.selected_goals = []
        self.object_states = []
        self.current_object_index = 0
        self.current_detected_objects: List[str] = []

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
        rect = patches.Rectangle((cx - w / 2, cy - h / 2), w, h, angle=angle)
        path = rect.get_path()
        tr = rect.get_transform()
        corners = tr.transform(path.vertices)[:4]
        return corners

    def _world_to_canvas(self, world_x, world_y):
        # World axes: x is vertical, y is horizontal.
        # Convert to canvas axes: x right, y up.
        return float(world_y), float(-world_x)

    def _world_angle_to_canvas(self, world_angle_rad):
        # Rotate the world frame by 90 degrees clockwise for display.
        return math.degrees(world_angle_rad) - 90.0

    def _pose_to_display_state(self, pose, dims):
        display_cx, display_cy = self._world_to_canvas(pose[0, 3], pose[1, 3])
        world_angle = math.atan2(pose[1, 0], pose[0, 0])
        display_angle = self._world_angle_to_canvas(world_angle)
        display_dims = (
            (dims[1], dims[0], dims[2]) if len(dims) >= 3 else (dims[1], dims[0])
        )
        return display_cx, display_cy, display_angle, display_dims

    def check_overlap(self):
        for i in range(len(self.object_states)):
            state1 = self.object_states[i]
            for j in range(i + 1, len(self.object_states)):
                state2 = self.object_states[j]

                corners1 = self._rect_corners(
                    state1["cx"],
                    state1["cy"],
                    state1["dims"][0],
                    state1["dims"][1],
                    state1["angle"],
                )
                corners2 = self._rect_corners(
                    state2["cx"],
                    state2["cy"],
                    state2["dims"][0],
                    state2["dims"][1],
                    state2["angle"],
                )

                poly1 = Polygon(corners1)
                poly2 = Polygon(corners2)

                if poly1.intersects(poly2) or poly1.distance(poly2) < 1e-6:
                    logger.warning(
                        "Overlap detected between {} and {}",
                        state1["name"],
                        state2["name"],
                    )
                    return True

        return False

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

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(img_rgb)
        ax.axis("off")
        self._apply_figure_margins()
        self.canvas.draw()
        self.canvas.show()
        self.image_label.hide()

        try:
            pil_img = Image.fromarray(img_rgb)
            self.current_detected_objects = scan_objects(pil_img, self.masks_dir)
            logger.info(
                "mask scanning is done, detected objects: {}",
                self.current_detected_objects,
            )
        except Exception as e:
            logger.exception("scan_objects failed: {}", e)
            self.current_detected_objects = []

        # subprocess.Popen(
        #     [sys.executable, self.auto_tracking_gui_path],
        #     cwd=self.bundle_sdf_dir,
        #     env=os.environ.copy(),
        # )
        # # TODO clear existing running foundationpose instances if any

    def on_select(self):
        logger.info("User pressed Select Goals button")
        # TODO ask if user want to use default goal (last targets for recovery) or select new ones
        # load the mesh files
        # and get the bounding box extents (x, y, z size)
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

        # Populate object states
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
                {"name": name, "cx": cx, "cy": cy, "angle": angle, "dims": dims}
            )

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

        self._populate_object_combo()
        self._sync_sliders_from_state()
        self.update_plot()

    def on_start_pushing(self):
        # close the window and exit the app
        logger.info("Start Pushing button pressed")
        if self.object_states:
            for state in self.object_states:
                cx = state["cx"]
                cy = state["cy"]
                logger.info(
                    "Object '{}' bounding box center: ({:.4f}, {:.4f})",
                    state["name"],
                    cx,
                    cy,
                )
                logger.info("{}", state)
        else:
            logger.warning("No object states available to push.")
        self.close()
        QApplication.quit()

        # TODO add continuous mode flag

    def update_plot(self):
        self.canvas.figure.clear()
        if self.object_states:
            ax = self.canvas.figure.add_subplot(111)
            for i, state in enumerate(self.object_states):
                cx, cy = state["cx"], state["cy"]
                dims = state["dims"]
                angle = state["angle"]
                color = "blue" if i == self.current_object_index else "red"
                rect = patches.Rectangle(
                    (cx - dims[0] / 2, cy - dims[1] / 2),
                    dims[0],
                    dims[1],
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    angle=angle,
                )
                ax.add_patch(rect)
                # Add text label at center
                ax.text(cx, cy, state["name"], ha="center", va="center", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.75, 0.75)
            ax.set_aspect("equal")
            ax.set_title("Object Bounding Boxes")
            self._annotate_robot_frame(ax)

            # Check for overlaps
            if self.check_overlap():
                self.warning_label.setText("Warning: Bounding boxes overlap!")
                self.warning_label.show()
                self.valid_goals_label.hide()
            else:
                self.warning_label.hide()
                self.valid_goals_label.setText("Selected goals are valid.")
                self.valid_goals_label.show()
        else:
            ax = self.canvas.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No objects loaded",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("No Data")
            self.warning_label.hide()
            self.valid_goals_label.hide()
        self._apply_figure_margins()
        self.canvas.draw()

    def _annotate_robot_frame(self, ax) -> None:
        """Draw XY robot triad at origin (Z omitted); 2D plot is the X–Y plane."""
        L = 0.12
        z = 10
        # clip_on=False: vertical arrow sits on x=0; arrowhead can extend past the left spine and was clipped.
        kw = dict(
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            zorder=z,
            clip_on=False,
        )
        ax.add_patch(
            patches.FancyArrowPatch(
                (0.0, 0.0),
                (L, 0.0),
                color="darkred",
                **kw,
            )
        )
        ax.add_patch(
            patches.FancyArrowPatch(
                (0.0, 0.0),
                (0.0, L),
                color="darkgreen",
                **kw,
            )
        )
        ax.text(
            L + 0.02,
            0.0,
            "x",
            fontsize=10,
            color="darkred",
            zorder=z,
            va="center",
            clip_on=False,
        )
        ax.text(
            0.02,
            L + 0.02,
            "y",
            fontsize=10,
            color="darkgreen",
            zorder=z,
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.text(
            0.02,
            -0.06,
            "Robot Frame",
            fontsize=9,
            color="black",
            zorder=z,
            ha="left",
            va="top",
            clip_on=False,
        )

    def _apply_figure_margins(self) -> None:
        # Embedded Qt canvas needs explicit room for title and axis tick labels.
        self.canvas.figure.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.14)

    def _update_slider_value_labels(self) -> None:
        if not self.object_states or self.current_object_index < 0:
            for w in (self.value_slider_x, self.value_slider_y, self.value_slider_rot):
                w.setText("—")
            return
        st = self.object_states[self.current_object_index]
        self.value_slider_x.setText(f"{st['cx']:.3f} m")
        self.value_slider_y.setText(f"{st['cy']:.3f} m")
        self.value_slider_rot.setText(f"{st['angle']:.1f}°")

    def _sync_sliders_from_state(self) -> None:
        if not self.object_states or self.current_object_index < 0:
            self._update_slider_value_labels()
            return
        state = self.object_states[self.current_object_index]
        cx = int(round(max(-0.5, min(1.0, state["cx"])) * 1000))
        cy = int(round(max(-0.75, min(0.75, state["cy"])) * 1000))
        angle = max(-360.0, min(360.0, float(state["angle"])))
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
            self.object_states[self.current_object_index]["cx"] = value / 1000.0
        self._update_slider_value_labels()
        self.update_plot()

    def on_slider_y_changed(self, value: int) -> None:
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cy"] = value / 1000.0
        self._update_slider_value_labels()
        self.update_plot()

    def on_slider_rot_changed(self, value: int) -> None:
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["angle"] = value / 10.0
        self._update_slider_value_labels()
        self.update_plot()

    def _populate_object_combo(self) -> None:
        self.combo_object.blockSignals(True)
        self.combo_object.clear()
        for state in self.object_states:
            self.combo_object.addItem(state["name"])
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
