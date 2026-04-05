import sys
import os
from typing import List, Optional
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QPoint
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
        self.mesh_assets_dir = os.path.join(self.bundle_sdf_dir, "assets_textured")
        self.foundation_pose_dir = os.path.join(self.bundle_sdf_dir, "foundationPose")
        # self.masks_dir = os.path.join(self.bundle_sdf_dir, "assets")

        # TODO: will be removed once the testings on MacOS are done
        self.masks_dir = "/Users/hienbui/Downloads/"

        self.setWindowTitle("Interactive Image GUI")
        self.resize(600, 400)

        # Store clicked points
        self.points = []

        # --- Main layout ---
        main_layout = QVBoxLayout()

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.btn1 = QPushButton("Scan")
        self.btn2 = QPushButton("Select Goals")
        self.btn3 = QPushButton("Start Pushing")
        self.btn_adjust_x = QPushButton("Adjust X")
        self.btn_adjust_y = QPushButton("Adjust Y")
        self.btn_adjust_rot = QPushButton("Adjust Rotation")
        self.btn_next = QPushButton("Next Object")
        self.btn_x_plus = QPushButton("X+")
        self.btn_x_minus = QPushButton("X-")
        self.btn_y_plus = QPushButton("Y+")
        self.btn_y_minus = QPushButton("Y-")
        self.btn_rot_plus = QPushButton("Rot+")
        self.btn_rot_minus = QPushButton("Rot-")

        self.btn1.clicked.connect(self.on_scan)
        self.btn2.clicked.connect(self.on_select)
        self.btn3.clicked.connect(self.on_start_pushing)
        self.btn_adjust_x.clicked.connect(self.on_adjust_x)
        self.btn_adjust_y.clicked.connect(self.on_adjust_y)
        self.btn_adjust_rot.clicked.connect(self.on_adjust_rot)
        self.btn_next.clicked.connect(self.on_next_object)
        self.btn_x_plus.clicked.connect(self.on_x_plus)
        self.btn_x_minus.clicked.connect(self.on_x_minus)
        self.btn_y_plus.clicked.connect(self.on_y_plus)
        self.btn_y_minus.clicked.connect(self.on_y_minus)
        self.btn_rot_plus.clicked.connect(self.on_rot_plus)
        self.btn_rot_minus.clicked.connect(self.on_rot_minus)

        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)
        button_layout.addWidget(self.btn_adjust_x)
        button_layout.addWidget(self.btn_adjust_y)
        button_layout.addWidget(self.btn_adjust_rot)
        button_layout.addWidget(self.btn_next)
        button_layout.addWidget(self.btn_x_plus)
        button_layout.addWidget(self.btn_x_minus)
        button_layout.addWidget(self.btn_y_plus)
        button_layout.addWidget(self.btn_y_minus)
        button_layout.addWidget(self.btn_rot_plus)
        button_layout.addWidget(self.btn_rot_minus)

        # Initially hide adjust buttons
        self.btn_adjust_x.hide()
        self.btn_adjust_y.hide()
        self.btn_adjust_rot.hide()
        self.btn_next.hide()
        self.btn_x_plus.hide()
        self.btn_x_minus.hide()
        self.btn_y_plus.hide()
        self.btn_y_minus.hide()
        self.btn_rot_plus.hide()
        self.btn_rot_minus.hide()

        # --- Image label ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)  # needed for mouse events

        # Matplotlib canvas
        self.canvas = FigureCanvas(Figure())
        self.canvas.setMouseTracking(True)
        self.current_coord = ""
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.hide()

        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.hide()

        # Cursor coordinates label (always visible once hover starts)
        self.coord_label = QLabel("")
        self.coord_label.setStyleSheet("color: green;")

        # Scan image is drawn on self.canvas in on_scan
        self.image_label.setText("Press Scan to identify and start tracking objects")

        # Add widgets to layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.warning_label)
        main_layout.addWidget(self.coord_label)
        self.setLayout(main_layout)

        self.selected_goals = []
        self.object_states = []
        self.current_object_index = 0
        self.current_detected_objects: List[str] = []

    def on_mouse_move(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self.current_coord = f"({x:.3f}, {y:.3f})"
        self.coord_label.setText(f"Cursor: {self.current_coord}")

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
        self.canvas.figure.tight_layout()
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
        # load object names in object_names.txt
        object_names = []
        if os.path.exists(self.object_names_path):
            with open(self.object_names_path, "r") as f:
                object_names = [line.strip() for line in f.readlines()]
            logger.info("Loaded object names: {}", object_names)

        # load the mesh files
        object_dims = []
        for name in object_names:
            mesh_path = os.path.join(self.mesh_assets_dir, f"{name}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                # bounding box extents (x, y, z size)
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

        # Hide image and show plot and adjust buttons
        self.image_label.hide()
        self.canvas.show()
        self.btn_x_plus.show()
        self.btn_x_minus.show()
        self.btn_y_plus.show()
        self.btn_y_minus.show()
        self.btn_rot_plus.show()
        self.btn_rot_minus.show()
        self.btn_next.show()

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
            ax.set_xlim(-0.5, 1)
            ax.set_ylim(-0.75, 0.75)
            ax.set_aspect("equal")
            ax.set_title("Object Bounding Boxes")

            if self.current_coord:
                ax.text(
                    0.02,
                    0.98,
                    self.current_coord,
                    transform=ax.transAxes,
                    fontsize=9,
                    color="black",
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

            # Check for overlaps
            if self.check_overlap():
                self.warning_label.setText("Warning: Bounding boxes overlap!")
                self.warning_label.show()
            else:
                self.warning_label.hide()
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
        self.canvas.draw()

    def on_adjust_x(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cx"] += 0.05
        self.update_plot()

    def on_adjust_y(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cy"] += 0.05
        self.update_plot()

    def on_adjust_rot(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["angle"] += 10
        self.update_plot()

    def on_next_object(self):
        if self.object_states:
            self.current_object_index = (self.current_object_index + 1) % len(
                self.object_states
            )
        self.update_plot()

    def on_x_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cx"] += 0.05
        self.update_plot()

    def on_x_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cx"] -= 0.05
        self.update_plot()

    def on_y_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cy"] += 0.05
        self.update_plot()

    def on_y_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["cy"] -= 0.05
        self.update_plot()

    def on_rot_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["angle"] += 10
        self.update_plot()

    def on_rot_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]["angle"] -= 10
        self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveImageGUI()
    window.show()
    sys.exit(app.exec_())
