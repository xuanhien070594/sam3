import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import subprocess
import numpy as np
import datetime
import trimesh
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from shapely.geometry import Polygon
import math
import matplotlib.patches as patches

from gui_publisher import TargetPublisher
import psutil

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "test_image_1.jpg")


class InteractiveImageGUI(QWidget):
    def __init__(self):
        super().__init__()

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

        # Load image from same folder as script or capture from RealSense at startup
        self.image_path = image_path
        if rs is not None:
            capture_path = self._capture_realsense_frame()
            if capture_path:
                self.image_path = capture_path
            else:
                print("RealSense capture failed; using fallback image")

        self.pixmap_original = QPixmap(self.image_path)
        if self.pixmap_original.isNull():
            print("❌ Failed to load image")

        # Current displayed pixmap (will draw points on it)
        self.pixmap = self.pixmap_original.copy()
        self.update_image()

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

        # open gui_state.txt and write "started"
        gui_state_path = os.path.join(base_dir, "gui_state.txt")
        with open(gui_state_path, "w") as f:
            f.write("started")

        # self.publisher = TargetPublisher()

    def on_mouse_move(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self.current_coord = f"({x:.3f}, {y:.3f})"
        self.coord_label.setText(f"Cursor: {self.current_coord}")

    def _capture_realsense_frame(self):
        capture_path = os.path.join(base_dir, "realsense_capture.jpg")
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
            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image[..., ::-1]
            plt.imsave(capture_path, color_image)
            print(f"Saved RealSense capture to {capture_path}")
            return capture_path
        except Exception as e:
            print("RealSense capture failed:", e)
            return None
        finally:
            if started:
                pipeline.stop()

    def _rect_corners(self, cx, cy, w, h, angle):
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, angle=angle)
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
        display_dims = (dims[1], dims[0], dims[2]) if len(dims) >= 3 else (dims[1], dims[0])
        return display_cx, display_cy, display_angle, display_dims

    def check_overlap(self):
        for i in range(len(self.object_states)):
            state1 = self.object_states[i]
            for j in range(i + 1, len(self.object_states)):
                state2 = self.object_states[j]

                corners1 = self._rect_corners(
                    state1['cx'], state1['cy'], state1['dims'][0], state1['dims'][1], state1['angle']
                )
                corners2 = self._rect_corners(
                    state2['cx'], state2['cy'], state2['dims'][0], state2['dims'][1], state2['angle']
                )

                poly1 = Polygon(corners1)
                poly2 = Polygon(corners2)

                if poly1.intersects(poly2) or poly1.distance(poly2) < 1e-6:
                    print("⚠️ Overlap detected between {} and {}".format(state1['name'], state2['name']))
                    return True

        return False


    def update_image(self):
        """Scale image and redraw points"""
        if not self.pixmap_original.isNull():
            # Scale original pixmap to label size
            scaled = self.pixmap_original.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Draw points on scaled image
            painter = QPainter(scaled)
            pen = QPen(Qt.red)
            pen.setWidth(6)
            painter.setPen(pen)
            for pt in self.points:
                # Adjust points to scaled coordinates
                x_ratio = scaled.width() / self.pixmap_original.width()
                y_ratio = scaled.height() / self.pixmap_original.height()
                painter.drawPoint(int(pt.x() * x_ratio), int(pt.y() * y_ratio))
            painter.end()

            self.pixmap = scaled
            self.image_label.setPixmap(self.pixmap)

    # --- Button actions ---
    def on_scan(self):
        print("Scan button pressed")

        self.setWindowTitle("Button 1 Clicked")
        image_to_scan = self.image_path
        if rs is not None:
            print("Using the startup RealSense capture for scanning:", image_to_scan)
        else:
            print("pyrealsense2 not available; using existing image path for scanning")

        script_path = "push_anything_create_masks.py"
        subprocess.run(["uv", "run", "python", os.path.join(base_dir, script_path), "--image_path", image_to_scan])
        print("mask scanning is done!")

        auto_tracking_path = "/home/yufeiyang/Documents/BundleSDF/auto_tracking_gui.py"
        auto_tracking_base = "/home/yufeiyang/Documents/BundleSDF"

        subprocess.Popen(
            [sys.executable, auto_tracking_path],  # use same Python env
            cwd=auto_tracking_base,               # make the working dir correct
            env=os.environ.copy()                 # inherit environment
        )
        # TODO clear existing running foundationpose instances if any



    def on_select(self):
        print("Button 2 pressed. User are selecting object goals to push.")
        # TODO ask if user want to use default goal (last targets for recovery) or select new ones
        # load object names in object_names.txt
        object_names_path = os.path.join(base_dir, "object_names.txt")
        object_names = []
        if os.path.exists(object_names_path):
            with open(object_names_path, "r") as f:
                object_names = [line.strip() for line in f.readlines()]
            print("Loaded object names:", object_names)
        
        # load the mesh files
        mesh_base_path = "/home/yufeiyang/Documents/BundleSDF/assets_textured"
        object_dims = []
        for name in object_names:
            mesh_path = os.path.join(mesh_base_path, f"{name}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                # bounding box extents (x, y, z size)
                dimensions = mesh.bounding_box.extents
                print(dimensions)
                object_dims.append((name, dimensions))

            else:
                print(f"❌ Mesh file not found for {name}: {mesh_path}")
        
        # Populate object states
        self.object_states = []
        few_objects = object_dims[:3]
        obj_initial_pose_dir = "/home/yufeiyang/Documents/BundleSDF/foundationPose"
        for name, dims in few_objects:
            initial_pose_path = os.path.join(obj_initial_pose_dir, name, "obj_pose_in_world/", "00001.txt")
            print(f"Looking for initial pose at: {initial_pose_path}")
            cx = 0.5
            cy = 0.5
            angle = 0.0
            if os.path.exists(initial_pose_path):
                try:
                    pose = np.loadtxt(initial_pose_path)
                    print(f"Loaded pose for {name} from {initial_pose_path}:\n{pose}")
                    if pose.shape == (4, 4):
                        cx = float(pose[0, 3])
                        cy = float(pose[1, 3])
                        angle = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
                    else:
                        print(f"❌ Invalid pose matrix shape for {name}: {pose.shape}")
                except Exception as e:
                    print(f"❌ Failed to load pose matrix for {name}: {e}")
            else:
                print(f"❌ Pose matrix not found for {name}: {initial_pose_path}")

            self.object_states.append({
                'name': name,
                'cx': cx,
                'cy': cy,
                'angle': angle,
                'dims': dims
            })
        
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
        print("Start Pushing button pressed")
        if self.object_states:
            for state in self.object_states:
                cx = state['cx']
                cy = state['cy']
                print(f"Object '{state['name']}' bounding box center: ({cx:.4f}, {cy:.4f})")
                print(state)
        else:
            print("No object states available to push.")
        # print("Selected goals to push:", self.selected_goals)
        # write "finished" to gui_state.txt
        gui_state_path = os.path.join(base_dir, "gui_state.txt")
        with open(gui_state_path, "w") as f:
            f.write("finished")
        self.close()
        QApplication.quit()    



        # publisher target positions
        # obj_names = [state['name'] for state in self.object_states]
        # positions = [(state['cx'], state['cy']) for state in self.object_states]
        position_mat = []
        for pos in self.object_states:
            obj_publisher = TargetPublisher(pos['name'])
            print("pose", pos)
            mat = convert_to_matrix(pos)
            position_mat.append(mat)
            obj_publisher.publish_target(obj_name=pos['name'], pose=mat, control_mode=0)


        # List of script filenames you want to stop
        targets = [
            "auto_tracking_gui.py",
            "fpTracking_share3.py",
            "camera_memory.py"
        ]

        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if not proc.info['cmdline']:
                    continue
                cmdline = " ".join(proc.info['cmdline'])
                for target in targets:
                    if target in cmdline:
                        print(f"Killing PID {proc.pid}: {cmdline}")
                        proc.kill()
                        break  # stop checking other targets for this process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 


    def update_plot(self):
        self.canvas.figure.clear()
        if self.object_states:
            ax = self.canvas.figure.add_subplot(111)
            for i, state in enumerate(self.object_states):
                cx, cy = state['cx'], state['cy']
                dims = state['dims']
                angle = state['angle']
                color = 'blue' if i == self.current_object_index else 'red'
                rect = patches.Rectangle((cx - dims[0]/2, cy - dims[1]/2), dims[0], dims[1], 
                                         linewidth=2, edgecolor=color, facecolor='none', angle=angle)
                ax.add_patch(rect)
                # Add text label at center
                ax.text(cx, cy, state['name'], ha='center', va='center', fontsize=8)
            ax.set_xlim(-0.5, 1)
            ax.set_ylim(-0.75, 0.75)
            ax.set_aspect('equal')
            ax.set_title('Object Bounding Boxes')

            if self.current_coord:
                ax.text(
                    0.02, 0.98, self.current_coord,
                    transform=ax.transAxes,
                    fontsize=9, color='black',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            
            # Check for overlaps
            if self.check_overlap():
                self.warning_label.setText("Warning: Bounding boxes overlap!")
                self.warning_label.show()
            else:
                self.warning_label.hide()
        else:
            ax = self.canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No objects loaded', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Data')
            self.warning_label.hide()
        self.canvas.draw()

    def on_adjust_x(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cx'] += 0.05
        self.update_plot()

    def on_adjust_y(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cy'] += 0.05
        self.update_plot()

    def on_adjust_rot(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['angle'] += 10
        self.update_plot()

    def on_next_object(self):
        if self.object_states:
            self.current_object_index = (self.current_object_index + 1) % len(self.object_states)
        self.update_plot()

    def on_x_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cx'] += 0.05
        self.update_plot()

    def on_x_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cx'] -= 0.05
        self.update_plot()

    def on_y_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cy'] += 0.05
        self.update_plot()

    def on_y_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['cy'] -= 0.05
        self.update_plot()

    def on_rot_plus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['angle'] += 10
        self.update_plot()

    def on_rot_minus(self):
        if self.object_states and self.current_object_index >= 0:
            self.object_states[self.current_object_index]['angle'] -= 10
        self.update_plot()

        # TODO: converts the 2d points into 3d world coordinates using the camera calibration
        # world_T_cam = get_transform(base_path='/home/yufeiyang/Documents/ci_mpc_utils/calibrations')
        points_in_world = []
        # for pt in self.selected_goals:
        #     # convert to homogeneous coordinates
        #     pt_homog = np.array([pt[0], pt[1], 1.0, 1.0])  # (x, y, z=1 for plane, w=1)
        #     # transform to world coordinates
        #     pt_world = world_T_cam @ pt_homog
        #     points_in_world.append(pt_world[:3])  # take x, y, z
        # print("Selected goals in world coordinates:", points_in_world)
        return points_in_world
    
def convert_to_matrix(object_state):
    cx, cy = object_state['cx'], object_state['cy']
    angle_rad = math.radians(object_state['angle'])
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Construct homogeneous transformation matrix
    matrix = np.array([
        [cos_a, -sin_a, 0, cx],
        [sin_a,  cos_a, 0, cy],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    return matrix

def get_transform(base_path):
    # check if this is a valid path
    if os.path.exists(base_path):
        print("Path exists.")
    else:
        raise NotADirectoryError(f"Path is not a directory: {base_path}")
    folders = [
        f for f in os.listdir(base_path)
        # if os.path.isdir(os.path.join(base_path, f))
        # and f[:19].count('-') == 5 and '_' in f
    ]
    # Parse folder names as datetime objects
    folders_with_dates = []
    for folder in folders:
        try:
            dt = datetime.datetime.strptime(folder[:19], "%Y-%m-%d_%H-%M-%S")
            folders_with_dates.append((dt, folder))
        except ValueError:
            continue

    # Find the newest one
    if folders_with_dates:
        newest = max(folders_with_dates)[1]
        print("Newest folder:", newest)
    else:
        print("No valid timestamp folders found.")
    calibration_mat = f'{base_path}/{newest}/color_tf_world.npy'
    world_T_cam = np.load(calibration_mat)
    return np.linalg.inv(world_T_cam)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveImageGUI()
    window.show()
    sys.exit(app.exec_())