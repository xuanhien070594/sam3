import lcm

from lcmtypes.lcmt_target_poses_gui.lcmt_target_poses_gui import lcmt_target_poses_gui
import time
from typing import List
import numpy as np


class TargetPosesPublisher:
    def __init__(self):
        self.lc = lcm.LCM()

    def publish_target(
        self,
        obj_names: List[str],
        obj_positions: List[np.ndarray],
        obj_orientations: List[np.ndarray],
        goal_mode: int,
    ):
        # Instantiate the message object
        self.pose_msg = lcmt_target_poses_gui()
        # Set the message fields
        self.pose_msg.timestamp = int(time.time() * 1000000)
        self.pose_msg.num_objects = len(obj_names)
        self.pose_msg.goal_mode = goal_mode
        self.pose_msg.object_names = obj_names
        self.pose_msg.target_positions = obj_positions
        self.pose_msg.target_orientations = obj_orientations
        self.lc.publish("TARGET_POSES_GUI", self.pose_msg.encode())
