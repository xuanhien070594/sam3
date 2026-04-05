import lcm

from lcmt_target_positions.lcmt_target_positions import lcmt_target_positions
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class TargetPublisher:
    def __init__(self, system_name: str):
        self.lc = lcm.LCM()
        # assert system_name in ['jack', 't']
        self.prefix = system_name
        
    def publish_target(self, obj_name = "OBJECT", pose = None, control_mode = 0):
        # Instantiate the message object
        self.pose_msg = lcmt_target_positions()
        # Set the message fields
        self.pose_msg.timestamp = int(time.time() * 1000000)
        self.pose_msg.name = obj_name
        self.pose_msg.continuous_mode = control_mode
        self.pose_msg.num_positions = 7
        self.pose_msg.position_names = [
            f"{self.prefix}_qw", f"{self.prefix}_qx", f"{self.prefix}_qy",
            f"{self.prefix}_qz", f"{self.prefix}_x", f"{self.prefix}_y",
            f"{self.prefix}_z"]
        # Convert the pose to a 7-element array
        self.pose_msg.position = self.homogeneous_matrix_to_pose(pose)
        self.lc.publish(f"OBJECT_{obj_name}_TARGET", self.pose_msg.encode())

    def homogeneous_matrix_to_pose(self, homogeneous_matrix, enforce_planar=True):
        # Convert a 4x4 homogeneous matrix to a 7 element pose with quaternion
        pose = np.zeros(7)
        scipy_quat = R.from_matrix(homogeneous_matrix[:3,:3]).as_quat()
        pose[0] = scipy_quat[-1]
        pose[1:4] = scipy_quat[:3]
        if enforce_planar:
            pose[1] = 0
            pose[2] = 0
            pose[:4] /= np.linalg.norm(pose[:4])
        pose[4:] = homogeneous_matrix[:3,3]
        return pose