#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from faive_system.src.retargeter import Retargeter # type: ignore
from faive_system.src.common.utils import numpy_to_float32_multiarray # type: ignore
import os
from faive_system.src.viz.visualize_mano import ManoHandVisualizer # type: ignore
import torch
import time

class RetargeterNode(Node):
    def __init__(self, debug=False):
        super().__init__("rokoko_node")

        # start retargeter
        self.declare_parameter("retarget/mjcf_filepath", rclpy.Parameter.Type.STRING)
        self.declare_parameter("retarget/urdf_filepath", rclpy.Parameter.Type.STRING)
        self.declare_parameter("retarget/hand_scheme", rclpy.Parameter.Type.STRING)
        self.declare_parameter("retarget/mano_adjustments", "")
        self.declare_parameter("retarget/retargeter_cfg", "")
        self.declare_parameter("debug", True)
        self.declare_parameter("include_wrist_and_tower", True)

        try:
            mjcf_filepath = self.get_parameter("retarget/mjcf_filepath").value
        except:
            mjcf_filepath = None
        
        try:
            urdf_filepath = self.get_parameter("retarget/urdf_filepath").value
        except:
            urdf_filepath = None
        hand_scheme = self.get_parameter("retarget/hand_scheme").value
        # mano_adjustments's default value is None
        mano_adjustments = self.get_parameter("retarget/mano_adjustments").value
        if mano_adjustments == "":
            mano_adjustments = None

        retargeter_cfg = self.get_parameter("retarget/retargeter_cfg").value
        if retargeter_cfg == "":
            retargeter_cfg = None

        debug = self.get_parameter("debug").value
        device = "cuda" if torch.cuda.is_available() else "cpu"
        include_wrist_and_tower = self.get_parameter("include_wrist_and_tower").value

        
        # subscribe to glove topic
        self.glove_sub = self.create_subscription(
            Float32MultiArray, "/glove/r_full", self.glove_data_cb, 10
        )
        
        self.retargeter = Retargeter(
            device=device,  mjcf_filepath= mjcf_filepath, urdf_filepath=urdf_filepath, 
            hand_scheme=hand_scheme, mano_adjustments=mano_adjustments, retargeter_cfg=retargeter_cfg,
            include_wrist_and_tower=include_wrist_and_tower
        )
        
        self.joints_pub = self.create_publisher(
            Float32MultiArray, "/hand/policy_output", 10
        )
        self.debug = debug
        if self.debug:
            self.rviz_pub = self.create_publisher(MarkerArray, 'retarget/normalized_mano_points', 10)
            self.mano_hand_visualizer = ManoHandVisualizer(self.rviz_pub)
            
        
        self.timer = self.create_timer(0.005, self.timer_publish_cb)
        self.keypoint_positions = None
    
    def glove_data_cb(self, msg):
        # Convert glove data to MANO format
        # Original glove data format (25 joints):
        # [wrist(1),
        #  thumb(metacarpal,proximal,distal,tip)(4),
        #  index(metacarpal,proximal,intermediate,distal,tip)(5),
        #  middle(metacarpal,proximal,intermediate,distal,tip)(5),
        #  ring(metacarpal,proximal,intermediate,distal,tip)(5),
        #  pinky(metacarpal,proximal,intermediate,distal,tip)(5)]
        
        # MANO format (22 joints):
        # [forearm(1), wrist(1),
        #  thumb(4), index(4), middle(4), ring(4), pinky(4)]
        
        glove_data = np.array(msg.data).reshape(-1, 3)
        
        # Initialize MANO format array (22 joints)
        mano_data = np.zeros((22, 3))
        
        # Copy wrist data and duplicate it as forearm (since glove doesn't have forearm)
        mano_data[0] = glove_data[0]  # forearm (duplicated from wrist)
        mano_data[1] = glove_data[0]  # wrist
        
        # Convert thumb (using proximal, distal, tip)
        mano_data[2:6] = glove_data[2:6]  # thumb (proximal, distal, tip)
        
        # Convert fingers (using proximal, intermediate, distal, tip)
        # Index finger
        mano_data[6:10] = glove_data[6:10]  # index (proximal, intermediate, distal, tip)
        
        # Middle finger
        mano_data[10:14] = glove_data[11:15]  # middle (proximal, intermediate, distal, tip)
        
        # Ring finger
        mano_data[14:18] = glove_data[16:20]  # ring (proximal, intermediate, distal, tip)
        
        # Pinky finger
        mano_data[18:22] = glove_data[21:25]  # pinky (proximal, intermediate, distal, tip)
        
        self.keypoint_positions = mano_data
    
        
    def timer_publish_cb(self):
        if self.keypoint_positions is None:
            print("No keypoints received yet")
            time.sleep(1)
            return
        try:
            if self.debug:
                self.mano_hand_visualizer.reset_markers()

            debug_dict = {}
            joint_angles, debug_dict = self.retargeter.retarget(self.keypoint_positions, debug_dict)

            if self.debug:
                self.mano_hand_visualizer.generate_hand_markers(
                    debug_dict["normalized_joint_pos"],
                    stamp=self.get_clock().now().to_msg(),
                )
            self.joints_pub.publish(
                numpy_to_float32_multiarray(np.deg2rad(joint_angles))
            )

            if self.debug:
                self.mano_hand_visualizer.publish_markers()
        except Exception as e:
            print(f"Error in timer_publish_cb: {e}")
            pass


def main(args=None):
    rclpy.init(args=args)
    node = RetargeterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
