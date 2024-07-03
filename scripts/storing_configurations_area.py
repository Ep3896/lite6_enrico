#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from threading import Lock

from geometry_msgs.msg import Pose, Point
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import time


class StoringConfigurationsArea(Node):

    def __init__(self):
        super().__init__('storing_configurations_area')
        self.align_positions = []

        moveit_config = (
            MoveItConfigsBuilder(robot_name="UF_ROBOT", package_name="lite6_enrico")
            .robot_description_semantic(file_path="config/UF_ROBOT.srdf")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .robot_description(file_path="config/UF_ROBOT.urdf.xacro")
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .joint_limits(file_path="config/joint_limits.yaml")
            .moveit_cpp(file_path=get_package_share_directory("lite6_moveit_demos") + "/config/moveit_cpp.yaml")
            .to_moveit_configs()
        ).to_dict()

        self.lite6 = MoveItPy(node_name="moveit_py", config_dict=moveit_config)
        self.lite6_arm = self.lite6.get_planning_component("lite6_arm")

        self.get_logger().info("StoringConfigurationsArea node has been started")

        self.create_subscription(Float32, '/control/bbox_area', self.bbox_area_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)

        self.current_joint_states = None

    def bbox_area_callback(self, msg):
        if self.current_joint_states:
            joint_positions = list(self.current_joint_states.position)
            self.get_logger().info(f'Current joint positions: {joint_positions}')
            bbox_area = msg.data
            self.align_positions.append((joint_positions, bbox_area))
            self.get_logger().info(f'Added configuration with bbox area: {bbox_area}')

            if len(self.align_positions) >= 5:
                self.get_logger().info(f'Buffer size is {len(self.align_positions)}, selecting max area configuration')
                self.select_max_area_configuration()

    def joint_states_callback(self, msg):
        self.current_joint_states = msg

    def select_max_area_configuration(self):
        self.get_logger().info("Debugging - Entered select_max_area_configuration")
        if self.align_positions:
            max_config = max(self.align_positions, key=lambda x: x[1])
            self.get_logger().info(f'Selected configuration with max bbox area: {max_config[1]}')
            self.move_to_configuration(max_config[0])
            self.align_positions.clear()

    def move_to_configuration(self, joint_positions):
        self.lite6_arm.set_start_state_to_current_state()

        robot_state = RobotState(self.lite6.get_robot_model())
        robot_state.set_joint_group_positions("lite6_arm", joint_positions)

        self.lite6_arm.set_goal_state(robot_state=robot_state)

        plan_result = self.lite6_arm.plan()

        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.get_logger().info(f"Planned trajectory: {robot_trajectory}")
            self.lite6.execute(robot_trajectory, controllers=[])
            self.get_logger().info("Robot moved to the selected configuration")
            time.sleep(1.0)
        else:
            self.get_logger().info("Failed to plan the trajectory")


def main(args=None):
    rclpy.init(args=args)
    storing_configurations_area = StoringConfigurationsArea()

    try:
        rclpy.spin(storing_configurations_area)
    finally:
        storing_configurations_area.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
