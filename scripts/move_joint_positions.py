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


class Movejoints(Node):

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
        self.create_subscription(JointState, '/control/joint_states', self.joint_states_callback, 10)

        self.current_joint_states = None

        # Timer to print joint positions periodically
        self.timer = self.create_timer(1.0, self.print_joint_positions)
        # Timer to print the align_positions table periodically
        self.table_timer = self.create_timer(5.0, self.print_align_positions_table)

    def print_joint_positions(self):
        if self.current_joint_states:
            joint_positions = list(self.current_joint_states.position)
            self.get_logger().info(f'Print joint positions: {joint_positions}')
        else:
            self.get_logger().info('Joint positions not available yet.')

    def print_align_positions_table(self):
        if self.align_positions:
            self.get_logger().info("Joint Positions and BBox Areas Table:")
            self.get_logger().info(f"{'Joint Positions':<100} | {'BBox Area':<10}")
            self.get_logger().info("-" * 120)
            for joint_positions, bbox_area in self.align_positions:
                self.get_logger().info(f"{str(joint_positions):<100} | {bbox_area:<10}")
        else:
            self.get_logger().info("No configurations available yet.")

    def bbox_area_callback(self, msg):
        if self.current_joint_states:
            joint_positions = list(self.current_joint_states.position)
            #self.get_logger().info(f'Current joint positions: {joint_positions}')
            bbox_area = msg.data
            self.align_positions.append((joint_positions, bbox_area))
            self.get_logger().info(f'Added configuration with bbox area: {bbox_area}')
            self.get_logger().info(f'AAAAA')

            if len(self.align_positions) == 5:
                self.get_logger().info(f'Buffer size is {len(self.align_positions)}, selecting max area configuration')
                self.select_max_area_configuration()

    def joint_states_callback(self, msg):
        self.current_joint_states = msg

    def select_max_area_configuration(self):
        if self.align_positions:
            max_config = max(self.align_positions, key=lambda x: x[1])
            self.get_logger().info(f'Selected configuration with max bbox area: {max_config[1]}')
            self.move_to_configuration(max_config[0])
            self.align_positions.clear()

    def move_to_configuration(self, joint_positions):
        self.lite6_arm.set_start_state_to_current_state()

        robot_state = RobotState(self.lite6.get_robot_model())
        robot_state.set_joint_group_positions("lite6_arm", joint_positions)

        robot_state.update()

        self.lite6_arm.set_goal_state(robot_state=robot_state)
        robot_state.update()
        plan_result = self.lite6_arm.plan()

        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.get_logger().info(f"Planned trajectory: {robot_trajectory}")
            self.lite6.execute(robot_trajectory, controllers=[])
            self.get_logger().info("Robot moved to the selected configuration")
            rclpy.shutdown()

            time.sleep(5.0)
        else:
            self.get_logger().info("Failed to plan the trajectory")


def main(args=None):
    rclpy.init(args=args)
    storing_configurations_area = Movejoints()
    #storing_configurations_area.move_to_configuration([0.867841899394989, 1.1905378103256226, 0.6513537168502808, -0.8807912468910217, -0.7474675178527832, -1.4510573148727417])
    #storing_configurations_area.move_to_configuration([1.5603810548782349, 2.6416571140289307, 0.5761044025421143, -1.9227548837661743, -0.6542903184890747, -0.23332586884498596])
    #storing_configurations_area.move_to_configuration([0.0, 1.55, 0.0, 0.0, 1.5, 0.0])
    #storing_configurations_area.move_to_configuration([0.43086451292037964, 0.5473434925079346, 0.34440743923187256, -1.1794394254684448, -0.13853764533996582, 1.5742478370666504])
    #storing_configurations_area.move_to_configuration([0.2970631420612335, 0.8729060888290405, 0, -1.7449959516525269, -0.9600739479064941, -1.0469086170196533])
    storing_configurations_area.print_joint_positions()
    #storing_configurations_area.move_to_configuration([0.0001019981864374131, 1.5500705242156982, 1.463597800466232e-05, -0.0001513458846602589, 1.4999010562896729, -8.756755414651707e-05])
    #storing_configurations_area.move_to_configuration([-5.69404692214448e-05, 1.5500444173812866, 6.962205952731892e-05, -0.00014525174628943205, 1.4998596906661987, -4.0878539948607795e-06])
    #storing_configurations_area.move_to_configuration([0.7189179062843323, 0.7136635184288025, 0.8789536952972412, -0.6945827007293701, -0.8471831679344177, -1.6230249404907227])
    """
    planning_scene_monitor = storing_configurations_area.lite6.get_planning_scene_monitor()
    with planning_scene_monitor.read_only() as scene:
        robot_state_to_publish = scene.current_state
        robot_state_to_publish.update()
        joint_positions = robot_state_to_publish.get_joint_group_positions("lite6_arm")
        while True:
            print(joint_positions)
    """
            
    
    
    
    try:
        rclpy.spin(storing_configurations_area)
    finally:
        storing_configurations_area.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
