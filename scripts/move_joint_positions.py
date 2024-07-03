#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
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
        self.create_subscription(Float32, '/control/initial_distance_y', self.initial_distance_y_callback, 10)

        self.stop_execution_pub = self.create_publisher(Bool, '/control/stop_execution', 10)
        
        self.current_joint_states = None
        self.initial_distance_y = None

        # Timer to print joint positions periodically
        self.timer = self.create_timer(0.5, self.print_joint_positions) #it was 1.0
        # Timer to print the align_positions table periodically
        #self.table_timer = self.create_timer(5.0, self.print_align_positions_table)

    def initial_distance_y_callback(self, msg):
        self.initial_distance_y = msg.data
        self.get_logger().info(f'Initial distance y: {self.initial_distance_y}')

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
            print('                            ')
            print('Dimensione align_positions', len(self.align_positions))
            print('                            ')

            if len(self.align_positions) >= 3:
                self.get_logger().info(f'Buffer size is {len(self.align_positions)}, selecting max area configuration')
                self.select_max_area_configuration()
                stop_msg = Bool()
                stop_msg.data = True
                self.stop_execution_pub.publish(stop_msg)

    def joint_states_callback(self, msg):
        self.current_joint_states = msg

    def select_max_area_configuration(self):
        #if self.align_positions:
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
            self.get_logger().info("Moving to camera position")
            self.move_ee_to_camera_pos()
            time.sleep(1.5)
            self.get_logger().info("Moving above card")
            self.move_above_card()
            time.sleep(1.5)
            self.get_logger().info("Moving down to card")
            self.move_down_to_card()
            rclpy.shutdown()

            
            #rclpy.shutdown()


            #time.sleep(5.0)
        else:
            self.get_logger().info("Failed to plan the trajectory")

    def move_ee_to_camera_pos(self):
        self.get_logger().info("Moving EE to camera position")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            ee_pose = robot_state.get_pose("link_tcp")
            camera_pose = robot_state.get_pose("camera_color_optical_frame")

            pose_goal = Pose()
            pose_goal.position = camera_pose.position
            pose_goal.orientation = ee_pose.orientation

            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=1.0)

            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()
        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5)














































    
    def move_above_card(self):
        self.get_logger().info("Moving EE above card")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            ee_pose = robot_state.get_pose("link_tcp")
            distance_y = float(self.initial_distance_y) - ee_pose.position.y

            pose_goal = Pose()
            pose_goal.position.x = ee_pose.position.x
            pose_goal.position.y = ee_pose.position.y + distance_y + 0.08
            pose_goal.position.z = ee_pose.position.z + 0.08
            
            pose_goal.orientation.x = 0.073046
            pose_goal.orientation.y = 0.99627
            pose_goal.orientation.z = -0.040845
            pose_goal.orientation.w = 0.020963


            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=1.0)

            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()
        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5)

    def move_down_to_card(self):
        self.get_logger().info("Moving EE down to card")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            ee_pose = robot_state.get_pose("link_tcp")

            pose_goal = Pose()
            pose_goal.position.x = ee_pose.position.x
            pose_goal.position.y = ee_pose.position.y - 0.02
            pose_goal.position.z = ee_pose.position.z/2
            
            pose_goal.orientation.x = 0.073046
            pose_goal.orientation.y = 0.99627
            pose_goal.orientation.z = -0.040845
            pose_goal.orientation.w = 0.020963

            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=1.0)

            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()
        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5)


    def plan_and_execute(self, robot, planning_component, logger, sleep_time, single_plan_parameters=None, multi_plan_parameters=None):
        logger.info("Planning trajectory")

        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(multi_plan_parameters=multi_plan_parameters)
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(single_plan_parameters=single_plan_parameters)
        else:
            plan_result = planning_component.plan()

        if plan_result:
            logger.info("Executing plan")
            robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=[])

        else:
            logger.error("Planning failed")
            time.sleep(0.5)

        time.sleep(sleep_time)




def main(args=None):
    rclpy.init(args=args)
    storing_configurations_area = Movejoints()
    #storing_configurations_area.print_joint_positions()
    #time.sleep(1.0)
    #storing_configurations_area.move_ee_to_camera_pos()
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
