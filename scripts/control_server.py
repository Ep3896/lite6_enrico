#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.logging import get_logger
from lite6_enrico_interfaces.action import GoToPose
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit.planning import MultiPipelinePlanRequestParameters
import numpy as np
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
import os
from math import pi, sqrt

# Plan and execute function
def plan_and_execute(robot, planning_component, logger, single_plan_parameters=None, multi_plan_parameters=None, sleep_time=0.0):
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

    time.sleep(sleep_time)

def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def go_to_position(movx, movy, movz, previous_position, movement_threshold, rotate=False):
    plan = False
    planning_scene_monitor = lite6.get_planning_scene_monitor()
    with planning_scene_monitor.read_write() as scene:
        robot_state = scene.current_state
        original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
        lite6_arm.set_start_state_to_current_state()
        check_init_pose = robot_state.get_pose("camera_depth_frame")

        # Calculate the distance between the previous position and the new position
        distance = sqrt((movx - previous_position.x)**2 + (movy - previous_position.y)**2 + (movz - previous_position.z)**2)

        # Check if the distance exceeds the movement threshold
        if distance > movement_threshold:
            scale = movement_threshold / distance
            movx = previous_position.x + scale * (movx - previous_position.x)
            movy = previous_position.y + scale * (movy - previous_position.y)
            movz = previous_position.z + scale * (movz - previous_position.z)

        pose_goal = Pose()
        pose_goal.position.x = max(movx, 0.1)
        pose_goal.position.y = max(min(movy, 0.3), -0.45)
        pose_goal.position.z = max(min(movz, 0.45), 0.16)

        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.orientation.z = 0.0
        pose_goal.orientation.w = 0.0

        result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=1.0)
        if not result:
            logger.error("IK solution was not found!")
            logger.error("Failed goal is: {}".format(pose_goal))
            return
        else:
            logger.info("IK solution found!")
            logger.info("\033[92mGoal is: {}\033[0m".format(pose_goal))

            plan = True
            lite6_arm.set_goal_state(robot_state=robot_state)
            robot_state.update()
            check_updated_pose = robot_state.get_pose("link_tcp")
            print("New_pose:", check_updated_pose)
            logger.info("Go to goal")

            robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
            robot_state.update()

    if plan:
        plan_and_execute(lite6, lite6_arm, logger, sleep_time=0.5)
        return pose_goal.position  # Return the new position

class GoToPoseActionServer(Node):

    def __init__(self):
        super().__init__('control_server')
        self._action_server = ActionServer(
            self,
            GoToPose,
            'go_to_pose',
            execute_callback=self.execute_callback)
        self._logger = get_logger("go_to_pose_action_server")
        self.create_subscription(Bool, 'rotation_flag', self.rotation_callback, 10)
        self.rotate = False
        self.previous_position = Point(x=0.3106, y=0.017492, z=0.44321)  # Initial position
        self.movement_threshold = 0.05  # Maximum allowed movement in meters

    def rotation_callback(self, msg):
        if msg.data:
            self.rotate = True

    def execute_callback(self, goal_handle):
        self._logger.info('Executing goal...')
        goal = goal_handle.request.pose
        movx, movy, movz = goal.position.x, goal.position.y, goal.position.z

        new_position = go_to_position(movx, movy, movz, self.previous_position, self.movement_threshold, self.rotate)
        if new_position:
            self.previous_position = new_position  # Update the previous position

        goal_handle.succeed()
        self.rotate = False  # Reset the rotation flag after execution

        result = GoToPose.Result()
        result.success = True
        return result

def main(args=None):
    rclpy.init(args=args)
    global logger, lite6, lite6_arm
    logger = get_logger("moveit_py.pose_goal")

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

    lite6 = MoveItPy(node_name="moveit_py", config_dict=moveit_config)
    lite6_arm = lite6.get_planning_component("lite6_arm")

    action_server = GoToPoseActionServer()
    rclpy.spin(action_server)

    action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
