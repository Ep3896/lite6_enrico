#!/usr/bin/env python3
import time
import rclpy
from rclpy.logging import get_logger
from moveit.core.robot_state import RobotState
from moveit.planning import (MoveItPy,MultiPipelinePlanRequestParameters)


import numpy as np
from geometry_msgs.msg import PoseStamped, Pose    # set pose goal with PoseStamped messsage

from math import pi

# we need to specify our moveit_py config at the top of each notebook we use. 
# this is since we will start spinning a moveit_py node within this notebook.
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory



good_configurations = []
failed = True


def plan_and_execute( #Helper function to plan and execute a motion.
        robot,
        planning_component,
        logger,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        sleep_time=0.0,
):
    
    #plan to goal
    logger.info("Planning Trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()
    
    #execute the plan
    if plan_result:
        logger.info("Executing Plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
        failed = False
    else:
        logger.error("Failed to plan trajectory")
        failed = True

    time.sleep(sleep_time)


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]



moveit_config = (
        MoveItConfigsBuilder(robot_name="UF_ROBOT", package_name="lite6_enrico")
        .robot_description_semantic(file_path="config/UF_ROBOT.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description(file_path="config/UF_ROBOT.urdf.xacro")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .moveit_cpp(
            file_path=get_package_share_directory("lite6_moveit_demos")
            + "/config/moveit_cpp.yaml"
        )
        .to_moveit_configs()
    ).to_dict()


# initialise rclpy (only for logging purposes)
rclpy.init()
logger = get_logger("moveit_py.pose_goal")

# instantiate moveit_py instance and a planning component for the panda_arm
lite6 = MoveItPy(node_name="moveit_py", config_dict=moveit_config)
lite6_arm = lite6.get_planning_component("lite6_arm")



# KERNEL CRASHA, PERCHÉ? A quanto pare il kernel crasha quando si cerca di settare una posa iniziale e finale con lo stesso stato del robot. Perché?
planning_scene_monitor = lite6.get_planning_scene_monitor()
with planning_scene_monitor.read_write() as scene:
        robot_state = scene.current_state
        original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")

        lite6_arm.set_start_state_to_current_state()

        check_init_pose = robot_state.get_pose("link_tcp")
        print("Initial_pose:", check_init_pose)

        # Set the pose goal
        pose_goal = Pose()
        pose_goal.position.x = 0.4
        pose_goal.position.y = 0.2
        pose_goal.position.z = 0.4
        pose_goal.orientation.w = 1.0
        
        # Set the robot state and check collisions
        result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=2.0)
        if not result:
            logger.error("IK solution was not found!")
        else:
            logger.info("IK solution found!")
        
        robot_state.update() # otherwise the kernel crashes

        check_updated_pose = robot_state.get_pose("link_tcp")

        print("New_pose:", check_updated_pose)

        # set goal state to the initialized robot state
        logger.info("Go to goal")
        lite6_arm.set_goal_state(robot_state=robot_state)

        #robot_state.update()

        print("Initial_pose2:", check_init_pose)


# Questa cella funziona solo dopo che ho mosso manualmente il robot attraverso rviz, dopodiche funziona. Il problema risiede ad un update sbagliato della planning scene o del robot state.
print("current:", check_init_pose)
robot_state.update()
plan_and_execute(lite6, lite6_arm, logger,sleep_time=3.0)

# Restore the original state
"""
robot_state.set_joint_group_positions(
        "lite6_arm",
        original_joint_positions,
)
"""
robot_state.update()  # required to update transforms