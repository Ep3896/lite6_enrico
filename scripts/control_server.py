#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.logging import get_logger
from lite6_enrico_interfaces.action import GoToPose
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit.planning import MultiPipelinePlanRequestParameters
import numpy as np
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os
from geometry_msgs.msg import Point, Quaternion
from math import pi



# Plan and execute function
def plan_and_execute(robot, planning_component, logger, single_plan_parameters=None, multi_plan_parameters=None, sleep_time=0.0):
    # plan to goal
    logger.info("Planning trajectory")
    stopper = lite6.get_trajactory_execution_manager()
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(multi_plan_parameters=multi_plan_parameters)
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(single_plan_parameters=single_plan_parameters)
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
        stopper.stop_execution()
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)

# Convert Euler angles to quaternions
def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

# Go to pose function
def got_to_position(movx, movy, movz):
        plan = False
        planning_scene_monitor = lite6.get_planning_scene_monitor()
        #with planning_scene_monitor.read_write() as scene:
        with planning_scene_monitor.read_write() as scene:

            # instantiate a RobotState instance using the current robot model
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")

            lite6_arm.set_start_state_to_current_state()
            check_init_pose = robot_state.get_pose("camera_depth_frame")

            quaternion = get_quaternion_from_euler (0, pi, 0)

            # With this configuration of the orientation the gripper is always parallel to the x axis, so it aims  orientation [1.0,0.0,0.0,0.0] -> for link_tcp

            pose_goal = Pose() 

            """
            if movz > -0.25 and movz < 0.14:
                pose_goal.position.z = movz
            elif movz > 0.14:
                pose_goal.position.z = 0.14
            else:
                pose_goal.position.z = -0.25
            
            if movy > -0.60 and movy < 0.30:
                pose_goal.position.y = movy
            elif movy > 0.30:
                pose_goal.position.y = 0.30
            else:
                pose_goal.position.y = -0.60


            if movx > 0.15 and movx < 0.42:
                pose_goal.position.x = movx
            elif movx > 0.42:
                pose_goal.position.x = 0.42
            else:
                pose_goal.position.x = 0.15
            """
            """
            if pose_goal.position.x <= 0.32:
                pose_goal.position.x = movx + 0.1
            elif pose_goal.position.x > 0.32:
                pose_goal.position.x = 0.42
            else:
                print("Error in x position, risk of collision with itself")

            if pose_goal.position.y >= -0.45 or pose_goal.position.y <= 0.30:
                pose_goal.position.y = movy + 0.25
            
            if pose_goal.position.y >= -0.65 or pose_goal.position.y <= -0.45:
                pose_goal.position.y = movy + 0.25
            elif pose_goal.position.y < -0.65:
                pose_goal.position.y = -0.45

            if movz > 0.15:
                pose_goal.position.z = movz
            else:
                pose_goal.position.z = 0.15
            """
            pose_goal.position.x = movx
            pose_goal.position.y = movz
            if check_init_pose.position.z >= 0.16:
                pose_goal.position.z = check_init_pose.position.z - 0.01
            else:
                pose_goal.position.z = 0.15
            
            """
            """
            #this configuration is for the camera_depth_frame
            pose_goal.orientation.x = 0.0
            pose_goal.orientation.y = 0.68
            pose_goal.orientation.z = 0.0
            pose_goal.orientation.w = 0.72
            """
            pose_goal.orientation.x = 1.0
            pose_goal.orientation.y = 0.0
            pose_goal.orientation.z = 0.0
            pose_goal.orientation.w = 0.0
            """
            # Set the robot state and check collisions
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_depth_frame", timeout=1.0)
            #result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_depth_frame", timeout=5.0)
            if not result:
                logger.error("IK solution was not found!")
                logger.error("Failed goal is: {}".format(pose_goal))
                return
            else:
                logger.info("IK solution found!")
                logger.info("\033[92mGoal is: {}\033[0m".format(pose_goal))

                plan = True
            
                lite6_arm.set_goal_state(robot_state=robot_state)   
            
                robot_state.update() # otherwise the kernel crashes

                check_updated_pose = robot_state.get_pose("camera_depth_frame")
                #check_updated_pose = robot_state.get_pose("camera_depth_frame")

                print("New_pose:", check_updated_pose)

                logger.info("Go to goal")

                robot_state.set_joint_group_positions(
                "lite6_arm",
                original_joint_positions,
                    )
                robot_state.update()

        if plan == True:
            plan_and_execute(lite6, lite6_arm, logger,sleep_time=0.5)
            """
            time.sleep(3.0)

            lite6_arm.set_start_state_to_current_state()
            lite6_arm.set_goal_state(configuration_name="Ready")
            # plan to goal
            plan_result = lite6_arm.plan()

            # execute the plan
            if plan_result:
                logger.info("Executing plan")
                robot_trajectory = plan_result.trajectory
                lite6.execute(robot_trajectory, controllers=[])
            else:
                logger.error("Planning failed, no getting back to initial position!")
            """


# Action server class
class GoToPoseActionServer(Node):

    def __init__(self):
        super().__init__('control_server')
        self._action_server = ActionServer(
            self,
            GoToPose,
            'go_to_pose',
            execute_callback=self.execute_callback)
        self._logger = get_logger("go_to_pose_action_server")

    def execute_callback(self, goal_handle):
        self._logger.info('Executing goal...')
        goal = goal_handle.request.pose
        movx, movy, movz = goal.position.x, goal.position.y, goal.position.z

        # Call the go_to_pose function
        got_to_position(movx, movy, movz)

        # Notify the action client that the goal was completed
        goal_handle.succeed()

        result = GoToPose.Result()

        result.success = True
        """
        result.pose = Pose(
            position=Point(x=movx, y=movy, z=movz),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )
        """
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
