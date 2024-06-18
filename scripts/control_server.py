#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.logging import get_logger
from lite6_enrico_interfaces.action import GoToPose
from geometry_msgs.msg import Pose, Point
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit.planning import MultiPipelinePlanRequestParameters
import numpy as np
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
import os
from math import pi, sqrt
from simple_pid import PID

# PID gains
KP = 0.5 #0.05 seems correct, 
KI = 0.005 
KD = 0.01

# Maximum movement threshold
MAX_MOVEMENT_THRESHOLD = 1.0  # Meters

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

        # PID controllers for x, y, z
        self.pid_x = PID(KP, KI, KD)
        self.pid_y = PID(KP, KI, KD)
        self.pid_z = PID(KP, KI, KD)
        self.pid_x.sample_time = 0.0333
        self.pid_y.sample_time = 0.0333
        self.pid_z.sample_time = 0.0333

        self.previous_position = Point(x=0.30, y=0.017, z=0.40)

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
        print("                            ")
        print("                            ")
        print("                            ")
        print("                            ")
        print("Lite6 initialized")

    def rotation_callback(self, msg):
        if msg.data:
            self.rotate = True

    def execute_callback(self, goal_handle):
        #self._logger.info('Executing goal...')
        goal = goal_handle.request.pose

        # Set PID setpoints to the goal positions
        self.pid_x.setpoint = goal.position.x
        self.pid_y.setpoint = goal.position.y
        self.pid_z.setpoint = goal.position.z

        print("                        ")
        print("                        ")
        print("++++++++Goal position+++++:", goal.position)

        updated_camera_position = self.go_to_position(goal.position.x, goal.position.y, goal.position.z)

        print("                        ")
        print("Updated camera position:", updated_camera_position)
        print("                        ")

        result = GoToPose.Result()
        result.success = True
        if updated_camera_position:
            result.updated_camera_position = Pose(
                position=updated_camera_position,
                orientation=goal.orientation  # Assuming the orientation remains the same
            )
        goal_handle.succeed()
        return result

    def go_to_position(self, movx, movy, movz):
        plan = False
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        updated_camera_position = None

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            
            self.lite6_arm.set_start_state_to_current_state()
            check_init_pose = robot_state.get_pose("camera_depth_frame")

            # PID control
            current_position = check_init_pose.position
            velocity_x = self.pid_x(current_position.x)
            velocity_y = self.pid_y(current_position.y)
            velocity_z = self.pid_z(current_position.z)

            # Compute the new position
            movx = current_position.x + velocity_x * self.pid_x.sample_time
            movy = current_position.y + velocity_y * self.pid_y.sample_time
            movz = current_position.z + velocity_z * self.pid_z.sample_time

            # Ensure the new position is within the movement threshold
            dist_x = abs(movx - self.previous_position.x)
            dist_y = abs(movy - self.previous_position.y)
            dist_z = abs(movz - self.previous_position.z)
            
            if dist_x > MAX_MOVEMENT_THRESHOLD:
                movx = self.previous_position.x + (MAX_MOVEMENT_THRESHOLD if movx > self.previous_position.x else -MAX_MOVEMENT_THRESHOLD)
            
            if dist_y > MAX_MOVEMENT_THRESHOLD:
                movy = self.previous_position.y + (MAX_MOVEMENT_THRESHOLD if movy > self.previous_position.y else -MAX_MOVEMENT_THRESHOLD)
            
            if dist_z > MAX_MOVEMENT_THRESHOLD:
                movz = self.previous_position.z + (MAX_MOVEMENT_THRESHOLD if movz > self.previous_position.z else -MAX_MOVEMENT_THRESHOLD)

            # Clipping the movement within specified boundaries
            movx = min(max(movx, 0.1), 0.45)
            movy = min(max(movy, -0.3), 0.3)
            movz = min(max(movz, 0.05), 0.40)

            pose_goal = Pose()
            pose_goal.position.x = movx
            pose_goal.position.y = movy
            pose_goal.position.z = movz

            pose_goal.orientation.x = 0.0
            pose_goal.orientation.y = 0.7
            pose_goal.orientation.z = 0.0
            pose_goal.orientation.w = 0.7

            #print("Pose goal:", pose_goal)


            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_depth_frame", timeout=1.0)
            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
            else:
                self._logger.info("IK solution found!")
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                check_updated_pose = robot_state.get_pose("camera_depth_frame")
                print("New_pose:", check_updated_pose)
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()

        if plan:
            plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5)
            updated_camera_position = robot_state.get_pose("camera_depth_frame").position
            print("                        ")
            print("Updated camera position after plan:", updated_camera_position)
            print("                        ")
            self.previous_position = updated_camera_position
            return updated_camera_position

def main(args=None):
    rclpy.init(args=args)
    action_server = GoToPoseActionServer()
    rclpy.spin(action_server)
    action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
