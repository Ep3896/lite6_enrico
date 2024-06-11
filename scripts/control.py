#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.logging import get_logger
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit.planning import MultiPipelinePlanRequestParameters
import numpy as np
from geometry_msgs.msg import Pose, Point
from yolov8_msgs.msg import DetectionArray
from lite6_enrico_interfaces.action import GoToPose  # Ensure this matches your action definition
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import random
from math import pi
import os
from visualization_msgs.msg import MarkerArray

# Plan and execute function
def plan_and_execute(robot, planning_component, logger, single_plan_parameters=None, multi_plan_parameters=None, sleep_time=0.0):
    # plan to goal
    logger.info("Planning trajectory")
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


# Action client class
class PoseActionClient(Node):

    def __init__(self):
        super().__init__('control')
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')
        self.subscription = self.create_subscription(MarkerArray, '/yolo/dgb_bb_markers', self.detection_callback, 10)
        self.recieved = False
        self.task = True

    def detection_callback(self, msg):
        if msg.markers and self.task == True:
            print("Detected markers")
            marker = msg.markers[0]  # Process the first marker, or modify as needed
            goal_pose = Pose()
            goal_pose.position.x = marker.pose.position.x
            goal_pose.position.y = marker.pose.position.y
            goal_pose.position.z = marker.pose.position.z
            print("Goal pose:", goal_pose)

            if self.recieved == False:
                self.send_goal(goal_pose)
                self.task = False

    def send_goal(self, pose):
        goal_msg = GoToPose.Goal()
        goal_msg.pose = pose

        print("Waiting for server")
        self._action_client.wait_for_server()
        print("Sending goal: ", goal_msg)
        self.recieved = True
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
        self.recieved = False
        if result:
            self.get_logger().info('Goal succeeded!')
            self.task = True
        else:
            self.get_logger().info('Goal failed!')


        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        """
        # Extract pose from result and call go_to_pose
        goal_pose = result.pose
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Assuming some default or calculated values for roll, pitch, and yaw
        movx = goal_pose.position.x
        movy = goal_pose.position.y
        movz = goal_pose.position.z
        go_to_pose(roll, pitch, yaw, movx, movy, movz)
        """

    

def main(args=None):
    rclpy.init(args=args)


    # Clear the terminal output
    os.system('cls' if os.name == 'nt' else 'clear')

    print("ROS2 control node initialized")
    #create a task variable
    
    action_client = PoseActionClient()
    rclpy.spin(action_client)

    action_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
