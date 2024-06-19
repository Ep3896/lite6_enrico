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
from moveit.planning import PlanRequestParameters
import numpy as np
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
import os
from math import pi, sqrt
from simple_pid import PID
import math
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import Constraints, JointConstraint




# PID gains
KP = 0.1 #0.05 seems correct, 
KI = 0.01 # 0.005 seems correct 
KD = 0.01

# Maximum movement threshold
MAX_MOVEMENT_THRESHOLD = 1.0  # Meters

# Plan and execute function
def plan_and_execute(robot, planning_component, logger, sleep_time, single_plan_parameters=None, multi_plan_parameters=None):
    logger.info("Planning trajectory")

    constraints = Constraints()
    constraints.name = "joints_constraints"

    joint_5_contraint = JointConstraint()
    joint_5_contraint.joint_name = "joint5"
    joint_5_contraint.position = 1.363802  # 90 degrees in radians
    joint_5_contraint.tolerance_above = 2.5
    joint_5_contraint.tolerance_below = 2.5
    joint_5_contraint.weight = 1.0

    constraints.joint_constraints.append(joint_5_contraint)

    joint_4_contraint = JointConstraint()
    joint_4_contraint.joint_name = "joint4"
    joint_4_contraint.position = 0.000010 #0.0  
    joint_4_contraint.tolerance_above = 2.5
    joint_4_contraint.tolerance_below = 2.5
    joint_4_contraint.weight = 1.0

    constraints.joint_constraints.append(joint_4_contraint)

    joint_6_contraint = JointConstraint()
    joint_6_contraint.joint_name = "joint6"
    joint_6_contraint.position = 0.0
    joint_6_contraint.tolerance_above = 2.5
    joint_6_contraint.tolerance_below = 2.5
    joint_6_contraint.weight = 1.0

    constraints.joint_constraints.append(joint_6_contraint)
    
    planning_component.set_path_constraints(constraints)


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

        updated_camera_position = self.go_to_position(goal.position.x, goal.position.y + 0.25, goal.position.z) # 0.25 was added to the y position as offset

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
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state # Get the current state of the robot, including the joint values. 
                                              #This is the state of the robot at the time of the last update of the planning scene
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm") # Get the joint positions of the robot
            
            self.lite6_arm.set_start_state_to_current_state() # Set the START STATE of the robot to the current state of the robot
            check_init_pose = robot_state.get_pose("camera_depth_frame") # Get the pose of the camera_depth_frame

            # PID control
            current_position = check_init_pose.position # Get the position of the camera_depth_frame
            velocity_x = self.pid_x(current_position.x) # Get the velocity of the camera_depth_frame in the x direction
            velocity_y = self.pid_y(current_position.y) # Get the velocity of the camera_depth_frame in the y direction
            velocity_z = self.pid_z(current_position.z) # Get the velocity of the camera_depth_frame in the z direction

            # Compute the new position
            movx = current_position.x + 5*velocity_x * self.pid_x.sample_time  # increase the velocity by 3 times, added by myself
            movy = current_position.y + 5*velocity_y * self.pid_y.sample_time
            movz = current_position.z + 5*velocity_z * self.pid_z.sample_time

            # Ensure the new position is within the movement threshold
            dist_x = abs(movx - self.previous_position.x)
            dist_y = abs(movy - self.previous_position.y)
            dist_z = abs(movz - self.previous_position.z)
            
            if dist_x > 2*MAX_MOVEMENT_THRESHOLD:
                movx = self.previous_position.x + (MAX_MOVEMENT_THRESHOLD if movx > self.previous_position.x else -MAX_MOVEMENT_THRESHOLD)
            
            if dist_y > 2*MAX_MOVEMENT_THRESHOLD:
                movy = self.previous_position.y + (MAX_MOVEMENT_THRESHOLD if movy > self.previous_position.y else -MAX_MOVEMENT_THRESHOLD)
            
            if dist_z > MAX_MOVEMENT_THRESHOLD/2:
                movz = self.previous_position.z + (MAX_MOVEMENT_THRESHOLD if movz > self.previous_position.z else -MAX_MOVEMENT_THRESHOLD)

            # Clipping the movement within specified boundaries
            movx = min(max(movx, 0.2), 0.45) # 0.1 was fine for the x axis
            movy = min(max(movy, -0.3), 0.3) 
            movz = min(max(movz, 0.13), 0.40) # 0.13 was fine for the z axis, it is the minimum height of the camera_depth_frame beacuse 0.15, in the other node, is the average heigh that has to be reached to end the task.

            pose_goal = Pose()
            pose_goal.position.x = movx
            pose_goal.position.y = movy
            pose_goal.position.z = movz

            pose_goal.orientation.x = 0.0
            pose_goal.orientation.y = 0.7
            pose_goal.orientation.z = 0.0
            pose_goal.orientation.w = 0.7

            #print("Pose goal:", pose_goal) 
            #robot_state = RobotState(self.lite6.get_robot_model())

            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_depth_frame", timeout=1.0) # Set the robot state from the inverse kinematics solution

            # I added this to check if the robot is in collision, the problem is a segmentation fault, could it be that requires more time to check for collision? maybe time sleep is insufficient

            robot_collision_status = scene.is_state_colliding(
            robot_state=robot_state, joint_model_group_name="panda_arm", verbose=True
                                                            )
            self._logger.info(f"\nRobot is in collision: {robot_collision_status}\n")

            

            if robot_collision_status == True:
                time.sleep(0.5)
                return
                                                                                                         # The robot state is set to the pose goal, this state is the goal state
            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
            else:
                #self._logger.info("IK solution found!")
                print("                        ")
                self._logger.error(f"Valid goal is: {pose_goal}")
                print("                        ")   
                plan = True


                self.lite6_arm.set_goal_state(robot_state=robot_state) # set the GOAL STATE of the robot to the goal state
                robot_state.update() # Update the robot state, this will update the robot state to the goal state
                check_updated_pose = robot_state.get_pose("camera_depth_frame")
                print("New_pose:", check_updated_pose)
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions) # Set the joint group positions of the robot to the original joint positions
                robot_state.update() 

        if plan:
            plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.1)
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
