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
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import JointState
import os
import math
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import Constraints, JointConstraint
import move_joint_positions

# Maximum movement threshold
MAX_MOVEMENT_THRESHOLD = 0.025  # Meters -----> before it was 0.01
DEPTH_DISTANCE_STEP = 0.025
MINIMUM_DEPTH_DISTANCE = 0.15
MIN_DISTANCE_FROM_OBJECT = 0.15

class GoToPoseActionServer(Node):

    def __init__(self):
        super().__init__('control_server')
        self._action_server = ActionServer(
            self,
            GoToPose,
            'go_to_pose',
            execute_callback=self.execute_callback)
        self._logger = get_logger("go_to_pose_action_server")

        self.previous_position = Point(x=0.20749, y=0.059674, z=0.20719)

        self.joint_states_pub = self.create_publisher(JointState, '/control/joint_states', 10)
        self.stop_execution_sub = self.create_subscription(Bool, '/control/stop_execution', self.stop_execution_callback, 30)

        self.create_subscription(Float32, "/control/depth_adjustment", self.depth_adjustment_callback, 10)
        self.first_movement_publisher = self.create_publisher(Bool, "/control/first_movement", 10)
        self.initial_distance_y_pub = self.create_publisher(Float32, "/control/initial_distance_y", 10)

        self.pick_card = True
        self.first_movement = True
        self.initial_distance_y = 0.0
        self.distance_from_object = Float32()
        self.count = 0
        self.stop_execution = False

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
        self.get_logger().info("Lite6 initialized")

    def stop_execution_callback(self, msg):
        self.stop_execution = msg.data
        if self.stop_execution:
            while True:
                self.get_logger().info("Execution halted due to stop signal.")
                time.sleep(1.0)
            self._logger.info("Received stop execution signal. Halting operations.")

    def plan_and_execute(self, robot, planning_component, logger, sleep_time, single_plan_parameters=None, multi_plan_parameters=None, constraints=None):
        if self.stop_execution:
            logger.info("Execution halted due to stop signal.")
            return

        logger.info("Planning trajectory")

        if constraints is not None:
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
            self.first_movement = False
            msg = Bool()
            msg.data = False
            self.first_movement_publisher.publish(msg)


            if self.pick_card:
                # Publish joint states after executing the plan
                planning_scene_monitor = self.lite6.get_planning_scene_monitor()
                with planning_scene_monitor.read_only() as scene:
                    robot_state_to_publish = scene.current_state
                    robot_state_to_publish.update()

                    joint_state_msg = JointState()
                    joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                    joint_state_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
                    
                    joint_positions = robot_state_to_publish.get_joint_group_positions('lite6_arm')

                    if joint_positions is not None:
                        joint_positions_list = []
                        for pos in joint_positions:
                            joint_positions_list.append(pos)
                        joint_state_msg.position = joint_positions_list

                    self.joint_states_pub.publish(joint_state_msg)
                
                self.count += 1

                while self.count == 20:
                    self.joint_states_pub.publish(joint_state_msg)
                    time.sleep(5.0)
                    print("SLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEPPPPPPPPPPPPPPPPPPPPPIIIIIIIIIIIIIIIIIIIIIIIINNNNNNG")

        else:
            logger.error("Planning failed")
            time.sleep(0.5)

        time.sleep(sleep_time)

    def depth_adjustment_callback(self, msg):
        self.distance_from_object.data = msg.data

    def execute_callback(self, goal_handle):
        goal = goal_handle.request.pose

        self.get_logger().info(f"Received goal - Position: {goal.position}, Orientation: {goal.orientation}")

        if self.first_movement and goal.position.y != 0.0:
            self.initial_distance_y = goal.position.y
            distance_y_msg = Float32()
            distance_y_msg.data = self.initial_distance_y
            self.initial_distance_y_pub.publish(distance_y_msg)

        updated_camera_position = self.go_to_position(goal.position.x, goal.position.y, goal.position.z, goal.orientation)

        self.get_logger().info(f"Updated camera position: {updated_camera_position}")

        result = GoToPose.Result()
        result.success = True
        if updated_camera_position:
            result.updated_camera_position = Pose(
                position=updated_camera_position,
                orientation=goal.orientation
            )
        goal_handle.succeed()
        return result

    def go_to_position(self, movx, movy, movz, orientation):
        if self.stop_execution:
            self._logger.info("Execution halted due to stop signal.")
            rclpy.shutdown()
            return None

        plan = False
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        updated_camera_position = None
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")

            self.lite6_arm.set_start_state_to_current_state()
            check_init_pose = robot_state.get_pose("camera_color_optical_frame")

            if self.first_movement:  # First movement of the robot for reaching an object, then it aligns with the object
                print("First movement")
                if not self.pick_card:
                    movz = 0.15
                else:
                    movz = max(movz, 0.10)
                    #time.sleep(1.0)
                self.previous_position = check_init_pose.position

            else:  # Aligning the robot with the object

                if self.pick_card:  # For CreditCard
                    movy = check_init_pose.position.y
                else:  # For POS object
                    movy = check_init_pose.position.y + (movy - self.previous_position.y)

                # Update movz conditionally
                if movz > MIN_DISTANCE_FROM_OBJECT and not self.pick_card:  # For Pos object
                    movz = check_init_pose.position.z - DEPTH_DISTANCE_STEP
                else:  # for CreditCard
                    movz = check_init_pose.position.z + (movz - self.previous_position.z)

                # Ensure the new position is within the movement threshold
                dist_x = abs(movx - self.previous_position.x)
                dist_y = abs(movy - self.previous_position.y)
                dist_z = abs(movz - self.previous_position.z)

                if dist_x > MAX_MOVEMENT_THRESHOLD and not self.pick_card:
                    movx = self.previous_position.x + (MAX_MOVEMENT_THRESHOLD if movx > self.previous_position.x else -MAX_MOVEMENT_THRESHOLD)
                elif self.pick_card:
                    movx = self.previous_position.x + (MAX_MOVEMENT_THRESHOLD / 2 if movx > self.previous_position.x else -MAX_MOVEMENT_THRESHOLD / 2)

                if dist_y > MAX_MOVEMENT_THRESHOLD and not self.pick_card:
                    movy = self.previous_position.y + (MAX_MOVEMENT_THRESHOLD if movy > self.previous_position.y else -MAX_MOVEMENT_THRESHOLD)

                if dist_z > MAX_MOVEMENT_THRESHOLD / 2:
                    movz = self.previous_position.z + (MAX_MOVEMENT_THRESHOLD if movz > self.previous_position.z else -MAX_MOVEMENT_THRESHOLD)

            # Clipping the movement within specified boundaries
            movx = min(max(movx, 0.0), 0.45)
            movy = min(max(movy, -0.3), 0.45)
            if not self.pick_card:
                movz = min(max(movz, MINIMUM_DEPTH_DISTANCE), 0.40)
            else:
                movz = min(max(movz, 0.10), 0.40)

            pose_goal = Pose()
            pose_goal.position.x = movx
            pose_goal.position.y = movy
            pose_goal.position.z = movz
            pose_goal.orientation = orientation

            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=1.0)
            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
                return

            constraints = Constraints()
            constraints.name = "joints_constraints"

            if orientation.x == 0.64135:  # CreditCard
                joint_4_constraint = JointConstraint()
                joint_4_constraint.joint_name = "joint4"
                joint_4_constraint.position = original_joint_positions[3]
                joint_4_constraint.tolerance_above = 1.5
                joint_4_constraint.tolerance_below = 1.5
                joint_4_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_4_constraint)

                joint_5_constraint = JointConstraint()
                joint_5_constraint.joint_name = "joint5"
                joint_5_constraint.position = original_joint_positions[4]
                joint_5_constraint.tolerance_above = 1.5
                joint_5_constraint.tolerance_below = 1.5
                joint_5_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_5_constraint)

                joint_6_constraint = JointConstraint()
                joint_6_constraint.joint_name = "joint6"
                joint_6_constraint.position = original_joint_positions[5]
                joint_6_constraint.tolerance_above = 1.5
                joint_6_constraint.tolerance_below = 1.5
                joint_6_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_6_constraint)

            else:
                joint_4_constraint = JointConstraint()
                joint_4_constraint.joint_name = "joint4"
                joint_4_constraint.position = original_joint_positions[3]
                joint_4_constraint.tolerance_above = 1.5
                joint_4_constraint.tolerance_below = 1.5
                joint_4_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_4_constraint)

                joint_5_constraint = JointConstraint()
                joint_5_constraint.joint_name = "joint5"
                joint_5_constraint.position = original_joint_positions[4]
                joint_5_constraint.tolerance_above = 1.5
                joint_5_constraint.tolerance_below = 1.5
                joint_5_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_5_constraint)

                joint_6_constraint = JointConstraint()
                joint_6_constraint.joint_name = "joint6"
                joint_6_constraint.position = original_joint_positions[5]
                joint_6_constraint.tolerance_above = 1.5
                joint_6_constraint.tolerance_below = 1.5
                joint_6_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_6_constraint)

            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()

        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, constraints=constraints)
            time.sleep(1.0)
            updated_camera_position = robot_state.get_pose("camera_color_optical_frame").position
            self.previous_position = updated_camera_position
            return updated_camera_position


def main(args=None):
    rclpy.init(args=args)
    #initial_position = move_joint_positions.Movejoints() ###
    #initial_position.move_to_ready_position(position_name="CameraSearching") ###
    #time.sleep(3.0) ###
    action_server = GoToPoseActionServer()
    rclpy.spin(action_server)
    action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
