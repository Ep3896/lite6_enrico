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
import math
from simple_pid import PID
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import Constraints, JointConstraint
from visualization_msgs.msg import MarkerArray
from yolov8_msgs.msg import DetectionArray



# PID gains
KP = 0.1
KI = 0.01
KD = 0.01

# Maximum movement threshold
MAX_MOVEMENT_THRESHOLD = 1.0  # Meters

# Plan and execute function
def plan_and_execute(robot, planning_component, logger, sleep_time, single_plan_parameters=None, multi_plan_parameters=None, constraints=None):
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
        self.create_subscription(DetectionArray, '/yolo/detections_3d', self.detection_callback, 30)
        self.rotate = False
        self.current_detections = []

        self.pick_card = True
        self.current_detections = []

        # PID controllers for x, y, z
        self.pid_x = PID(KP, KI, KD)
        self.pid_y = PID(KP, KI, KD)
        self.pid_z = PID(KP, KI, KD)
        self.pid_x.sample_time = 0.0333
        self.pid_y.sample_time = 0.0333
        self.pid_z.sample_time = 0.0333

        # Initial camera position during Ready pose
        self.previous_position = Point(x=0.20687, y=0.068421, z=0.21905)

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

        # Initial configuration based on the task
        if self.pick_card:  # For now, this flag will depend on the callback of the control_logic node
            self.lite6_arm.set_start_state_to_current_state()
            self.lite6_arm.set_goal_state(configuration_name="CameraSearching")
            plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.1)
            print("Moving to CameraSearching")
        
    def detection_callback(self, msg: DetectionArray):
        if msg.detections:
            self.current_detections = msg.detections

    def rotation_callback(self, msg):
        if msg.data:
            self.rotate = True

    def execute_callback(self, goal_handle):
        goal = goal_handle.request.pose

        """
        # Adjust position if picking the card
        if self.pick_card == True and self.current_detections:
            goal.position.y -= self.current_detections[0].bbox3d.size.y/2# Adjust y by half the size of the detected object
            if goal.position.y - self.current_detections[0].bbox3d.size.y < 0.0:
                goal.position.y += 0.1
            goal.position.z -= self.current_detections[0].bbox3d.size.z # Adjust z by half the size of the detected object
            if goal.position.z - self.current_detections[0].bbox3d.size.z < 0.0:
                goal.position.z = 0.0
        """

        # Set PID setpoints to the goal positions
        self.pid_x.setpoint = goal.position.x
        self.pid_y.setpoint = goal.position.y
        self.pid_z.setpoint = goal.position.z

        print("                        ")
        print("++++++++Goal position+++++:", goal.position)
        print("                        ")
        print("++++++++Goal orientation+++++:", goal.orientation)

        updated_camera_position = self.go_to_position(goal.position.x, goal.position.y + 0.25, goal.position.z, goal.orientation)

        print("                        ")
        print("Updated camera position:", updated_camera_position)
        print("                        ")

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
        plan = False
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        updated_camera_position = None
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            print("Original joint positions:", original_joint_positions)

            self.lite6_arm.set_start_state_to_current_state()
            check_init_pose = robot_state.get_pose("camera_depth_frame")

            # PID control
            current_position = check_init_pose.position
            velocity_x = self.pid_x(current_position.x)
            velocity_y = self.pid_y(current_position.y)
            velocity_z = self.pid_z(current_position.z)

            # Compute the new position
            movx = current_position.x + 5 * velocity_x * self.pid_x.sample_time
            movy = current_position.y + 5 * velocity_y * self.pid_y.sample_time
            movz = current_position.z + 5 * velocity_z * self.pid_z.sample_time

            # Ensure the new position is within the movement threshold
            dist_x = abs(movx - self.previous_position.x)
            dist_y = abs(movy - self.previous_position.y)
            dist_z = abs(movz - self.previous_position.z)

            if dist_x > 2 * MAX_MOVEMENT_THRESHOLD:
                movx = self.previous_position.x + (MAX_MOVEMENT_THRESHOLD if movx > self.previous_position.x else -MAX_MOVEMENT_THRESHOLD)

            if dist_y > 2 * MAX_MOVEMENT_THRESHOLD:
                movy = self.previous_position.y + (MAX_MOVEMENT_THRESHOLD if movy > self.previous_position.y else -MAX_MOVEMENT_THRESHOLD)

            if dist_z > MAX_MOVEMENT_THRESHOLD / 2:
                movz = self.previous_position.z + (MAX_MOVEMENT_THRESHOLD if movz > self.previous_position.z else -MAX_MOVEMENT_THRESHOLD)


            # Clipping the movement within specified boundaries
            movx = min(max(movx, 0.2), 0.45)
            movy = min(max(movy, -0.3), 0.5)  # it was 0.3
            movz = min(max(movz, 0.13), 0.40)

            pose_goal = Pose()
            pose_goal.position.x = movx
            pose_goal.position.y = movy
            pose_goal.position.z = movz

            pose_goal.orientation = orientation

            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_depth_frame", timeout=1.0)
            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
                return

            constraints = Constraints()
            constraints.name = "joints_constraints"

            if orientation.x == 0.69237:

                joint_4_constraint = JointConstraint()
                joint_4_constraint.joint_name = "joint4"
                joint_4_constraint.position = original_joint_positions[3]  # 0.0
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
                joint_5_constraint = JointConstraint()
                joint_5_constraint.joint_name = "joint5"
                joint_5_constraint.position = 1.5708  # 90 degrees in radians
                joint_5_constraint.tolerance_above = 1.0
                joint_5_constraint.tolerance_below = 1.0
                joint_5_constraint.weight = 1.0

                constraints.joint_constraints.append(joint_5_constraint)

                joint_4_constraint = JointConstraint()
                joint_4_constraint.joint_name = "joint4"
                joint_4_constraint.position = 0.000010  # 0.0
                joint_4_constraint.tolerance_above = 1.0
                joint_4_constraint.tolerance_below = 1.0
                joint_4_constraint.weight = 1.0

                constraints.joint_constraints.append(joint_4_constraint)

                joint_6_constraint = JointConstraint()
                joint_6_constraint.joint_name = "joint6"
                joint_6_constraint.position = 0.0
                joint_6_constraint.tolerance_above = 1.0
                joint_6_constraint.tolerance_below = 1.0
                joint_6_constraint.weight = 1.0

                constraints.joint_constraints.append(joint_6_constraint)
            """
            robot_collision_status = scene.is_state_colliding(
                robot_state=robot_state, joint_model_group_name="lite6_arm", verbose=True
            )
            self._logger.info(f"\nRobot is in collision: {robot_collision_status}\n")

            if robot_collision_status:
                time.sleep(0.5)
                return
            """

            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                check_updated_pose = robot_state.get_pose("camera_depth_frame")
                print("New_pose:", check_updated_pose)
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()

        if plan:
            plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, constraints=constraints)
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
