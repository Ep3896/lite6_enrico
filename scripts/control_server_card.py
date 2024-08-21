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
from std_msgs.msg import Bool, Float32, String
from sensor_msgs.msg import JointState
import os
import math
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import Constraints, JointConstraint
#import move_joint_positions
import locked_movement
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

# Maximum movement threshold
MAX_MOVEMENT_THRESHOLD = 0.025  # Meters -----> before it was 0.01
DEPTH_DISTANCE_STEP = 0.025
MINIMUM_DEPTH_DISTANCE = 0.15
MIN_DISTANCE_FROM_OBJECT = 0.15

class GoToPoseActionServer(Node):

    def __init__(self):
        super().__init__('control_server')

        movement_group = MutuallyExclusiveCallbackGroup()

        self._action_server = ActionServer(
            self,
            GoToPose,
            'go_to_pose',
            execute_callback=self.execute_callback, 
            callback_group=movement_group)
        self._logger = get_logger("go_to_pose_action_server")

        #self.previous_position = Point(x=0.20749, y=0.059674, z=0.20719) # This is the initial position of the camera, this has to be dinamic not static
        
        self.stop_execution_sub = self.create_subscription(Bool, '/control/stop_execution', self.stop_execution_callback, 30)
        self.create_subscription(Float32, "/control/depth_adjustment", self.depth_adjustment_callback, 10)
        self.create_subscription(String,'/control/obj_to_reach', self.obj_to_reach_callback, 10)
        self.searching_card_sub = self.create_subscription(Bool, '/control/searching_card', self.searching_card_callback, qos_profile=1, callback_group=movement_group)
        self.create_subscription(Point, '/control/bounding_box_center', self.bounding_box_center_callback, 10)

        self.joint_states_pub = self.create_publisher(JointState, '/control/joint_states', 10)
        self.pos_joint_positions_pub = self.create_publisher(JointState, '/control/pos_joint_positions', 10)
        self.first_movement_publisher = self.create_publisher(Bool, "/control/first_movement", 10)
        self.initial_distance_y_pub = self.create_publisher(Float32, "/control/initial_distance_y", 10)
        self.stop_locked_movement_pub = self.create_publisher(Bool, '/control/stop_locked_movement', 10)

        self.pick_card = True
        self.first_movement = True
        self.initial_distance_y = 0.0
        self.distance_from_object = Float32()
        self.count = 0
        self.stop_execution = False
        self.bounding_box_center = None #Point(x=0.0, y=200.0, z=0.0)
        self.direction = -1  # Initially moving left

        self.camera_searching = True

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

        robot_state = RobotState(self.lite6.get_robot_model())
        check_init_pose = robot_state.get_pose("camera_color_optical_frame")
        self.previous_position = check_init_pose.position


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def bounding_box_center_callback(self, msg):
        self.bounding_box_center = msg

    def obj_to_reach_callback(self, msg):
        if msg.data == "POS":
            self.pick_card = False
            self.get_logger().info("Received object to reach: POS")
            stop_msg = Bool()
            stop_msg.data = True
            self.stop_locked_movement_pub.publish(stop_msg)
        else:
            self.pick_card = True
            self.get_logger().info("Received object to reach: CreditCard")


    # This function is called when the camera is searching for the card, it has to move the robot from the initial position
    # to the position where the camera can detect the card
    """
    def searching_card_callback(self, msg):
        self.camera_searching = msg.data

        if self.stop_execution:
            self._logger.info("NOT MOVING DUE TO STOP EXECUTION SIGNAL")
            return
        
        if self.pick_card == True:
        # If the card is found and this is not the first movement, then the robot has to stop searching for the card
            if self.camera_searching == False: #or not self.first_movement:
                print("Card is found")
                return
            else:
                self.get_logger().info('Searching for the card...')
                planning_scene_monitor = self.lite6.get_planning_scene_monitor()
                robot_state = RobotState(self.lite6.get_robot_model())

                with planning_scene_monitor.read_write() as scene:
                    robot_state = scene.current_state
                    ee_pose = robot_state.get_pose("camera_color_optical_frame")

                    pose_goal = Pose()


                    # Retrieve the y-coordinate of the centroid of the object last seen
                    centroid_y = self.bounding_box_center.y  # Assuming bounding_box_center stores [x, y]
                    print("Centroid y: ", centroid_y)

                    # Adjust pose_goal.position.x based on centroid_y value
                    if centroid_y < 200:
                        pose_goal.position.x = ee_pose.position.x - 0.005
                        print("Moving to the left")
                    else:
                        pose_goal.position.x = ee_pose.position.x + 0.005
                        print("Moving to the right")
                    #### It has to be added here a condition so if the last centroid seen is 
                    #pose_goal.position.x = ee_pose.position.x + 0.005
                    pose_goal.position.y = ee_pose.position.y 
                    pose_goal.position.z = ee_pose.position.z

                    pose_goal.orientation.x = 0.64135
                    pose_goal.orientation.y = 0.6065
                    pose_goal.orientation.z = 0.3936
                    pose_goal.orientation.w = -0.25673


                    original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                    result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=1.0)

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
                    self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5) # it was 0.5 sleep time
                    updated_camera_position = robot_state.get_pose("camera_color_optical_frame").position
                    self.previous_position = updated_camera_position
    """
    def searching_card_callback(self, msg):
        self.camera_searching = msg.data

        if self.stop_execution:
            self._logger.info("NOT MOVING DUE TO STOP EXECUTION SIGNAL")
            return

        if self.pick_card:
            if not self.camera_searching:
                print("Card is found")
                return
            else:
                self.get_logger().info('Searching for the card...')
                planning_scene_monitor = self.lite6.get_planning_scene_monitor()
                robot_state = RobotState(self.lite6.get_robot_model())

                with planning_scene_monitor.read_write() as scene:
                    robot_state = scene.current_state
                    ee_pose = robot_state.get_pose("camera_color_optical_frame")

                    pose_goal = Pose()

                    # If no centroid is detected
                    if self.bounding_box_center is None or self.bounding_box_center.y == 0.0:
                        # Change direction if limits are reached
                        if ee_pose.position.x <= 0.1:
                            self.direction = 1  # Move right
                            print("Reached lower limit. Changing direction to right.")
                        elif ee_pose.position.x >= 0.39:
                            self.direction = -1  # Move left
                            print("Reached upper limit. Changing direction to left.")

                        # Apply the movement based on the current direction
                        pose_goal.position.x = ee_pose.position.x + (0.005 * self.direction)
                    else:
                        # Adjust pose_goal.position.x based on centroid_y value
                        centroid_y = self.bounding_box_center.y
                        if centroid_y < 220: ########à ERA 200
                            pose_goal.position.x = ee_pose.position.x - 0.005
                            print("Moving to the left based on centroid:", centroid_y)
                        else:
                            pose_goal.position.x = ee_pose.position.x + 0.005
                            print("Moving to the right based on centroid", centroid_y)

                    # Ensure the position stays within the limits
                    #pose_goal.position.x = max(0.1, min(pose_goal.position.x, 0.39))
                    pose_goal.position.y = ee_pose.position.y
                    pose_goal.position.z = ee_pose.position.z

                    pose_goal.orientation.x = 0.64135
                    pose_goal.orientation.y = 0.6065
                    pose_goal.orientation.z = 0.3936
                    pose_goal.orientation.w = -0.25673

                    original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                    result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=1.0)

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
                    updated_camera_position = robot_state.get_pose("camera_color_optical_frame").position
                    self.previous_position = updated_camera_position




    # this is recieved by move_joint_positions.py after the robot has aligned with the object and has to stop other movements to it can perform the pick
    # Ideally , it has to be a service, but I am using a topic for now
    # Moreover, this snippet has to be fixed as it has to resume the movement after the pick for reaching the POS object
    def stop_execution_callback(self, msg):
        self.stop_execution = msg.data
        if self.stop_execution:
            # This is a blocking loop, I think it has to stay here until the robot has reached the ready position again after the pick
            #while True:
            #    self.get_logger().info("Execution halted due to stop signal.")
            #    time.sleep(1.0)
            # INSTEAD OF THIS, COULD I JUST RETURN?
            robot_state = RobotState(self.lite6.get_robot_model())
            self.previous_position = robot_state.get_pose("camera_color_optical_frame").position

            return



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
            if not self.camera_searching:
                msg = Bool()
                msg.data = self.first_movement
                self.first_movement_publisher.publish(msg)
                self.first_movement = False


            if self.pick_card and not self.camera_searching:
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

            elif not self.pick_card:
                # What do I have to do here?
                # 1) Publish the joint states to a new node that will find the best position for the camera to align with the POS object
                # 2) After that, the new node will publish a stop message to this node, so this node will stop the execution
                print("Publishing joint states for the POS object")
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

        else:
            logger.error("Planning failed")
            time.sleep(0.5)

        time.sleep(sleep_time)

    def depth_adjustment_callback(self, msg):
        self.distance_from_object.data = msg.data

    # This function is called when a new goal is received, this is the main function that moves the robot.
    # It includes the logic for the first movement and the alignment with the object
    # It also includes the logic for the depth adjustment
    # It comprises the go_to_position function that is responsible for moving the robot
        
    def execute_callback(self, goal_handle):
        goal = goal_handle.request.pose

        self.get_logger().info(f"Received goal - Position: {goal.position}, Orientation: {goal.orientation}")

        if self.first_movement and goal.position.y != 0.0:
            self.initial_distance_y = goal.position.y
            distance_y_msg = Float32()
            distance_y_msg.data = self.initial_distance_y
            self.initial_distance_y_pub.publish(distance_y_msg)
            print("Initial distance y: ", self.initial_distance_y)

        if self.stop_execution:
            self._logger.info("NOT MOVING DUE TO STOP EXECUTION SIGNAL")
            goal_handle.abort()  # Abort the goal, this means that I ignore the movement request from the client
            result_server = GoToPose.Result()
            result_server.success = False
            return result_server
        else:
            updated_camera_position = self.go_to_position(goal.position.x, goal.position.y, goal.position.z, goal.orientation)

            self.get_logger().info(f"Updated camera position: {updated_camera_position}")

            result_server = GoToPose.Result()
            result_server.success = True
            if updated_camera_position:
                result_server.updated_camera_position.position = updated_camera_position
                result_server.updated_camera_position.orientation = goal.orientation
            else:
                result_server.success = False
            goal_handle.succeed()
            return result_server


    def go_to_position(self, movx, movy, movz, orientation):

        # Check if the execution has been stopped, it has to be handled using another approach --------------->
        # it has to wait until the robot has reached the ready position again after the pick
        if self.stop_execution:
            self._logger.info("Execution halted due to stop signal.")
            rclpy.shutdown()
            return None
        
        if self.camera_searching:
            self._logger.info("Camera is searching for the card. Not moving the robot.")
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
                    movz = 0.2  #################### FOR THE POS OBJECT, this has to be changed, I think it would be good to read the depth from the camera sensor.
                else:
                    movz = max(movz, 0.15)
                    movy = movy - 0.1 #0.12 was good # this 0.1 has been done beacuse the camera has to be a bit distant from the object , otherwise it will not detect it

            else:                   # Aligning the robot with the object

                if self.pick_card:  # For CreditCard
                    movy = check_init_pose.position.y
                else:               # For POS object
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
                    movx = self.previous_position.x + (-MAX_MOVEMENT_THRESHOLD if movx > self.previous_position.x else MAX_MOVEMENT_THRESHOLD)
                elif self.pick_card:
                    movx = self.previous_position.x + (-MAX_MOVEMENT_THRESHOLD / 2 if movx > self.previous_position.x else MAX_MOVEMENT_THRESHOLD / 2) ###CHANGED THE SIGN

                if dist_y > MAX_MOVEMENT_THRESHOLD and not self.pick_card:
                    movy = self.previous_position.y + (MAX_MOVEMENT_THRESHOLD if movy > self.previous_position.y else -MAX_MOVEMENT_THRESHOLD)

                if dist_z > MAX_MOVEMENT_THRESHOLD / 2:
                    movz = self.previous_position.z + (MAX_MOVEMENT_THRESHOLD if movz > self.previous_position.z else -MAX_MOVEMENT_THRESHOLD)

            # Clipping the movement within specified boundaries
            movx = min(max(movx, 0.0), 0.45)
            movy = min(max(movy, -0.3), 0.45)
            if not self.pick_card:
                movz = min(max(movz, 0.30), 0.30)
            else:
                movz = min(max(movz, 0.15), 0.40)

            pose_goal = Pose()
            pose_goal.position.x = movx # changed the sign!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_goal.position.y = movy
            pose_goal.position.z = movz
            pose_goal.orientation = orientation

            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=1.0)
            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                self._logger.error(f"Failed goal is: {pose_goal}")
                updated_camera_position = robot_state.get_pose("camera_color_optical_frame").position
                self.previous_position = updated_camera_position
                return updated_camera_position

            constraints = Constraints()
            constraints.name = "joints_constraints"

            if orientation.x == 0.64135:  # CreditCard


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
    action_server = GoToPoseActionServer()
    rclpy.spin(action_server)
    action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
