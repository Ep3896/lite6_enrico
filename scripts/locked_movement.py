import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Bool, Float32MultiArray, String
from sensor_msgs.msg import JointState
from threading import Event
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import time
import threading
import robot_control
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import Constraints, JointConstraint
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit.core.planning_scene import PlanningScene



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

        self.create_subscription(Float32, '/control/bbox_area', self.bbox_area_callback, 30)
        self.create_subscription(JointState, '/control/joint_states', self.joint_states_callback, 10)
        self.create_subscription(Float32, '/control/initial_distance_y', self.initial_distance_y_callback, 10)
        self.create_subscription(Float32, '/control/depth_adjustment', self.depth_adjustment_callback, 100)
        self.create_subscription(Float32MultiArray, '/control/bbox_center', self.bbox_center_callback, 10)
        self.create_subscription(Bool, '/control/alignment_status', self.alignment_status_callback, 30)
        self.create_subscription(Bool, '/control/stop_locked_movement', self.stop_locked_movement_callback, 10)
        self.create_subscription(Bool, '/control/line_is_far', self.line_is_far_callback, 30)

        self.stop_execution_pub = self.create_publisher(Bool, '/control/stop_execution', 30)
        self.pointcloud_pub = self.create_publisher(Bool, '/control/start_pointcloud', 10)
        self.template_matching_pub = self.create_publisher(Bool, '/control/start_template_matching', 10)
        self.card_edge_detection_pub = self.create_publisher(Bool, '/control/start_card_edge_detection', 10)
        self.object_to_reach_pub = self.create_publisher(String, '/control/obj_to_reach', 10)
        
        self.current_joint_states = None
        self.initial_distance_y = None
        self.depth_adjustment = None  # Initialize the depth adjustment variable
        self.previous_depth_adjustment = None  # Initialize the previous depth adjustment variable
        self.bbox_center = None  # Initialize bounding box center
        self.bbox_center_lock = threading.Lock()  # Lock for bbox_center
        self.alignment_ok_event = Event()  # Initialize alignment event
        self.bbox_center_event = Event()  # Event to wait for bbox_center
        self.alignment_within_threshold_event = Event()  # Event to wait for alignment within threshold

        self.card_position_values = None

        self.first_alignment = True

        self.previous_error_x = 0
        self.stop_locked_movement = False

        self.attempts = 1
        self.x_movement = 0.0

        self.line_is_far = None

        # Timer to print joint positions periodically
        #self.timer = self.create_timer(0.5, self.print_joint_positions) #it was 1.0

        # Publish True to start pointcloud at the beginning
        self.pointcloud_pub.publish(Bool(data=True))

        # Timer for alignment with card edge
        self.alignment_timer = self.create_timer(2.0, self.align_with_card_edge_callback) # it was 2.0 


    def line_is_far_callback(self, msg):
        self.line_is_far = msg.data

    def depth_adjustment_callback(self, msg):
        self.previous_depth_adjustment = self.depth_adjustment
        self.depth_adjustment = msg.data
        self.get_logger().info(f'Received depth adjustment: {self.depth_adjustment} meters')

    def initial_distance_y_callback(self, msg):
        self.initial_distance_y = msg.data
        self.get_logger().info(f'Initial distance y: {self.initial_distance_y}')

    def bbox_center_callback(self, msg):
        with self.bbox_center_lock:
            self.bbox_center = msg.data
        self.bbox_center_event.set()  # Set the event when bbox_center is received
        self.get_logger().info(f'Received bounding box center: {self.bbox_center}')

    def alignment_status_callback(self, msg):
        self.get_logger().info(f'Received alignment status: {msg.data}')
        if msg.data == True: # This condition is true when camera and card are aligned
            self.alignment_ok_event.set()  # Set the event when alignment is OK
            self.get_logger().info("Alignment OK event set")
            if self.first_alignment:
                self.first_alignment = False
                #self.move_ee_to_camera_pos()
                #time.sleep(1.0)
                #self.continue_process_after_alignment()
        else:
            self.alignment_ok_event.clear()  # Clear the event otherwise
            self.get_logger().info("Alignment OK event cleared")

    def stop_locked_movement_callback(self, msg):
        self.stop_locked_movement = msg.data
        if self.stop_locked_movement:
            self.get_logger().info("Received stop locked movement signal.")

    def print_joint_positions(self):
        if self.current_joint_states:
            joint_positions = list(self.current_joint_states.position)
            self.get_logger().info(f'Print joint positions: {joint_positions}')
        else:
            self.get_logger().info('Joint positions not available yet.')

    def bbox_area_callback(self, msg):
        if self.current_joint_states and not self.stop_locked_movement:
            joint_positions = list(self.current_joint_states.position)
            bbox_area = msg.data
            self.align_positions.append((joint_positions, bbox_area))
            self.get_logger().info(f'Added configuration with bbox area: {bbox_area}')
            print('                            ')
            print('Dimensione align_positions', len(self.align_positions))
            print('                            ')

            if len(self.align_positions) >= 3 and not self.stop_locked_movement:
                self.get_logger().info(f'Buffer size is {len(self.align_positions)}, selecting max area configuration')
                stop_msg = Bool()
                stop_msg.data = True
                self.stop_execution_pub.publish(stop_msg)
                self.select_max_area_configuration()

    def joint_states_callback(self, msg):
        self.current_joint_states = msg

    def select_max_area_configuration(self):
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
            time.sleep(0.5)
            self.get_logger().info("Moving above card")
            #self.move_above_card_translation()
            #time.sleep(1.5)
            self.move_above_card_rotation()
            # Start the alignment timer
            self.alignment_ok_event.clear()
            self.alignment_within_threshold_event.clear()  # Clear the event before starting alignment
            
            alignment_thread = threading.Thread(target=self.wait_for_alignment)
            alignment_thread.start()

    def wait_for_alignment(self):
        self.get_logger().info("Waiting for alignment within threshold...")
        self.alignment_within_threshold_event.wait()  # Block until alignment is within threshold
        self.get_logger().info("Alignment within threshold, proceeding with next steps...")
        #self.continue_process_after_alignment()

    def continue_process_after_alignment(self):
        self.get_logger().info("Moving to camera position")
        #self.move_ee_to_camera_pos()
        robot = robot_control.RobotControl()
        robot.open_gripper()
        time.sleep(0.5)
        self.move_down_to_card()
        time.sleep(0.5)
        robot.close_gripper()
        time.sleep(0.5)
        self.store_card_position()
        time.sleep(0.5)
        self.move_to_ready_position(position_name="PosSearching")
        #time.sleep(1.5)
        #self.mv_to_card_position()
        ######################## HERE I HAVE TO PUBLSISH TO /CONTROL/OBJ_TO_REACH IN ORDER TO REACH THE POS AND START THE TEMPLATE MATCHING
        # Unblock the cotrol_server_card node
        stop_msg = Bool()
        stop_msg.data = False
        self.stop_execution_pub.publish(stop_msg)
        # Search for POS
        object_to_reach_msg = String()
        object_to_reach_msg.data = "POS"
        self.object_to_reach_pub.publish(object_to_reach_msg)
        # I also need to start again the pointcloud to detect the POS
        self.pointcloud_pub.publish(Bool(data=True))
        time.sleep(1.5)
        #rclpy.shutdown()

        #time.sleep(20)


    def move_ee_to_camera_pos(self):
        self.get_logger().info("Moving EE to camera position")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            ee_pose = robot_state.get_pose("link_tcp")
            camera_pose = robot_state.get_pose("camera_color_optical_frame") #camera_color_optical_frame

            pose_goal = Pose()
            """
            ###### Maybe it is better to refactor it so 
            if camera_pose.position.x > 0.35: #When the card is a bit far away from the robot link, it is better to change the way to catch the card
                pose_goal.position.x = camera_pose.position.x + 0.03
                pose_goal.position.y = camera_pose.position.y
                pose_goal.position.z = ee_pose.position.z

                pose_goal.orientation.x = 0.707
                pose_goal.orientation.y = 0.0
                pose_goal.orientation.z = 0.707
                pose_goal.orientation.w = 0.0
            else:
                pose_goal.position.x = camera_pose.position.x + 0.03
                pose_goal.position.y = camera_pose.position.y
                pose_goal.position.z = ee_pose.position.z

                pose_goal.orientation = ee_pose.orientation
            """
            pose_goal.position.x = camera_pose.position.x #+ 0.03
            pose_goal.position.y = camera_pose.position.y
            pose_goal.position.z = ee_pose.position.z

            pose_goal.orientation = ee_pose.orientation
          
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=2.0)

            robot_state.update()

            while not result : #and camera_pose.position.x > 0.3:
                self._logger.error("IK solution was not found, changing grasping pose")

                pose_goal.position.x = camera_pose.position.x + 0.03
                pose_goal.position.y = camera_pose.position.y
                pose_goal.position.z = ee_pose.position.z
 
                pose_goal.orientation.x = 0.707
                pose_goal.orientation.y = 0.0
                pose_goal.orientation.z = 0.707
                pose_goal.orientation.w = 0.0

                original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=2.0)

                robot_state.update()

                if not result:
                    self._logger.error("IK solution was not found despite changing the grasping pose")
                    return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()
        
        # initialise multi-pipeline plan request parameters
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
        self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
        )
        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, multi_plan_parameters=multi_pipeline_plan_request_params)

    def move_above_card_translation(self):
        self.get_logger().info("Moving EE above card")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame") #link_tcp before
            print('Final y of the target', self.depth_adjustment)

            pose_goal = Pose()
            pose_goal.position.x = camera_pose.position.x 
            pose_goal.position.y = camera_pose.position.y  # Adjust using depth adjustment value
            if self.depth_adjustment is not None:
                pose_goal.position.y += self.depth_adjustment
            else:
                if self.previous_depth_adjustment is not None:
                    pose_goal.position.y += self.previous_depth_adjustment
                else:
                    self.get_logger().warning('Depth adjustment not available')
            pose_goal.position.z = camera_pose.position.z + 0.05 # Adjust using depth adjustment value

            pose_goal.orientation.x = camera_pose.orientation.x
            pose_goal.orientation.y = camera_pose.orientation.y
            pose_goal.orientation.z = camera_pose.orientation.z
            pose_goal.orientation.w = camera_pose.orientation.w
            
            robot_state.update()

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
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=1.5)


    def move_above_card_rotation(self):
        self.get_logger().info("Moving EE above card Rotation")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame") #link_tcp before
            print('Final y of the target', self.depth_adjustment)

            pose_goal = Pose()
            pose_goal.position.x = camera_pose.position.x 
            pose_goal.position.y = camera_pose.position.y  # Adjust using depth adjustment value

            if self.depth_adjustment is not None:
                pose_goal.position.y += self.depth_adjustment
            else:
                if self.previous_depth_adjustment is not None:
                    pose_goal.position.y += self.previous_depth_adjustment
                else:
                    self.get_logger().warning('Depth adjustment not available')

            pose_goal.position.z = camera_pose.position.z + 0.05 # Adjust using depth adjustment value

            # -0.71, 0.71, -0.02, 0.02
            pose_goal.orientation.x = -0.71
            pose_goal.orientation.y = 0.71
            pose_goal.orientation.z = -0.02
            pose_goal.orientation.w = 0.02
            
            robot_state.update()

            
            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            #result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=60.0)
            

            ##################### I must check joint two of the new robot state, if joint 2 is less than 1.0, I have to recompute the IK solution


            # Loop to find a valid IK solution where joint2 >= 1.0
            max_attempts = 100  # To prevent infinite loops, limit the number of attempts
            valid_solution = False

            while self.attempts < max_attempts:
                # Perturb or randomize the robot state before each attempt
                # Instead of set_to_random_positions(), perturb around the current state
                joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                
                print('Attempt', self.attempts)

                if self.attempts < 3:
                    robot_state = scene.current_state
                else:
                    robot_state.set_to_random_positions()
                
                robot_state.update()

                # Attempt to solve IK with the perturbed joint positions
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=5.0)
                joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                joint2_position = joint_positions[1]  # Assuming joint2 is at index 1

                time.sleep(1.0)  # Short pause between attempts for safety

                print('Result', result)

                if result and joint2_position <= 1.0:  # Check if the IK solution is valid and joint2 meets the condition
                    valid_solution = True
                    break  # Valid solution found, exit the loop
                else:
                    self.get_logger().info(f"Invalid IK solution with joint2 = {joint2_position}. Retrying...")

                self.attempts += 1

                #if attempts < 5:
                #pose_goal.position.x += 0.01  # Increase the z position slightly for each attempt
                #pose_goal.position.x += np.random.uniform(0.0, 0.02)  # Small perturbation range
                #if pose_goal.position.x < 0.2:
                #    self.x_movement += 0.01*(1 - 1/(self.attempts+1))
                #    print('X movement', self.x_movement)
                #    pose_goal.position.x += self.x_movement  # Small perturbation range



            if not valid_solution:
                self._logger.error("Failed to find a valid IK solution with joint2 >= 1.0 after multiple attempts.")
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)

                ################
                """
                collision_request = CollisionRequest()
                collision_result = CollisionResult()
                #self.lite6.get_planning_component("lite6_arm").get_planning_scene().check_self_collision(collision_request, collision_result)
                planning_scene = PlanningScene(self.lite6.get_robot_model())
                planning_scene.check_self_collision(collision_request, collision_result, robot_state)
                if collision_result.collision:
                    self.get_logger().info("Self-collision detected")
                    rclpy.shutdown()
                    # Handle the collision scenario, e.g., stopping the robot
                else:
                    self.get_logger().info("No self-collision detected")

                rclpy.shutdown()
                """
                ################


                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()

            constraints = Constraints()
            constraints.name = "joints_constraints"

            joint_2_constraint = JointConstraint()
            joint_2_constraint.joint_name = "joint2"
            joint_2_constraint.position = original_joint_positions[1]
            joint_2_constraint.tolerance_above = 1.5   # Upper limit = Desired value + Tolerance above, this is the one that prevents the collision between link 3 aand robot base
            joint_2_constraint.tolerance_below = 1.5   # Lower limit = Desired value − Tolerance below
            joint_2_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_2_constraint)
            
            # Before all the blocks of code before if plan was tabbed on the left
            # initialise multi-pipeline plan request parameters
            multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
            )


        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, constraints=constraints ,multi_plan_parameters=multi_pipeline_plan_request_params)#, constraints=constraints)
        # Switch to card edge detection after moving above card
        self.pointcloud_pub.publish(Bool(data=False))
        self.card_edge_detection_pub.publish(Bool(data=True))

    def move_down_to_card(self):
        self.get_logger().info("Moving EE down to card")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state 
            ee_pose = robot_state.get_pose("link_tcp")

            pose_goal = Pose()
            #if pose_goal.position.x < 0.2 and pose_goal.position.x != 0.0:
            if self.line_is_far == True:
                pose_goal.position.x = ee_pose.position.x + 0.03 #np.clip(0.01* (1/pose_goal.position.x), 0.0, 0.05) #- 1.5*self.x_movement 
                print('LINE IS FAR', pose_goal.position.x)
            else:
                pose_goal.position.x = ee_pose.position.x
            pose_goal.position.y = ee_pose.position.y
            if ee_pose.position.z <= 0.16:
                pose_goal.position.z = ee_pose.position.z - 0.05   ### This can be retrieved by looking at the depth in the pixel that show the board
            else:
                pose_goal.position.z = ee_pose.position.z - 0.08

            pose_goal.orientation = ee_pose.orientation
            self.x_movement = 0.0

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

    def align_with_card_edge_callback(self):
        if not self.alignment_ok_event.is_set():
            self.get_logger().info("Aligning with card edge")
            with self.bbox_center_lock:
                bbox_center = self.bbox_center

            if bbox_center:
                bbox_center_x = bbox_center[0]
                frame_center_x = 320   # In this case, frame width is 640 and height is 480
                error_x = frame_center_x - bbox_center_x

                self.get_logger().info(f"Error in X axis: {error_x}")
                self.get_logger().info(f"Bounding box center: {bbox_center}")

                if abs(error_x) > 20:  # Adjust this threshold as needed
                    self.adjust_robot_position(error_x)
                    self.get_logger().info("Looping for alignment")  # Beware that there was an attempt in which here continous_process_after_alignment was called method was called
                    #self.move_ee_to_camera_pos()
                    #time.sleep(1.0)
                    #self.continue_process_after_alignment()
                else:
                    self.get_logger().info("Alignment within threshold, breaking loop")
                    self.alignment_within_threshold_event.set()  # Set the event to indicate alignment is within threshold
                    self.alignment_ok_event.set()  # Manually set the event to break the loop
                    # Now I have to cancel the timer
                    self.alignment_timer.cancel()
                    self.move_ee_to_camera_pos()
                    time.sleep(0.5)
                    self.continue_process_after_alignment()
            else:
                self.get_logger().info("Bounding box center not available")

    def adjust_robot_position(self, error_x):
        if self.stop_locked_movement:
            self.get_logger().info("Adjust movement halted due to stop signal.")
            return

        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame")
            
            pose_goal = Pose()
            pose_goal.position.x = camera_pose.position.x   # Adjust scaling factor as needed
            pose_goal.position.z = camera_pose.position.z
            # it can be positive or negative, depending on the error
            # if the error is positive, the robot has to move to the left
            # if the error is negative, the robot has to move to the right
            # I also measure the difference between the previous error and the current one
            # So I can understand if the robot is moving in the right direction and if the error is decreasing, move slower
            # I call Kd the scaling factor that is the difference between the previous error and the current one
            # Proportional-Derivative (PD) control law
            Kp = 0.0001  # Proportional gain, adjust as needed
            #Kd = 0.0001  # Derivative gain, adjust as needed
                
            # Calculate derivative term (rate of change of error)
            #delta_error_x = error_x - self.previous_error_x

            adjustment =  Kp * error_x #+ Kd * delta_error_x

            pose_goal.position.y = camera_pose.position.y + adjustment

            self.previous_error_x = error_x

            pose_goal.orientation.x = camera_pose.orientation.x
            pose_goal.orientation.y = camera_pose.orientation.y
            pose_goal.orientation.z = camera_pose.orientation.z
            pose_goal.orientation.w = camera_pose.orientation.w

            print('Alignment adjustment on y to be made', adjustment)

            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=2.0)

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

    def move_to_ready_position(self, position_name): ####
        self.get_logger().info("Moving EE to Ready position")

        self.lite6_arm.set_start_state_to_current_state()
        self.lite6_arm.set_goal_state(configuration_name=position_name)  ####
        plan_result = self.lite6_arm.plan()

        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.lite6.execute(robot_trajectory, controllers=[])

    def store_card_position(self):
        self.get_logger().info("Moving to card initial position")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state_grasping = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_scene_grasping = scene.current_state
            self.card_position_values = robot_scene_grasping.get_joint_group_positions("lite6_arm")

    def mv_to_card_position(self):
        self.get_logger().info("Moving to card position")
        self.lite6_arm.set_start_state_to_current_state()
        robot_state = RobotState(self.lite6.get_robot_model())
        robot_state.set_joint_group_positions("lite6_arm", self.card_position_values)
        robot_state.update()
        self.lite6_arm.set_goal_state(robot_state=robot_state)
        plan_result = self.lite6_arm.plan()
        if plan_result and self.card_position_values is not None:
            robot_trajectory = plan_result.trajectory
            self.lite6.execute(robot_trajectory, controllers=[])
            self.get_logger().info("Robot moved to the card position")

    def plan_and_execute(self, robot, planning_component, logger, sleep_time, constraints=None,single_plan_parameters=None, multi_plan_parameters=None):
        if self.stop_locked_movement:
            logger.info("Execution halted due to stop signal.")
            return

        logger.info("Planning trajectory")
    
        if constraints is not None:
            print('Constraints AAAAAAAAAAAAAAAAAAAAAAAAAAAAA', constraints)
            planning_component.set_path_constraints(constraints)

        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(multi_plan_parameters=multi_plan_parameters)
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(single_plan_parameters=single_plan_parameters)
        else:
            plan_result = planning_component.plan()

        if plan_result:
            if self.stop_locked_movement:
                logger.info("Execution halted due to stop signal after planning.")
                return
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
    #storing_configurations_area.move_above_card_rotation()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(storing_configurations_area)
    
    try:
        executor.spin()
    finally:
        storing_configurations_area.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
