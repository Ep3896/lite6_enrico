import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Bool, Float32MultiArray, String
from sensor_msgs.msg import JointState, Image as msg_Image, CameraInfo
from threading import Event
from geometry_msgs.msg import Pose, PoseStamped, Point
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from moveit.core.planning_scene import PlanningScene
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit_msgs.msg import Constraints, JointConstraint



class PosMovement(Node):

    def __init__(self):
        super().__init__('pos_movement')
        self.align_positions = []
        self.obj_to_reach = 'CreditCard'  # Initialize obj_to_reach ############### BEWARE, TO CHANGE FOR TESTING ONLY POS

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

        self.get_logger().info("PosMovement node has been started")

        self.create_subscription(Float32, '/control/bbox_area', self.bbox_area_callback, 10)
        self.create_subscription(JointState, '/control/joint_states', self.joint_states_callback, 10)
        self.create_subscription(Float32, '/control/initial_distance_y', self.initial_distance_y_callback, 10)
        self.create_subscription(Float32, '/control/depth_adjustment', self.depth_adjustment_callback, 100)
        self.create_subscription(Float32MultiArray, '/control/bbox_center', self.bbox_center_callback, 30)
        self.create_subscription(Bool, '/control/alignment_status', self.alignment_status_callback, 10)
        self.create_subscription(Bool, '/control/stop_pos_movement', self.stop_pos_movement_callback, 10)
        self.create_subscription(Bool, '/control/start_moving_along_y', self.move_along_y_axis_callback, 10)
        self.create_subscription(Float32, '/control/depth_at_centroid', self.depth_at_centroid_callback, 100)
        self.create_subscription(String, '/control/obj_to_reach', self.obj_to_reach_callback, 10)
        self.create_subscription(Point, '/control/bounding_box_center', self.bounding_box_center_callback, 10) # From control_server

        self.stop_execution_pub = self.create_publisher(Bool, '/control/stop_execution', 30)
        self.pointcloud_pub = self.create_publisher(Bool, '/control/start_pointcloud', 10)
        self.template_matching_pub = self.create_publisher(Bool, '/control/start_template_matching', 10)

        self.current_joint_states = None
        self.initial_distance_y = None
        self.depth_adjustment = None  # Initialize the depth adjustment variable
        self.previous_depth_adjustment = None  # Initialize the previous depth adjustment variable
        self.bbox_center = None  # Initialize bounding box center
        self.bbox_center_template = None  # Initialize bounding box center template
        self.bbox_center_lock = threading.Lock()  # Lock for bbox_center
        self.alignment_ok_event = Event()  # Initialize alignment event
        self.bbox_center_event = Event()  # Event to wait for bbox_center
        self.alignment_within_threshold_event = Event()  # Event to wait for alignment within threshold
        self.start_move_along_y = None
        self.difference = None
        self.payment_depth = None
        self.x_position_pos = None
        self.start_getting_areas = False

        self.first_alignment = True
        self.previous_error_x = 0
        self.stop_pos_movement = False
        self.phase = 0  # Starting with the first phase
        self.depth_at_centroid = None
        self.failed_planning = False
        self.failed = False

        self.pointcloud_pub.publish(Bool(data=True))

        self.alignment_bbox_timer = self.create_timer(1.5, self.align_with_bbox_logo_callback) # it was 2.0

    def bounding_box_center_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            return

        with self.bbox_center_lock:
            self.bbox_center = (msg.x, msg.y)
        self.bbox_center_event.set()

    def obj_to_reach_callback(self, msg):
        self.obj_to_reach = msg.data
        self.get_logger().info(f"Received obj_to_reach: {self.obj_to_reach}")

    def depth_at_centroid_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.depth_at_centroid = msg.data

    def plan_down_to_logo(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving camera down to the logo")

            try:
                planning_scene_monitor = self.lite6.get_planning_scene_monitor()
                robot_state = RobotState(self.lite6.get_robot_model())

                # Use a context manager to handle locking automatically
                with planning_scene_monitor.read_write() as scene:
                    robot_state = scene.current_state
                    camera_pose = robot_state.get_pose("camera_color_optical_frame")
                    ee_pose = robot_state.get_pose("link_tcp")
                    robot_state.update()

                    # Create and add the collision object using the external function
                    collision_object = self.create_collision_object(
                        frame_id="world",
                        object_id="POS",
                        position=[ee_pose.position.x, camera_pose.position.y, 0.0],#camera_pose.position.z - self.depth_at_centroid],   ee_pose.position.x had + 0.05
                        dimensions=[0.1, 0.1, camera_pose.position.z - self.depth_at_centroid]
                    )

                    scene.apply_collision_object(collision_object)
                    scene.current_state.update()
                    
                time.sleep(0.5)
            except Exception as e:
                self.get_logger().error(f"Exception in plan_down_to_logo: {str(e)}")


    def go_down_to_logo(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving camera down to the logo")
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene: ############ BEFORE IT WAS READ AND WRITE
                robot_state = scene.current_state
                camera_pose = robot_state.get_pose("camera_color_optical_frame")

                # Calculate the target depth
                self.payment_depth = 3 * (camera_pose.position.z - self.depth_at_centroid) / 4

                pose_goal = Pose()
                pose_goal.position.x = camera_pose.position.x - 0.11
                pose_goal.position.y = camera_pose.position.y - self.difference
                pose_goal.position.z = max(camera_pose.position.z - self.depth_at_centroid + self.payment_depth, 0.15)  # Increased minimum z to 0.15
                #pose_goal.position.z = camera_pose.position.z - self.depth_at_centroid #- 0.01

                # Log the calculated positions
                self.get_logger().info(f"Depth at centroid: {self.depth_at_centroid}")
                self.get_logger().info(f"Target position z: {pose_goal.position.z}")

                if (camera_pose.position.z - self.depth_at_centroid - 0.01) <= 0.15:
                    self.get_logger().warn("Target position z is too low, adjusting to safe value.")

                pose_goal.orientation.x = 0.0
                pose_goal.orientation.y = 0.7
                pose_goal.orientation.z = 0.0
                pose_goal.orientation.w = 0.7

                robot_collision_status = True
                attempts = 0

                while robot_collision_status and attempts < 20:  # Limit the number of attempts to avoid infinite loops
                    original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")

                    robot_state.set_to_random_positions()
                    robot_state.update()

                    result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=5.0)

                    robot_state.update(True)

                    robot_collision_status = scene.is_state_colliding(
                        robot_state=robot_state, joint_model_group_name="lite6_arm", verbose=True
                    )

                    self.get_logger().info(f"Robot is in collision: {robot_collision_status}")


                    collision_request = CollisionRequest()
                    collision_result = CollisionResult()
                    collision_request.contacts = True
                    #self.lite6.get_planning_component("lite6_arm").get_planning_scene().check_self_collision(collision_request, collision_result)
                    planning_scene = PlanningScene(self.lite6.get_robot_model())
                    planning_scene.check_self_collision(collision_request, collision_result, robot_state)
                    if collision_result.collision:
                        self.get_logger().info("Self-collision detected")
                        # Handle the collision scenario, e.g., stopping the robot
                    else:
                        self.get_logger().info("No self-collision detected")



                    if result and not robot_collision_status and not collision_result.collision:  # If a valid and collision-free IK solution is found
                        plan = True

                        self.lite6_arm.set_goal_state(robot_state=robot_state)
                        robot_state.update()
                        robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                        robot_state.update()
                        break
                    else:
                        self._logger.error("IK solution was not found!")
                        # Adjust the goal slightly to attempt a new solution
                        print('Pose goal position x BEFORE:', pose_goal.position.x)
                        pose_goal.position.x -= 0.01 # ha fatto effetto a quanto pare!
                        #pose_goal.position.z += 0.005  # Adjust the z position slightly upwards
                        #print('Pose goal position x AFTER:', pose_goal.position.x)
                        self.failed = True
                        robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                        robot_state.update(True)
                        attempts += 1
                        #break  ############################## ADDED THIS BREAK -------------->> PUT IT BACK TO MAKE IT WORK
                        

                    time.sleep(0.5)

                if not result or robot_collision_status:
                    self.get_logger().error("Unable to find a collision-free IK solution after several attempts.")
                    return

            if plan:
                multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
                    self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
                )
                self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, multi_plan_parameters=multi_pipeline_plan_request_params)

    def final_go_down_to_logo(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving camera down to the logo")
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene: ############ BEFORE IT WAS READONLY
                robot_state = scene.current_state
                camera_pose = robot_state.get_pose("camera_color_optical_frame")

                pose_goal = Pose()
                pose_goal.position.x = camera_pose.position.x #- np.random.uniform(-0.01, 0.01)
                pose_goal.position.y = camera_pose.position.y - 0.01
                pose_goal.position.z = camera_pose.position.z - self.payment_depth #- 0.005 # Increased minimum z to 0.15
                #pose_goal.position.z = camera_pose.position.z - self.depth_at_centroid #- 0.01

                pose_goal.orientation = camera_pose.orientation

                #pose_goal.orientation.x = 0.0
                #pose_goal.orientation.y = 0.7
                #pose_goal.orientation.z = 0.0
                #pose_goal.orientation.w = 0.7

                robot_collision_status = True
                attempts = 0

                while robot_collision_status and attempts < 20:  # Limit the number of attempts to avoid infinite loops
                    original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")

                    robot_state = scene.current_state
                    robot_state.update()

                    result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=15.0)

                    robot_state.update()

                    robot_collision_status = scene.is_state_colliding(
                        robot_state=robot_state, joint_model_group_name="lite6_arm", verbose=True
                    )

                    self.get_logger().info(f"Robot is in collision: {robot_collision_status}")


                    collision_request = CollisionRequest()
                    collision_result = CollisionResult()
                    collision_request.contacts = True
                    #self.lite6.get_planning_component("lite6_arm").get_planning_scene().check_self_collision(collision_request, collision_result)
                    planning_scene = PlanningScene(self.lite6.get_robot_model())
                    planning_scene.check_self_collision(collision_request, collision_result, robot_state)
                    if collision_result.collision:
                        self.get_logger().info("Self-collision detected")
                        # Handle the collision scenario, e.g., stopping the robot
                    else:
                        self.get_logger().info("No self-collision detected")



                    if result and not robot_collision_status and not collision_result.collision:  # If a valid and collision-free IK solution is found
                        plan = True

                        self.lite6_arm.set_goal_state(robot_state=robot_state)
                        robot_state.update()
                        robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                        robot_state.update()
                        break
                    else:
                        self._logger.error("IK solution was not found!")
                        # Adjust the goal slightly to attempt a new solution
                        #print('Pose goal position x BEFORE:', pose_goal.position.x)
                        pose_goal.position.x -= 0.01 # ha fatto effetto a quanto pare!
                        #pose_goal.position.z += 0.005  # Adjust the z position slightly upwards
                        #print('Pose goal position x AFTER:', pose_goal.position.x)
                        #self.failed = True
                        #robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                        #robot_state.update(True)
                        attempts += 1
                        #break  ############################## ADDED THIS BREAK -------------->> PUT IT BACK TO MAKE IT WORK
                        

                    time.sleep(0.5)

                if not result or robot_collision_status:
                    self.get_logger().error("Unable to find a collision-free IK solution after several attempts.")
                    return

            if plan:

                constraints = Constraints()
                constraints.name = "joints_constraints"


                joint_3_constraint = JointConstraint()
                joint_3_constraint.joint_name = "joint3"
                joint_3_constraint.position = original_joint_positions[2]
                joint_3_constraint.tolerance_above = 2.5   # Upper limit = Desired value + Tolerance above, this is the one that prevents the collision between link 3 aand robot base
                joint_3_constraint.tolerance_below = 2.5   # Lower limit = Desired value − Tolerance below
                joint_3_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_3_constraint)

                joint_4_constraint = JointConstraint()
                joint_4_constraint.joint_name = "joint4"
                joint_4_constraint.position = original_joint_positions[3]
                joint_4_constraint.tolerance_above = 1.5   # Upper limit = Desired value + Tolerance above, this is the one that prevents the collision between link 3 aand robot base
                joint_4_constraint.tolerance_below = 1.5   # Lower limit = Desired value − Tolerance below
                joint_4_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_4_constraint)


                joint_5_constraint = JointConstraint()
                joint_5_constraint.joint_name = "joint5"
                joint_5_constraint.position = original_joint_positions[4]
                joint_5_constraint.tolerance_above = 1.5   # Upper limit = Desired value + Tolerance above, this is the one that prevents the collision between link 3 aand robot base
                joint_5_constraint.tolerance_below = 1.5   # Lower limit = Desired value − Tolerance below
                joint_5_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_5_constraint)


                joint_6_constraint = JointConstraint()
                joint_6_constraint.joint_name = "joint6"
                joint_6_constraint.position = original_joint_positions[5]
                joint_6_constraint.tolerance_above = 1.5   # Upper limit = Desired value + Tolerance above, this is the one that prevents the collision between link 3 aand robot base
                joint_6_constraint.tolerance_below = 1.5   # Lower limit = Desired value − Tolerance below
                joint_6_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_6_constraint)



                multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
                    self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
                )
                self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, constraints=constraints, multi_plan_parameters=multi_pipeline_plan_request_params)



    def create_collision_object(self, frame_id="world", object_id="table", position=None, dimensions=None):
        """
        Creates a collision object for the planning scene.

        :param frame_id: The reference frame for the object (e.g., "world").
        :param object_id: A unique ID for the collision object.
        :param position: A list or tuple [x, y, z] for the object's position.
        :param dimensions: A list or tuple [x, y, z] for the object's dimensions (e.g., [0.5, 0.5, 0.05]).
        :return: CollisionObject ready to be added to the planning scene.
        """
        if position is None:
            position = [0.1, - 0.1, 0.6]
        if dimensions is None:
            dimensions = [0.1, 0.1, 0.05]

        # Define the collision object
        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.header.frame_id = frame_id

        # Define the object's shape and dimensions
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = dimensions

        # Define the pose of the collision object
        object_pose = Pose()
        object_pose.position.x = position[0]
        object_pose.position.y = position[1]
        object_pose.position.z = position[2]
        object_pose.orientation.w = 1.0  # Identity quaternion

        # Assign the shape and pose to the collision object
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(object_pose)
        collision_object.operation = CollisionObject.ADD

        return collision_object
    
    def pay_moving_down_card(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            print('                    ')
            self.get_logger().info("PAYMENT MOVEMENT FAILED")
            print('                    ')
            #time.sleep(5.0)
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene:
                robot_state = scene.current_state
                ee_pose = robot_state.get_pose("link_tcp")
                camera_pose = robot_state.get_pose("camera_color_optical_frame")

                pose_goal = Pose()
                pose_goal.position.x = camera_pose.position.x  # the first 0.05 is half the distance of the POS
                pose_goal.position.y = camera_pose.position.y - 0.02
                #pose_goal.position.z = ee_pose.position.z - self.payment_depth - 0.015 # offset to be sure that the card is on the POS
                pose_goal.position.z = camera_pose.position.z #- 0.05

                pose_goal.orientation.x = 0.0
                pose_goal.orientation.y = 0.7
                pose_goal.orientation.z = 0.0
                pose_goal.orientation.w = 0.7

                original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=5.0)

                robot_state.update()

                if not result:
                    self._logger.error("IK solution was not found!")
                    #rclpy.shutdown()
                    return
                else:
                    plan = True
                    self.lite6_arm.set_goal_state(robot_state=robot_state)
                    robot_state.update()
                    robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                    robot_state.update()
            if plan:
                multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
                    self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
                )
                self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, multi_plan_parameters=multi_pipeline_plan_request_params)


    def pay_moving_up_card(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving Card Up! Payment done")
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene:
                robot_state = scene.current_state
                ee_pose = robot_state.get_pose("link_tcp")
                #camera_pose = robot_state.get_pose("camera_color_optical_frame")

                pose_goal = Pose()
                pose_goal.position.x = ee_pose.position.x
                pose_goal.position.y = ee_pose.position.y
                pose_goal.position.z = ee_pose.position.z + self.payment_depth + 0.01 # offset to be sure that the card is on the POS

                pose_goal.orientation = ee_pose.orientation

                original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=5.0)

                robot_state.update()

                if not result:
                    self._logger.error("IK solution was not found!")
                    #rclpy.shutdown()
                    return
                else:
                    plan = True
                    self.lite6_arm.set_goal_state(robot_state=robot_state)
                    robot_state.update()
                    robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                    robot_state.update()
            if plan:
                multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
                    self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
                )
                self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, multi_plan_parameters=multi_pipeline_plan_request_params)


    def move_ee_to_camera_pos(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving EE to camera position")
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene:
                robot_state = scene.current_state
                ee_pose = robot_state.get_pose("link_tcp")
                camera_pose = robot_state.get_pose("camera_color_optical_frame")

                pose_goal = Pose()
                pose_goal.position.x = ee_pose.position.x
                pose_goal.position.y = ee_pose.position.y
                pose_goal.position.z = camera_pose.position.z

                self.difference = abs(camera_pose.position.y) - abs(ee_pose.position.y)

                pose_goal.orientation = ee_pose.orientation

                original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=5.0)

                robot_state.update()

                if not result:
                    self._logger.error("IK solution was not found!")
                    #rclpy.shutdown()
                    return
                else:
                    plan = True
                    self.lite6_arm.set_goal_state(robot_state=robot_state)
                    robot_state.update()
                    robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                    robot_state.update()
            if plan:
                multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
                    self.lite6, ["ompl_rrtc", "pilz_lin", "chomp_b", "ompl_rrt_star", "stomp_b"]
                )
                self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5, multi_plan_parameters=multi_pipeline_plan_request_params)

    def move_along_y_axis_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.start_move_along_y = msg.data

    def depth_adjustment_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.previous_depth_adjustment = self.depth_adjustment  
            self.depth_adjustment = msg.data
            self.get_logger().info(f'Received depth adjustment: {self.depth_adjustment} meters')

    def initial_distance_y_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.initial_distance_y = msg.data
            self.get_logger().info(f'Initial distance y: {self.initial_distance_y}')

    def bbox_center_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            with self.bbox_center_lock:
                    self.bbox_center_template = msg.data
            self.get_logger().info(f'Received bounding box center template: {self.bbox_center_template}')

    def alignment_status_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info(f'Received alignment status: {msg.data}')
            if msg.data:
                self.alignment_ok_event.set()
                self.get_logger().info("Alignment OK event set")
                if self.first_alignment:
                    self.first_alignment = False
            else:
                self.alignment_ok_event.clear()
                self.get_logger().info("Alignment OK event cleared")

    def stop_pos_movement_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.stop_pos_movement = msg.data
            if self.stop_pos_movement:
                self.get_logger().info("Received stop POS movement signal.")

    def bbox_area_callback(self, msg):
        if self.obj_to_reach == 'CreditCard':
            return

        if self.current_joint_states and self.start_getting_areas:
            joint_positions = list(self.current_joint_states.position)
            bbox_area = msg.data
            self.align_positions.append((joint_positions, bbox_area))
            self.get_logger().info(f'Added configuration with bbox area: {bbox_area}')

            if len(self.align_positions) >= 3:
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
            self.phase = 1
            #self.template_matching_pub.publish(Bool(data=True))

            #self.move_along_y_axis()

    def align_with_bbox_logo_callback(self):
        print('Phase:', self.phase)
        if self.obj_to_reach == 'CreditCard':
            return
        
        self.start_getting_areas = True

        with self.bbox_center_lock:
            bbox_center = self.bbox_center
            bbox_center_template = self.bbox_center_template

        if bbox_center or self.phase == 2:
            frame_center_x = 320
            frame_center_y = 130 # era 240

            if self.phase == 1:
                # Align with POS bbox
                print('Inside Phase 1')
                #frame_center_x = 640
                #frame_center_y = 360
                self.align_with_pos_bbox(frame_center_x=640, frame_center_y=360, bbox_center=bbox_center)
            elif self.phase == 2:
                # Move along -y axis until y alignment
                print('Inside Phase 2')
                self.align_y_axis(frame_center_y=200, bbox_center=bbox_center)
            elif self.phase == 3:
                # Align x axis with template matching bbox
                self.align_x_axis(frame_center_x=260, bbox_center=bbox_center_template)
                #self.align_y_axis(frame_center_y=240, bbox_center_template=bbox_center_template) # ADDED THIS, I DON'T KNOW IF IT WILL WORK
                self.pointcloud_pub.publish(Bool(data=True))
                self.move_ee_to_camera_pos()
                time.sleep(0.5)
                self.plan_down_to_logo()
                time.sleep(0.5)
                #self.move_to_ready_position("Ready")
                #time.sleep(1.5)
                self.go_down_to_logo() # The problem lies here!
                time.sleep(0.5)
                #if self.failed:
                #    self.get_logger().info("Failed to find a collision-free IK solution, stopping and shutting down")
                #    self.pay_moving_down_card()
                self.final_go_down_to_logo()
                #while self.failed_planning:
                #    self.get_logger().info("Failed to find a collision-free IK solution")
                    #self.pay_moving_down_card()
                    #time.sleep(0.5)
                    #self.final_go_down_to_logo()
                time.sleep(0.5)
                self.pay_moving_up_card()
                time.sleep(1.5)
                #self.move_to_ready_position("Ready")
                #time.sleep(0.5)
                print(" MOVE TO CAMERASEATCHING POSITION")
                #self.move_to_ready_position("CameraSearching")
                #time.sleep(1.5)
                rclpy.shutdown()

    def move_to_ready_position(self, position_name): ####
        self.get_logger().info("Moving EE to Ready position")

        self.lite6_arm.set_start_state_to_current_state()
        self.lite6_arm.set_goal_state(configuration_name=position_name)  ####
        plan_result = self.lite6_arm.plan()

        if plan_result:
            robot_trajectory = plan_result.trajectory
            self.lite6.execute(robot_trajectory, controllers=[])

    def align_with_pos_bbox(self, frame_center_x, frame_center_y, bbox_center):
        self.get_logger().info("Aligning with POS bbox")
        bbox_center_x = bbox_center[0]
        bbox_center_y = bbox_center[1]
        error_x = frame_center_x - bbox_center_x
        error_y = frame_center_y - bbox_center_y

        if abs(error_x) > 50 :
            self.adjust_robot_position_xy(error_x, 0)
            self.get_logger().info(f"Error in X axis: {error_x}, Error in Y axis: {error_y}, Adjusting position for alignment")
        else:
            self.get_logger().info("POS bbox alignment within threshold, moving to phase 2")
            self.phase = 2
            self.template_matching_pub.publish(Bool(data=True))




    def align_y_axis(self, frame_center_y, bbox_center):
        self.get_logger().info("Moving along -y axis until y alignment with template matching")
        if bbox_center and self.bbox_center_template is not None:
            bbox_center_y = self.bbox_center_template[1]
            error_y = frame_center_y - bbox_center_y

            if abs(error_y) > 15:   # it was 5
                self.adjust_robot_position_xy(0, error_y)
                self.get_logger().info(f"Error in Y axis: {error_y}, Adjusting position for alignment")
            else:
                self.get_logger().info("Y axis alignment within threshold, moving to phase 3")
                self.phase = 3
        else:
            self.get_logger().info("No bounding box center received")
            self.move_along_y_axis()

    def align_x_axis(self, frame_center_x, bbox_center):
        self.get_logger().info("Aligning x axis with template matching bbox")
        bbox_center_x = bbox_center[0]
        error_x = frame_center_x - bbox_center_x

        if abs(error_x) > 15: # IT WAS 10
            self.adjust_robot_position_xy(error_x, 0)
            self.get_logger().info(f"Error in X axis: {error_x}, Adjusting position for alignment")
        else:
            self.get_logger().info("X axis alignment within threshold, stopping and shutting down")

    def stop_and_shutdown(self):
        self.get_logger().info("Stopping and shutting down")
        self.stop_pos_movement = True
        self.destroy_node()
        rclpy.shutdown()

    def adjust_robot_position_xy(self, error_x, error_y):
        if self.stop_pos_movement:
            self.get_logger().info("Adjust movement halted due to stop signal.")
            return

        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame")

            pose_goal = Pose()
            Kp = 0.0001  # Proportional gain for adjustment

            adjustment_x = Kp * error_x
            adjustment_y = Kp * error_y

            if self.phase == 1:
                Kp = 0.00001
                pose_goal.position.x = camera_pose.position.x + np.clip(adjustment_x, -0.005, 0.005)
                pose_goal.position.y = camera_pose.position.y 
                pose_goal.position.z = camera_pose.position.z
            elif self.phase == 2:
                pose_goal.position.x = camera_pose.position.x + adjustment_x
                pose_goal.position.y = camera_pose.position.y - adjustment_y # it was minus 
                pose_goal.position.z = camera_pose.position.z

            self.previous_error_x = error_x

            pose_goal.orientation.x = camera_pose.orientation.x
            pose_goal.orientation.y = camera_pose.orientation.y
            pose_goal.orientation.z = camera_pose.orientation.z
            pose_goal.orientation.w = camera_pose.orientation.w

            print('Alignment adjustment on x to be made', adjustment_x)
            print('Alignment adjustment on y to be made', adjustment_y)

            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=2.0)

            robot_state.update()

            if not result:
                self.get_logger().error("IK solution was not found!")
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()

        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self.get_logger(), sleep_time=0.5)

    def move_along_y_axis(self):
        if self.obj_to_reach == 'CreditCard':
            self.get_logger().info("In IDLE state, not doing anything.")
            return
        elif self.obj_to_reach == 'POS':
            self.get_logger().info("Moving along Y axis")
            planning_scene_monitor = self.lite6.get_planning_scene_monitor()
            robot_state = RobotState(self.lite6.get_robot_model())

            with planning_scene_monitor.read_write() as scene:
                robot_state = scene.current_state
                camera_pose = robot_state.get_pose("camera_color_optical_frame")

                pose_goal = Pose()
                pose_goal.position.x = camera_pose.position.x
                pose_goal.position.y = camera_pose.position.y - 0.025  # Adjust as necessary, CHANGED , TEST IT
                pose_goal.position.z = camera_pose.position.z

                pose_goal.orientation.x = camera_pose.orientation.x
                pose_goal.orientation.y = camera_pose.orientation.y
                pose_goal.orientation.z = camera_pose.orientation.z
                pose_goal.orientation.w = camera_pose.orientation.w

                robot_state.update()

                original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
                result = robot_state.set_from_ik("lite6_arm", pose_goal, "camera_color_optical_frame", timeout=1.0)

                robot_state.update()

                if not result:
                    self.get_logger().error("IK solution was not found!")
                    return
                else:
                    plan = True
                    self.lite6_arm.set_goal_state(robot_state=robot_state)
                    robot_state.update()
                    robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                    robot_state.update()

            if plan:
                self.plan_and_execute(self.lite6, self.lite6_arm, self.get_logger(), sleep_time=0.5)

    def plan_and_execute(self, robot, planning_component, logger, sleep_time,constraints=None, single_plan_parameters=None, multi_plan_parameters=None):
        if self.stop_pos_movement:
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
            if self.stop_pos_movement:
                logger.info("Execution halted due to stop signal after planning.")
                return
            self.failed_planning = False
            logger.info("Executing plan")
            robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=[])
        else:
            logger.error("Planning failed")
            self.failed_planning = True
            #time.sleep(0.5)

        time.sleep(sleep_time)

def main(args=None):
    rclpy.init(args=args)
    pos_movement = PosMovement()
    #pos_movement.move_down_to_logo()
    #pos_movement.create_collision_object(frame_id=str("world"), object_id=str("POS"))

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(pos_movement)

    try:
        executor.spin()
    finally:
        pos_movement.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()