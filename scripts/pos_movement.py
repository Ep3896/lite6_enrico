import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Bool, Float32MultiArray, String
from sensor_msgs.msg import JointState, Image as msg_Image, CameraInfo
from threading import Event
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
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

class PosMovement(Node):

    def __init__(self):
        super().__init__('pos_movement')
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

        self.get_logger().info("PosMovement node has been started")

        self.create_subscription(Float32, '/control/bbox_area', self.bbox_area_callback, 10)
        self.create_subscription(JointState, '/control/joint_states', self.joint_states_callback, 10)
        self.create_subscription(Float32, '/control/initial_distance_y', self.initial_distance_y_callback, 10)
        self.create_subscription(Float32, '/control/depth_adjustment', self.depth_adjustment_callback, 100)
        self.create_subscription(Float32MultiArray, '/control/bbox_center', self.bbox_center_callback, 10)
        self.create_subscription(Bool, '/control/alignment_status', self.alignment_status_callback, 10)
        self.create_subscription(Bool, '/control/stop_pos_movement', self.stop_pos_movement_callback, 10)
        self.create_subscription(Bool, '/control/start_moving_along_y', self.move_along_y_axis_callback, 10)
        self.create_subscription(Float32, '/control/depth_at_centroid', self.depth_at_centroid_callback, 30)


        self.stop_execution_pub = self.create_publisher(Bool, '/control/stop_execution', 30)
        self.pointcloud_pub = self.create_publisher(Bool, '/control/start_pointcloud', 10)
        self.template_matching_pub = self.create_publisher(Bool, '/control/start_template_matching', 10)
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
        self.start_move_along_y = None
        self.difference = None

        self.first_alignment = True
        self.previous_error_x = 0
        self.stop_pos_movement = False
        self.phase = 1  # Starting with the first phase
        self.depth_at_centroid = None

        self.pointcloud_pub.publish(Bool(data=True))

        self.alignment_bbox_timer = self.create_timer(2.0, self.align_with_bbox_logo_callback)

    def depth_at_centroid_callback(self, msg):
        self.depth_at_centroid = msg.data

    def move_down_to_logo(self):
        self.get_logger().info("Moving camera down to the logo")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame")

            pose_goal = Pose()
            pose_goal.position.x = camera_pose.position.x - 0.1
            pose_goal.position.y = camera_pose.position.y - self.difference
            pose_goal.position.z = camera_pose.position.z - self.depth_at_centroid - 0.01    ### This can be retrieved by looking at the depth in the pixel that show the board

            print('Depth at centroid', self.depth_at_centroid)
            #if pose_goal.position.z < 0.1:
            #    pose_goal.position.z = 0.1

            #-1.7619e-06; 0.70712; -8.5189e-06; 0.70709
            pose_goal.orientation.x = 0.0
            pose_goal.orientation.y = 0.7
            pose_goal.orientation.z = 0.0
            pose_goal.orientation.w = 0.7

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

    def move_ee_to_camera_pos(self):
        self.get_logger().info("Moving EE to camera position")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            ee_pose = robot_state.get_pose("link_tcp")
            camera_pose = robot_state.get_pose("camera_color_optical_frame") #camera_color_optical_frame

            pose_goal = Pose()
            pose_goal.position.x = ee_pose.position.x   #camera_pose.position.x 
            pose_goal.position.y = ee_pose.position.y #camera_pose.position.y     #camera_pose.position.y
            pose_goal.position.z = camera_pose.position.z

            self.difference = abs(camera_pose.position.y) - abs(ee_pose.position.y)

            pose_goal.orientation = ee_pose.orientation

            original_joint_positions = robot_state.get_joint_group_positions("lite6_arm")
            result = robot_state.set_from_ik("lite6_arm", pose_goal, "link_tcp", timeout=5.0)

            robot_state.update()

            if not result:
                self._logger.error("IK solution was not found!")
                rclpy.shutdown()
                return
            else:
                plan = True
                self.lite6_arm.set_goal_state(robot_state=robot_state)
                robot_state.update()
                robot_state.set_joint_group_positions("lite6_arm", original_joint_positions)
                robot_state.update()
        if plan:
            self.plan_and_execute(self.lite6, self.lite6_arm, self._logger, sleep_time=0.5)


    def move_along_y_axis_callback(self, msg):
        self.start_move_along_y = msg.data

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
        self.bbox_center_event.set()
        self.get_logger().info(f'Received bounding box center: {self.bbox_center}')

    def alignment_status_callback(self, msg):
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
        self.stop_pos_movement = msg.data
        if self.stop_pos_movement:
            self.get_logger().info("Received stop POS movement signal.")

    def bbox_area_callback(self, msg):
        if self.current_joint_states:
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
            time.sleep(1.5)
            self.template_matching_pub.publish(Bool(data=True))
            

    def align_with_bbox_logo_callback(self):
        with self.bbox_center_lock:
            bbox_center = self.bbox_center

        if bbox_center:
            frame_center_x = 320
            frame_center_y = 240

            if self.phase == 1:
                # Align with POS bbox
                self.align_with_pos_bbox(frame_center_x, frame_center_y, bbox_center)
            elif self.phase == 2:
                # Move along -y axis until y alignment
                self.align_y_axis(frame_center_y, bbox_center)
            elif self.phase == 3:
                # Align x axis with template matching bbox
                self.align_x_axis(frame_center_x, bbox_center)
                self.pointcloud_pub.publish(Bool(data=True))
                self.move_ee_to_camera_pos()
                time.sleep(1.5)
                self.move_down_to_logo()
                rclpy.shutdown()

    def align_with_pos_bbox(self, frame_center_x, frame_center_y, bbox_center):
        self.get_logger().info("Aligning with POS bbox")
        bbox_center_x = bbox_center[0]
        bbox_center_y = bbox_center[1]
        error_x = frame_center_x - bbox_center_x
        error_y = frame_center_y - bbox_center_y

        if abs(error_x) > 20 or abs(error_y) > 20:
            self.adjust_robot_position_xy(error_x, error_y)
            self.get_logger().info(f"Error in X axis: {error_x}, Error in Y axis: {error_y}, Adjusting position for alignment")
        else:
            self.get_logger().info("POS bbox alignment within threshold, moving to phase 2")
            self.phase = 2
            self.template_matching_pub.publish(Bool(data=True))


################################
    def align_y_axis(self, frame_center_y, bbox_center):
        self.get_logger().info("Moving along -y axis until y alignment with template matching")
        bbox_center_y = bbox_center[1]
        error_y = frame_center_y - bbox_center_y

        if abs(error_y) > 10:
            self.adjust_robot_position_xy(0, error_y)
            self.get_logger().info(f"Error in Y axis: {error_y}, Adjusting position for alignment")
        else:
            self.get_logger().info("Y axis alignment within threshold, moving to phase 3")
            self.phase = 3

    def align_x_axis(self, frame_center_x, bbox_center):
        self.get_logger().info("Aligning x axis with template matching bbox")
        bbox_center_x = bbox_center[0]
        error_x = frame_center_x - bbox_center_x

        if abs(error_x) > 20:
            self.adjust_robot_position_xy(error_x, 0)
            self.get_logger().info(f"Error in X axis: {error_x}, Adjusting position for alignment")
        else:
            self.get_logger().info("X axis alignment within threshold, stopping and shutting down")
            #self.stop_and_shutdown()

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

            pose_goal.position.x = camera_pose.position.x + adjustment_x
            pose_goal.position.y = camera_pose.position.y - adjustment_y
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
        self.get_logger().info("Moving along Y axis")
        planning_scene_monitor = self.lite6.get_planning_scene_monitor()
        robot_state = RobotState(self.lite6.get_robot_model())

        with planning_scene_monitor.read_write() as scene:
            robot_state = scene.current_state
            camera_pose = robot_state.get_pose("camera_color_optical_frame")

            pose_goal = Pose()
            pose_goal.position.x = camera_pose.position.x
            pose_goal.position.y = camera_pose.position.y - 0.02  # Adjust as necessary
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

    def plan_and_execute(self, robot, planning_component, logger, sleep_time, single_plan_parameters=None, multi_plan_parameters=None):
        if self.stop_pos_movement:
            logger.info("Execution halted due to stop signal.")
            return

        logger.info("Planning trajectory")

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
            logger.info("Executing plan")
            robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=[])
        else:
            logger.error("Planning failed")
            time.sleep(0.5)

        time.sleep(sleep_time)

def main(args=None):
    rclpy.init(args=args)
    pos_movement = PosMovement()
    #pos_movement.move_down_to_logo()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(pos_movement)

    try:
        executor.spin()
    finally:
        pos_movement.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
