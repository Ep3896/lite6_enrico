#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point, PointStamped, Pose
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from lite6_enrico_interfaces.action import GoToPose
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs.tf2_geometry_msgs
import math
import time

# Thresholds
Z_THRESHOLD = 0.0013  # Meters

class CameraControllerNode(Node):

    def __init__(self):
        super().__init__("camera_control_node")

        # Subscriber for desired goal
        self.create_subscription(MarkerArray, '/yolo/dgb_bb_markers', self.detection_callback, 30)

        # Publisher for rotation flag
        self.rotation_flag_publisher = self.create_publisher(Bool, 'rotation_flag', 10)

        # Action client for sending goal to the action server
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_position = Pose()

        # Timer to send goals periodically
        self.goal_timer = self.create_timer(0.033, self.timer_callback)  # Adjust the interval as needed

        # Variables for control loop
        self.updated_camera_position = None

        # Shutdown flag
        self.shutdown_flag = False

        # Error Z buffer
        self.error_z_buffer = []

    def detection_callback(self, msg: MarkerArray):
        if msg.markers:
            marker = msg.markers[0]  # Process the first marker, or modify as needed
            bbox_center = PointStamped()
            bbox_center.header.frame_id = 'camera_depth_frame'
            bbox_center.header.stamp = self.get_clock().now().to_msg()
            bbox_center.point = marker.pose.position

            # Transform the point to the world frame
            try:
                transform = self.tf_buffer.lookup_transform('world', 'camera_depth_frame', rclpy.time.Time(), rclpy.duration.Duration(seconds=0.0))
                world_point = tf2_geometry_msgs.do_transform_point(bbox_center, transform).point

                # Update the target position for the timer callback
                self.target_position.position = world_point

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().error(f'Could not transform point: {e}')

    def timer_callback(self):
        if self.target_position.position:
            self.send_goal(self.target_position.position)

    def send_goal(self, target_position: Point):
        goal_msg = GoToPose.Goal()
        goal_msg.pose.position = target_position

        goal_msg.pose.orientation.x = 0.0  # Assuming a default orientation
        goal_msg.pose.orientation.y = 0.7  # Assuming a default orientation
        goal_msg.pose.orientation.z = 0.0  # Assuming a default orientation
        goal_msg.pose.orientation.w = 0.7  # Assuming a default orientation

        self._action_client.wait_for_server(timeout_sec=0.01)  # Wait for the server to be up
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
        print(f'Result: {result.success, result.updated_camera_position, result.updated_camera_position.position.z}')
        if result.success:
            print('Goal succeeded!')
            self.updated_camera_position = result.updated_camera_position.position
            print("          ")
            print("          ")
            print("Updated camera position:", self.updated_camera_position)

            # Calculate error_z
            error_z = self.updated_camera_position.z - self.target_position.position.z
            print("\033[91m" + "Error: ", error_z)

            # Update the error_z_buffer
            self.update_error_z_buffer(error_z)

            # Check if the buffer is full and compute the mean
            if len(self.error_z_buffer) == 40:
                mean_error_z = sum(self.error_z_buffer) / len(self.error_z_buffer)
                print("\033[93m" + "Mean Error Z: ", mean_error_z)
                if 0.1500 <= mean_error_z <= 0.1501:
                    self.get_logger().info("Stopping: Mean Error Z is within the threshold")
                    self.shutdown_flag = True

    def update_error_z_buffer(self, error_z):
        if len(self.error_z_buffer) >= 40:
            self.error_z_buffer.pop(0)
        self.error_z_buffer.append(error_z)

def main(args=None):
    rclpy.init(args=args)
    camera_controller_node = CameraControllerNode()

    try:
        while rclpy.ok() and not camera_controller_node.shutdown_flag:
            rclpy.spin_once(camera_controller_node)
    finally:
        camera_controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()