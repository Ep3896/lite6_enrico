#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point, PointStamped
from lite6_enrico_interfaces.action import GoToPose
from yolov8_msgs.msg import DetectionArray
from sensor_msgs.msg import Image as msg_Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs2
from simple_pid import PID
import tf2_ros
import tf2_geometry_msgs
import os
import sys

# PID gain values
KP = 0.0001
KI = 0.00005
KD = 0.0001

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node_card")

        # Subscribers
        self.create_subscription(DetectionArray, '/yolo/detections_3d', self.detections_callback, 30)
        self.create_subscription(msg_Image, '/camera/camera/depth/image_rect_raw', self.depth_image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.depth_info_callback, 10)

        # Action client for sending goal to the action server
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')

        # Timer to send goals periodically
        self.goal_timer = self.create_timer(0.033, self.timer_callback)

        self.new_target = False
        self.base_position = Point(x=0.30104, y=0.017488, z=0.44326)
        self.goal_handle = None

        # Variables for control loop
        self.intrinsics = None
        self.pix = None
        self.depth_image = None
        self.line = None
        self.pix_grade = None
        self.bounding_box_center = []
        self.target_position = Point(x=0.0, y=0.0, z=0.0)
        self.bridge = CvBridge()

        # Desired center of the image for picking mode
        self.desired_x = 640
        self.desired_y = 360

        # PID controllers for x and y
        self.pid_x = PID(KP, KI, KD, setpoint=self.desired_x)
        self.pid_y = PID(KP, KI, KD, setpoint=self.desired_y)

        # Set PID's update rate
        self.pid_x.sample_time = 0.033
        self.pid_y.sample_time = 0.033

        # Control clip value
        self.clip_val = 30.0

        self.shutdown_flag = False
        self.pick_card = False  # ---------------------------------------------------------------------------------> TO CHANGE BASED ON THE OBJECT TO PICK

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Controller node initialized")

    def detections_callback(self, msg: DetectionArray):
        for detection in msg.detections:
            if detection.class_name == 'POS':  # ---------------------------------------------------------------------------------> TO CHANGE BASED ON THE OBJECT TO PICK
                self.process_detection(detection)

    def process_detection(self, detection):
        bbox_center_x = detection.bbox.center.position.x
        bbox_center_y = detection.bbox.center.position.y

        self.bounding_box_center = [bbox_center_x, bbox_center_y]

        self.get_logger().info(f'Bounding Box Center: X: {bbox_center_x}, Y: {bbox_center_y}')

        # Update PID controllers with the current position
        error_x = self.pid_x(bbox_center_x)
        error_y = self.pid_y(bbox_center_y)

        # Clip adjustments to prevent excessive movements
        error_x = max(min(error_x, self.clip_val), -self.clip_val)
        error_y = max(min(error_y, self.clip_val), -self.clip_val)

        # Convert pixel error to camera frame using intrinsic parameters
        spatial_error_x = 1000 * error_x / self.intrinsics.fx if self.intrinsics else 0.0
        spatial_error_y = 1000 * error_y / self.intrinsics.fy if self.intrinsics else 0.0

        print(f'\n spatial_error_x: {spatial_error_x} \n spatial_error_y: {spatial_error_y} \n')

        # Create a point in the camera frame
        camera_error_point = PointStamped()
        camera_error_point.header.frame_id = 'camera_color_optical_frame'
        camera_error_point.header.stamp = self.get_clock().now().to_msg()
        camera_error_point.point.x = spatial_error_x
        camera_error_point.point.y = spatial_error_y
        camera_error_point.point.z = 0.0

        # Transform the point to the world frame
        try:
            transform = self.tf_buffer.lookup_transform('world', 'camera_color_optical_frame', rclpy.time.Time(), rclpy.duration.Duration(seconds=0.0))
            world_error_point = tf2_geometry_msgs.do_transform_point(camera_error_point, transform).point

            print(f'\n world_error_point: {world_error_point} \n')

            # Update the target position for the servoing ( THIS IS NOT A ROBUST SOLUTION, IT IS JUST USEFUL WHEN THE CAMERA POINTS DOWNWARDS)
            self.target_position.x = 0.578 - world_error_point.x  # this is not clear why it is mirrored but it was necessary
            self.target_position.y = -world_error_point.y  # y axis was inverted, need to check possible boundary problems

            # Use depth information for Z-axis adjustment if available
            if self.pix and self.depth_image is not None:
                depth_adjustment = self.depth_at_pixel(self.pix[0], self.pix[1])
                self.target_position.z = depth_adjustment / 1000.0  # Convert to meters

            self.get_logger().info(f'Target Position: X: {self.target_position.x}, Y: {self.target_position.y}, Z: {self.target_position.z}')

            if self.target_position.z > 0.2:
                self.send_goal(self.target_position)
            else:
                self.get_logger().info('Target position is too close, stopping movement.')
                self.stop_movement()

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 Error: {e}')

    def depth_at_pixel(self, x, y):
        if self.intrinsics and self.depth_image is not None:
            if 0 <= y < self.depth_image.shape[0] and 0 <= x < self.depth_image.shape[1]:  # Boundary check
                depth = self.depth_image[int(y), int(x)]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
                self.line += '  Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
                if self.pix_grade is not None:
                    self.line += ' Grade: %2d' % self.pix_grade
                self.line += '\r'
                sys.stdout.write(self.line)
                sys.stdout.flush()
                return result[2]  # Z-coordinate in camera space
        return self.target_position.z

    def depth_image_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            if self.bounding_box_center:
                self.pix = (int(self.bounding_box_center[0]), int(self.bounding_box_center[1]))
                if self.pix[0] < self.depth_image.shape[1] and self.pix[1] < self.depth_image.shape[0]:  # Boundary check
                    self.line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (self.pix[0], self.pix[1], self.depth_image[self.pix[1], self.pix[0]])
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        except ValueError as e:
            return

    def depth_info_callback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.k[2]
            self.intrinsics.ppy = cameraInfo.k[5]
            self.intrinsics.fx = cameraInfo.k[0]
            self.intrinsics.fy = cameraInfo.k[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.d]
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

    def timer_callback(self):
        if self.target_position:
            self.send_goal(self.target_position)

    def send_goal(self, target_position: Point):
        goal_msg = GoToPose.Goal()
        goal_msg.pose.position = target_position

        if not self.pick_card:
            # POS picking orientation for the POS (camera_color_optical_frame)
            goal_msg.pose.orientation.x = -0.70388
            goal_msg.pose.orientation.y = 0.70991
            goal_msg.pose.orientation.z = -0.015868
            goal_msg.pose.orientation.w = 0.018082
        else:
            # Credit Card picking orientation for the Credit Card (camera_color_optical_frame)
            goal_msg.pose.orientation.x = 0.64135
            goal_msg.pose.orientation.y = 0.6065
            goal_msg.pose.orientation.z = 0.3936
            goal_msg.pose.orientation.w = -0.25673

            # self.get_logger().info("Working with CreditCard")

        self._action_client.wait_for_server(timeout_sec=0.033)
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self.goal_handle = goal_handle  # Store the goal handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info('Goal succeeded!')
            self.new_target = True
            self.updated_camera_position = result.updated_camera_position.position

    def stop_movement(self):
        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if cancel_response:
            self.get_logger().info('Goal cancelled successfully')

def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()

    try:
        rclpy.spin(controller_node)
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
