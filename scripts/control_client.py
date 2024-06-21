#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point, PointStamped, Pose
from std_msgs.msg import Bool
from lite6_enrico_interfaces.action import GoToPose
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs.tf2_geometry_msgs
import math
import time
from yolov8_msgs.msg import DetectionArray



# Thresholds
Z_THRESHOLD = 0.0013  # Meters

class ControllerNode(Node):

    def __init__(self):
        super().__init__("camera_control_node")

        # Subscribers
        self.create_subscription(MarkerArray, '/yolo/dgb_bb_markers', self.detection_callback_pos, 30)
        self.create_subscription(DetectionArray, '/yolo/detections_3d', self.detections_callback_card, 30)

        # Publisher for rotation flag
        self.rotation_flag_publisher = self.create_publisher(Bool, 'rotation_flag', 10)

        # Action client for sending goal to the action server
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')

        # Timer to send goals periodically
        self.goal_timer = self.create_timer(0.033, self.timer_callback)  # Adjust the interval as needed

        # Variables for control loop
        self.updated_camera_position = None

        # Error Z buffer
        self.error_z_buffer = []

        self.shutdown_flag = False

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_position = Pose()

        # Pick card flag
        self.pick_card = True


        # Desired center of the image
        self.desired_x = 700  
        self.desired_y = 425

        # PID control gains
        self.Kp = 0.005  # Proportional gain
        self.Kd = 0.005  # Derivative gain

        # Initialize previous errors
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0

        # Initialize previous goal position
        self.previous_goal = Point(x=0.20687, y=0.068421, z=0.21905)

        print("Controller node initialized")

    def detection_callback_pos(self, msg: MarkerArray):
        if msg.markers and not self.pick_card:
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

    def detections_callback_card(self, msg: DetectionArray):
        print("detection_callback_card")
        if self.pick_card:
            for detection in msg.detections:
                if detection.class_name == 'CreditCard':
                    bbox_center_x = detection.bbox.center.position.x
                    bbox_center_y = detection.bbox.center.position.y

                    print(f'Center of the bounding box: ({bbox_center_x}, {bbox_center_y})')

                    # Calculate the error in the image plane (in pixels)
                    error_x = self.desired_x - bbox_center_x
                    error_y = self.desired_y - bbox_center_y
                    

                    print(f'Error in X: {error_x}, Error in Y: {error_y}')

                    # Calculate proportional and derivative adjustments
                    adjustment_x = self.Kp * error_x + self.Kd * (error_x - self.previous_error_x)
                    adjustment_y = self.Kp * error_y + self.Kd * (error_y - self.previous_error_y)

                    print(f'Adjustment in X: {adjustment_x}, Adjustment in Y: {adjustment_y}')

                    # Update previous errors
                    self.previous_error_x = error_x
                    self.previous_error_y = error_y

                    proposed_goal_x = self.previous_goal.x + adjustment_x
                    proposed_goal_y = self.previous_goal.y + adjustment_y

                    # Update the target position for the timer callback
                    self.target_position.position.x = proposed_goal_x
                    self.target_position.position.y = proposed_goal_y
                    self.target_position.position.z = self.previous_goal.z - 0.1 # Assuming no change in Z for now

                    # Log the target position
                    self.get_logger().info(f'Goal position: {self.target_position.position}')

    def send_goal(self, target_position: Point):
        goal_msg = GoToPose.Goal()
        goal_msg.pose.position.x = self.target_position.position.x
        goal_msg.pose.position.y = self.target_position.position.y
        goal_msg.pose.position.z = self.target_position.position.z
        if not self.pick_card:
            goal_msg.pose.orientation.x = 0.0  # Assuming a default orientation
            goal_msg.pose.orientation.y = 0.7  # Assuming a default orientation
            goal_msg.pose.orientation.z = 0.0  # Assuming a default orientation
            goal_msg.pose.orientation.w = 0.7  # Assuming a default orientation
        else:
            goal_msg.pose.orientation.x = 0.69237
            goal_msg.pose.orientation.y = 0.30768
            goal_msg.pose.orientation.z = -0.55548
            goal_msg.pose.orientation.w = -0.34263

        self._action_client.wait_for_server()
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
        if result.success:
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().info('Goal failed!')
        
            
def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()

    try:
        while rclpy.ok() and not controller_node.shutdown_flag:
            rclpy.spin_once(controller_node)
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
