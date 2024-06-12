#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point, PointStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
from lite6_enrico_interfaces.action import GoToPose
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs.tf2_geometry_msgs

# Main Control Node
class CameraControlNode(Node):
    def __init__(self):
        super().__init__('camera_control_node')
        self.subscription = self.create_subscription(MarkerArray, '/yolo/dgb_bb_markers', self.detection_callback, 10)
        self.goal_publisher = self.create_publisher(Float32MultiArray, 'goal_coordinates', 10)
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Image plane parameters
        self.image_width = 1280  # Adjust based on your camera resolution
        self.image_height = 720 # Adjust based on your camera resolution
        self.center_x = self.image_width / 2
        self.center_y = self.image_height / 2

    def detection_callback(self, msg):
        if msg.markers:
            marker = msg.markers[0]  # Process the first marker, or modify as needed
            bbox_center = PointStamped()
            bbox_center.header.frame_id = 'camera_depth_frame'
            bbox_center.header.stamp = self.get_clock().now().to_msg()
            bbox_center.point = marker.pose.position

            # Transform the point to the world frame
            try:
                transform = self.tf_buffer.lookup_transform('world', 'camera_depth_frame', rclpy.time.Time(), rclpy.duration.Duration(seconds=1))
                world_point = tf2_geometry_msgs.do_transform_point(bbox_center, transform).point

                # Adjust the goal based on the centroid's position in the image plane
                adjustment_x, adjustment_y = self.calculate_adjustments(marker.pose.position)

                goal_x = world_point.x + adjustment_x
                goal_y = world_point.y + adjustment_y
                goal_z = world_point.z
                print(f'Goal: ({goal_x}, {goal_y}, {goal_z})')

                # Publish the goal coordinates
                goal_msg = Float32MultiArray()
                goal_msg.data = [goal_x, goal_y, goal_z]
                self.goal_publisher.publish(goal_msg)

                # Send the new goal to the action server
                self.send_goal(goal_x, goal_y, goal_z)
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().error(f'Could not transform point: {e}')

    def calculate_adjustments(self, position):
        # Calculate the error in the image plane
        error_x = position.x - self.center_x
        error_y = position.y - self.center_y

        # Define the adjustment scale (tune these values based on your system)
        adjustment_scale_x = 0.01
        adjustment_scale_y = 0.01

        # Calculate the adjustments
        adjustment_x = -adjustment_scale_x * error_x / self.image_width
        adjustment_y = -adjustment_scale_y * error_y / self.image_height

        return adjustment_x, adjustment_y

    def send_goal(self, x, y, z):
        goal_msg = GoToPose.Goal()
        goal_msg.pose.position.x = x 
        goal_msg.pose.position.y = y 
        goal_msg.pose.position.z = z 
        goal_msg.pose.orientation.w = 1.0  # Assuming a default orientation

        print("Waiting for server")
        self._action_client.wait_for_server(timeout_sec=1.5)
        print("Sending goal")
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
    node = CameraControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
