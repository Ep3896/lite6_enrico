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
import numpy as np

# PID Controller Class
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

# Main Control Node
class CameraControlNode(Node):
    def __init__(self):
        super().__init__('camera_control_node')
        self.subscription = self.create_subscription(MarkerArray, '/yolo/dgb_bb_markers', self.detection_callback, 10)
        self.goal_publisher = self.create_publisher(Float32MultiArray, 'goal_coordinates', 10)
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')
        self.pid_x = PIDController(Kp=0.05, Ki=0.01, Kd=0.05)  # Depth (x)
        self.pid_y = PIDController(Kp=0.05, Ki=0.01, Kd=0.05)  # Horizontal (y)
        self.pid_z = PIDController(Kp=0.05, Ki=0.01, Kd=0.05)  # Vertical (z)
        self.previous_time = self.get_clock().now()

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

                current_time = self.get_clock().now()
                dt = (current_time - self.previous_time).nanoseconds / 1e9
                self.previous_time = current_time

                error_x = world_point.x
                error_y = world_point.y
                error_z = world_point.z

                control_signal_x = self.pid_x.update(error_x, dt)
                control_signal_y = self.pid_y.update(error_y, dt)
                control_signal_z = self.pid_z.update(error_z, dt)

                # Use the control signals to determine the new goal position
                goal_x = world_point.x + control_signal_x
                goal_y = world_point.y + control_signal_y
                goal_z = world_point.z + control_signal_z
                print(f'Goal: ({goal_x}, {goal_y}, {goal_z})')

                # Publish the goal coordinates
                goal_msg = Float32MultiArray()
                goal_msg.data = [goal_x, goal_y, goal_z]
                self.goal_publisher.publish(goal_msg)

                # Send the new goal to the action server
                self.send_goal(goal_x, goal_y, goal_z)
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().error(f'Could not transform point: {e}')

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
