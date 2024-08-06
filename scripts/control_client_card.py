import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float32, Bool, String
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
import sys
from collections import deque
import curses
import cv2
import os

# PID gain values
KP = 0.00005
KI = 0.0001
KD = 0.001
# Minimum distance to the object to be reached
MIN_Z_DISTANCE = 0.1
BUFFER_SIZE = 5

class ControllerNode(Node):

    def __init__(self, stdscr):
        super().__init__("controller_node_card")
        self.stdscr = stdscr
        curses.curs_set(0)
        self.stdscr.nodelay(1)

        # Initialize subscribers
        self.init_subscribers()

        # Initialize publishers
        self.init_publishers()

        # Initialize action client
        self._action_client = ActionClient(self, GoToPose, 'go_to_pose')

        # Timer to send goals periodically
        #self.goal_timer = self.create_timer(0.033, self.timer_callback)

        self.new_target = False
        self.base_position = Point(x=0.30104, y=0.017488, z=0.44326)
        self.goal_handle = None

        self.first_movement = True

        # Variables for control loop
        self.initialize_control_variables()

        self.print_info = {
            "bounding_box_center_x": 0,
            "bounding_box_center_y": 0,
            "spatial_error_x": 0.0,
            "spatial_error_y": 0.0,
            "world_error_point_x": 0.0,
            "world_error_point_y": 0.0,
            "world_error_point_z": 0.0,
            "depth_adjustment": 0.0,
            "target_position_x": 0.0,
            "target_position_y": 0.0,
            "target_position_z": 0.0,
            "depth_info": "",
            "distance": 0.0,
        }

        self.get_logger().info("Controller node initialized")

    def init_subscribers(self):
        self.create_subscription(DetectionArray, '/yolo/detections_3d', self.detections_callback, 30)
        self.create_subscription(msg_Image, '/camera/camera/depth/image_rect_raw', self.depth_image_callback, 100) #it was 10 before
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.depth_info_callback, 10) #it was 10 before
        self.create_subscription(Bool, '/control/first_movement', self.first_movement_callback, 10)
        self.create_subscription(String,'/control/obj_to_reach', self.obj_to_reach_callback, 10)

    def init_publishers(self):
        self.depth_adjustment_pub = self.create_publisher(Float32, '/control/depth_adjustment', 100)
        self.bbox_area_pub = self.create_publisher(Float32, '/control/bbox_area', 10)
        self.bbox_area_old = 0
        self.searching_card_pub = self.create_publisher(Bool, '/control/searching_card', 100)


    def initialize_control_variables(self):
        self.intrinsics = None
        self.pix = None
        self.depth_image = None
        self.line_coordinate = None
        self.line_depth_at_pixel = None
        self.pix_grade = None
        self.bounding_box_center = []
        self.bbox_area = []
        self.target_position = Point()
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
        self.clip_val = 10

        self.shutdown_flag = False
        self.pick_card = True  # Starts with Credit Card

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Depth buffer
        self.depth_buffer = deque(maxlen=BUFFER_SIZE)

    def obj_to_reach_callback(self, msg: String):
        if msg.data == 'CreditCard':
            self.pick_card = True
        elif msg.data == 'POS':
            self.pick_card = False
        else:
            self.get_logger().info('Invalid object to reach.')

    """
    def detections_callback(self, msg: DetectionArray):
        card_detected = False
        pos_detected = False
        for detection in msg.detections:
            if detection.class_name == 'CreditCard' and self.pick_card:  # Change based on the object to pick
                card_detected = True
                self.process_detection(detection)
            elif detection.class_name == 'POS' and not self.pick_card:  # Change based on the object to pick
                pos_detected = True
                self.process_detection(detection)
        if not card_detected and self.pick_card:
            self.get_logger().info('Credit Card not detected.')
            self.searching_card_pub.publish(Bool(data=True))
        elif not pos_detected and not self.pick_card:
            self.get_logger().info('POS not detected.')
            #self.searching_card_pub.publish(Bool(data=True))
        else:
            self.searching_card_pub.publish(Bool(data=False))
    """ 
    def detections_callback(self, msg: DetectionArray):
        if self.pick_card:
            card_detected = False
            for detection in msg.detections:
                if detection.class_name == 'CreditCard':  # Change based on the object to pick
                    card_detected = True
                    self.process_detection(detection)
            if not card_detected:
                self.get_logger().info('Credit Card not detected.')
                self.searching_card_pub.publish(Bool(data=True))
            else:
                self.searching_card_pub.publish(Bool(data=False))
        else:
            pos_detected = False
            for detection in msg.detections:
                if detection.class_name == 'POS':
                    pos_detected = True
                    self.searching_card_pub.publish(Bool(data=False))
                    self.process_detection(detection)
            if not pos_detected:
                self.get_logger().info('POS not detected.')

        """
        if not card_detected:
            self.get_logger().info('Credit Card not detected.')
            if self.pick_card:
                self.searching_card_pub.publish(Bool(data=True))
            else:
                self.searching_card_pub.publish(Bool(data=False))
                self.process_detection(msg.detections[0])
        else:
            self.searching_card_pub.publish(Bool(data=False))
        """

    def first_movement_callback(self, msg: Bool):
        self.first_movement = msg.data

    def process_detection(self, detection):
        bbox_center_x = detection.bbox.center.position.x
        bbox_center_y = detection.bbox.center.position.y

        self.bbox_area = detection.bbox.size.x * detection.bbox.size.y
        self.bounding_box_center = [bbox_center_x, bbox_center_y]

        self.update_print_info("bounding_box_center_x", bbox_center_x)
        self.update_print_info("bounding_box_center_y", bbox_center_y)

        # Reset PID targets
        self.pid_x.reset()
        self.pid_y.reset()

        # Update PID controllers with the current position
        error_x = self.pid_x(bbox_center_x)
        error_y = self.pid_y(bbox_center_y)

        # Clip adjustments to prevent excessive movements
        error_x = max(min(error_x, self.clip_val), -self.clip_val)
        error_y = max(min(error_y, self.clip_val), -self.clip_val)

        # Convert pixel error to camera frame using intrinsic parameters
        spatial_error_x = 1000 * error_x / self.intrinsics.fx if self.intrinsics else 0.0
        spatial_error_y = 1000 * error_y / self.intrinsics.fy if self.intrinsics else 0.0

        self.update_print_info("spatial_error_x", spatial_error_x)
        self.update_print_info("spatial_error_y", spatial_error_y)

        self.print_status()

        # Create a point in the camera frame
        camera_error_point = PointStamped()
        camera_error_point.header.frame_id = 'camera_color_optical_frame'
        camera_error_point.header.stamp = self.get_clock().now().to_msg()
        camera_error_point.point.x = spatial_error_x
        camera_error_point.point.y = spatial_error_y
        camera_error_point.point.z = 0.0

        try:
            transform = self.tf_buffer.lookup_transform('world', 'camera_color_optical_frame', rclpy.time.Time(), rclpy.duration.Duration(seconds=0.0))
            world_error_point = tf2_geometry_msgs.do_transform_point(camera_error_point, transform).point

            self.update_print_info("world_error_point_x", world_error_point.x)
            self.update_print_info("world_error_point_y", world_error_point.y)
            self.update_print_info("world_error_point_z", world_error_point.z)

            self.print_status()

            # Use depth information for Z-axis adjustment if available
            if self.pix and self.depth_image is not None:
                depth_adjustment = self.compute_depth_adjustment(self.pix[0], self.pix[1])
                if depth_adjustment >= 100:  # Discard pixels with depth less than 100 mm
                    self.update_print_info("depth_adjustment", depth_adjustment / 1000)
                    self.print_status()
                    self.process_depth_adjustment(depth_adjustment)
                    self.update_target_position(world_error_point, depth_adjustment)
                else:
                    self.get_logger().info('Depth adjustment discarded due to noise.')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 Error: {e}')

    def update_target_position(self, world_error_point, depth_adjustment):
        self.target_position.x = world_error_point.x + 0.01 # Offset to guarantee the camera to see the object
        self.target_position.y = depth_adjustment / 1000.0  # Convert to meters

        self.update_print_info("target_position_x", self.target_position.x)
        self.update_print_info("target_position_y", self.target_position.y)
        self.print_status()

    def process_depth_adjustment(self, depth_adjustment):
        if not self.pick_card:  # POS reaching
            if depth_adjustment / 1000 > MIN_Z_DISTANCE:
                self.target_position.z = depth_adjustment / 1000.0  # Convert to meters
                if self.target_position.x != 0.0 and self.target_position.y != 0.0:
                    self.send_goal(self.target_position)
                else:
                    print("All ZERO target were sent, invalid")
                self.update_print_info("target_position_z", self.target_position.z)
                self.print_status()
                self.get_logger().info(f'POS Target Position: X: {self.target_position.x}, Y: {self.target_position.y}, Z: {self.target_position.z}')
            else:
                self.get_logger().info('POS position is too close, stopping movement.')
        else:  # Credit Card reaching
            msg = Float32()
            msg.data = depth_adjustment / 1000
            self.depth_adjustment_pub.publish(msg)

            self.depth_buffer.append(depth_adjustment / 1000.0)  # Convert to meters and append to buffer
            if len(self.depth_buffer) == BUFFER_SIZE and np.mean(self.depth_buffer) < 0.1:
                self.get_logger().info('Credit Card position is too close, stopping movement.')
            else:
                self.target_position.y = depth_adjustment / 1000.0
                if self.target_position.x != 0.0 and self.target_position.y != 0.0:
                    self.send_goal(self.target_position)
                self.update_print_info("distance", self.target_position.y)  # Distance from the camera to the credit card
                self.print_status()

    def compute_depth_adjustment(self, x, y):
        if self.intrinsics and self.depth_image is not None:
            cv_image = self.depth_image  # Depth image in OpenCV format, cv_image is a numpy array of shape (height, width)

            # Scale the bounding box center coordinates from RGB to depth image resolution
            rgb_width, rgb_height = 1280, 720  # RGB image resolution
            depth_width, depth_height = cv_image.shape[1], cv_image.shape[0]  # Depth image resolution

            scale_x = depth_width / rgb_width
            scale_y = depth_height / rgb_height

            center_x = int(self.bounding_box_center[0] * scale_x)
            center_y = int(self.bounding_box_center[1] * scale_y)

            # Ensure a consistent neighborhood size (e.g., 31x31)
            neighborhood_size = 5  # 10

            min_x = max(0, center_x - neighborhood_size)
            max_x = min(cv_image.shape[1], center_x + neighborhood_size + 1)
            min_y = max(0, center_y - neighborhood_size)
            max_y = min(cv_image.shape[0], center_y + neighborhood_size + 1)

            neighborhood = cv_image[min_y:max_y, min_x:max_x]

            # Find the minimum non-zero depth value in the neighborhood
            non_zero_depths = neighborhood[neighborhood > 0]
            if non_zero_depths.size == 0:
                return float('inf')

            min_depth = non_zero_depths.min()

            # Get the coordinates of the minimum depth value
            indices = np.where(neighborhood == min_depth)
            indices = (indices[0][0], indices[1][0])  # Take the first occurrence

            # Convert neighborhood indices to original image indices
            original_x = min_x + indices[1]
            original_y = min_y + indices[0]

            depth = cv_image[original_y, original_x]

            if depth < 100:
                return float('inf')

            result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [original_x, original_y], depth)
            self.line = 'Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
            if self.pix_grade is not None:
                self.line += ' Grade: %2d' % self.pix_grade
            self.line += '\r'
            self.update_print_info("line", self.line)
            self.print_status()

            # Convert depth image to 8-bit for display
            display_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            display_image = cv2.applyColorMap(display_image, cv2.COLORMAP_JET)

            # Draw a red cross on the pixel with the minimum depth value
            cross_size = 5
            color = (0, 0, 255)  # Red color in BGR
            cv2.drawMarker(display_image, (original_x, original_y), color, markerType=cv2.MARKER_CROSS, markerSize=cross_size, thickness=2)

            # Display the image
            cv2.imshow('Depth Image with Min Depth Highlighted', display_image)
            cv2.waitKey(1)  # Wait for 1 ms to allow the image to be displayed

            return depth
        return self.target_position.z

    def depth_image_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            if self.bounding_box_center:
                self.pix = (int(self.bounding_box_center[0]), int(self.bounding_box_center[1]))
                if self.pix[0] < self.depth_image.shape[1] and self.pix[1] < self.depth_image.shape[0]:  # Boundary check
                    depth_adjustment = self.compute_depth_adjustment(self.pix[0], self.pix[1])
                    if depth_adjustment >= 100:  # Only consider pixels with depth >= 100 mm
                        self.line_depth_at_pixel = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (self.pix[0], self.pix[1], depth_adjustment)
                        self.update_print_info("line_depth_at_pixel", self.line_depth_at_pixel)
                        self.print_status()

                         # Publish the depth adjustment value if first_movement is False
                        if not self.first_movement and depth_adjustment != float('inf'):
                            msg = Float32()
                            msg.data = float(depth_adjustment) / 1000.0  # Convert to meters
                            self.depth_adjustment_pub.publish(msg)
                            
                    else:
                        self.get_logger().info('Depth value discarded due to noise.')
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

    def send_goal(self, target_position: Point):
        goal_msg = GoToPose.Goal()
        goal_msg.pose.position = target_position

        if not self.pick_card:
            goal_msg.pose.orientation.x = 0.0#-0.70388
            goal_msg.pose.orientation.y = 1.0 #0.70991
            goal_msg.pose.orientation.z = 0.0 #-0.015868
            goal_msg.pose.orientation.w = 0.0 #0.018082
            goal_msg.pose.position.y = -target_position.y

        else:
            goal_msg.pose.orientation.x = 0.64135
            goal_msg.pose.orientation.y = 0.6065
            goal_msg.pose.orientation.z = 0.3936
            goal_msg.pose.orientation.w = -0.25673
        
        if not self.pick_card:
            goal_filepath = os.path.expanduser("~/goal_pos.txt")
            with open(goal_filepath, "w") as file:
                file.write(f"Position:\n"
                           f"  x: {goal_msg.pose.position.x}\n"
                           f"  y: {goal_msg.pose.position.y}\n"
                           f"  z: {goal_msg.pose.position.z}\n"
                           f"Orientation:\n"
                           f"  x: {goal_msg.pose.orientation.x}\n"
                           f"  y: {goal_msg.pose.orientation.y}\n"
                           f"  z: {goal_msg.pose.orientation.z}\n"
                           f"  w: {goal_msg.pose.orientation.w}\n")
            self.get_logger().info(f"Goal saved to {goal_filepath}")

        self._action_client.wait_for_server() # it was 0.1 before
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

            self.update_print_info("target_position_x", self.updated_camera_position.x)
            self.update_print_info("target_position_y", self.updated_camera_position.y)
            self.update_print_info("target_position_z", self.updated_camera_position.z)
            self.print_status()

            if self.bbox_area != self.bbox_area_old:
                # Ensure bbox_area is a valid float before publishing
                try:
                    bbox_area_value = float(self.bbox_area)
                    self.bbox_area_pub.publish(Float32(data=bbox_area_value))
                    self.bbox_area_old = self.bbox_area
                except (ValueError, TypeError) as e:
                    self.get_logger().warning(f'Invalid bbox_area value: {self.bbox_area}. Error: {e}')
        else:
            self.get_logger().info('Goal failed :(')

    def stop_movement(self):
        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if cancel_response:
            self.get_logger().info('Goal cancelled successfully')

    def update_print_info(self, key, value):
        self.print_info[key] = value

    def print_status(self):
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, f"Bounding Box Center: X: {self.print_info['bounding_box_center_x']}, Y: {self.print_info['bounding_box_center_y']}")
        self.stdscr.addstr(1, 0, f"Spatial Error: X: {self.print_info['spatial_error_x']}, Y: {self.print_info['spatial_error_y']}")
        self.stdscr.addstr(2, 0, f"World Error Point: X: {self.print_info['world_error_point_x']}, Y: {self.print_info['world_error_point_y']}, Z: {self.print_info['world_error_point_z']}")
        self.stdscr.addstr(3, 0, f"Depth Adjustment: {self.print_info['depth_adjustment']}")
        self.stdscr.addstr(4, 0, f"Target Position: X: {self.print_info['target_position_x']}, Y: {self.print_info['target_position_y']}, Z: {self.print_info['target_position_z']}")
        self.stdscr.addstr(5, 0, self.print_info.get('line_coordinate', ''))
        self.stdscr.addstr(6, 0, self.print_info.get('line_depth_at_pixel', ''))
        self.stdscr.addstr(7, 0, f"Distance between camera and Credit Card along Y: {self.print_info['distance']}")
        self.stdscr.refresh()

def main(args=None):
    def curses_main(stdscr):
        rclpy.init(args=args)
        controller_node = ControllerNode(stdscr)

        try:
            rclpy.spin(controller_node)
        finally:
            controller_node.destroy_node()
            rclpy.shutdown()

    curses.wrapper(curses_main)

if __name__ == "__main__":
    main()
