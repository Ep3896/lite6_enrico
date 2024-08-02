import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from subprocess import Popen, run, PIPE
import os
import time

def is_device_available(device):
    try:
        result = run(['fuser', device], stdout=PIPE, stderr=PIPE)
        return not bool(result.stdout)
    except Exception as e:
        print(f"Error checking device availability: {e}")
        return False

def release_device(device):
    try:
        run(['fuser', '-k', device], stdout=PIPE, stderr=PIPE)
    except Exception as e:
        print(f"Error releasing device: {e}")

class CameraManager(Node):

    def __init__(self):
        super().__init__('camera_manager')
        self.current_mode = None
        self.process = None

        self.create_subscription(Bool, '/control/start_pointcloud', self.start_pointcloud_callback, 10)
        self.create_subscription(Bool, '/control/start_template_matching', self.start_template_matching_callback, 10)
        self.create_subscription(Bool, '/control/start_card_edge_detection', self.start_card_edge_detection_callback, 10)

        self.prompt_publisher=self.create_publisher(Bool, '/control/start_moving_along_y', 10)

    def start_pointcloud_callback(self, msg):
        if msg.data:
            self.start_rs_pointcloud()
        else:
            self.stop_rs_pointcloud()

    def start_template_matching_callback(self, msg):
        if msg.data:
            self.start_template_matching()
            move_along_y_msg = Bool()
            move_along_y_msg.data = True
            self.get_logger().info('Publishing start_moving_along_y')
            self.prompt_publisher.publish(move_along_y_msg)

        else:
            self.stop_template_matching()

    def start_card_edge_detection_callback(self, msg):
        if msg.data:
            self.start_card_edge_detection()
        else:
            self.stop_card_edge_detection()

    def start_rs_pointcloud(self):
        if self.current_mode == 'pointcloud':
            return
        self.stop_current_process()
        self.get_logger().info('Starting rs_pointcloud')
        self.process = Popen(['ros2', 'launch', 'realsense2_camera', 'rs_pointcloud_launch.py', 
                          'depth_module.profile:=640x360x30', 'depth_module.exposure:=6000'])
        self.current_mode = 'pointcloud'

    def stop_rs_pointcloud(self):
        if self.current_mode != 'pointcloud':
            return
        self.get_logger().info('Stopping rs_pointcloud')
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.current_mode = None

    def start_template_matching(self):
        if self.current_mode == 'template_matching':
            return
        self.stop_current_process()
        self.get_logger().info('Starting template matching')
        while not is_device_available('/dev/video4'):
            self.get_logger().info('Waiting for device /dev/video4 to become available...')
            release_device('/dev/video4')
            time.sleep(1)
        script_path = os.path.expanduser('/home/lite6/Desktop/Piacenti/2.Vision/3.Training/9.Training_10_07/Shape_Detector/template_matching.py')
        self.process = Popen(['python3', script_path])
        self.current_mode = 'template_matching'

    def stop_template_matching(self):
        if self.current_mode != 'template_matching':
            return
        self.get_logger().info('Stopping template matching')
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.current_mode = None

    def start_card_edge_detection(self):
        if self.current_mode == 'card_edge_detection':
            return
        self.stop_current_process()
        self.get_logger().info('Starting card edge detection')
        while not is_device_available('/dev/video4'):
            self.get_logger().info('Waiting for device /dev/video4 to become available...')
            release_device('/dev/video4')
            time.sleep(1)
        script_path = os.path.expanduser('/home/lite6/ros2_ws/ws_moveit2/src/lite6_enrico/vision/card_edge_detection.py')
        self.process = Popen(['python3', script_path])
        self.current_mode = 'card_edge_detection'

    def stop_card_edge_detection(self):
        if self.current_mode != 'card_edge_detection':
            return
        self.get_logger().info('Stopping card edge detection')
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.current_mode = None

    def stop_current_process(self):
        if self.current_mode == 'pointcloud':
            self.stop_rs_pointcloud()
        elif self.current_mode == 'template_matching':
            self.stop_template_matching()
        elif self.current_mode == 'card_edge_detection':
            self.stop_card_edge_detection()
        release_device('/dev/video4')
        time.sleep(2)  # Add delay to ensure the device is properly released



def main(args=None):
    rclpy.init(args=args)
    camera_manager = CameraManager()
    rclpy.spin(camera_manager)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
