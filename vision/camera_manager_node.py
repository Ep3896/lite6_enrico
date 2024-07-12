import rclpy
from rclpy.node import Node
from subprocess import Popen, run, PIPE
import os
import sys
import termios
import tty
import select
import time

def is_device_available(device):
    try:
        result = run(['fuser', device], stdout=PIPE, stderr=PIPE)
        if result.stdout:
            # Device is busy
            return False
        else:
            # Device is available
            return True
    except Exception as e:
        print(f"Error checking device availability: {e}")
        return False

def release_device(device):
    try:
        result = run(['fuser', '-k', device], stdout=PIPE, stderr=PIPE)
        if result.returncode == 0:
            print(f"Successfully released {device}")
        else:
            print(f"Failed to release {device}: {result.stderr.decode()}")
    except Exception as e:
        print(f"Error releasing device: {e}")

class CameraManager(Node):

    def __init__(self):
        super().__init__('camera_manager')
        self.current_mode = None
        self.process = None

        # Set up keyboard listener in the main thread
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.create_timer(0.1, self.keyboard_listener)

    def start_rs_pointcloud(self):
        if self.current_mode == 'pointcloud':
            return
        self.stop_current_process()
        self.get_logger().info('Starting rs_pointcloud')
        #rs_launch_file = os.path.expanduser('~/ros2_ws/yolo_detection/install/realsense2_camera/share/realsense2_camera/examples/pointcloud/rs_pointcloud_launch.py')
        self.process = Popen(['ros2', 'launch', 'realsense2_camera', 'rs_pointcloud_launch.py'])
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

    def stop_current_process(self):
        if self.current_mode == 'pointcloud':
            self.stop_rs_pointcloud()
        elif self.current_mode == 'template_matching':
            self.stop_template_matching()
        release_device('/dev/video4')
        time.sleep(2)  # Add delay to ensure the device is properly released

    def keyboard_listener(self):
        try:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 't':
                    self.toggle_mode()
        except Exception as e:
            self.get_logger().error(f"Keyboard listener error: {e}")

    def toggle_mode(self):
        if self.current_mode == 'pointcloud':
            self.start_template_matching()
        else:
            self.start_rs_pointcloud()

    def destroy_node(self):
        self.stop_current_process()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_manager = CameraManager()
    rclpy.spin(camera_manager)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

