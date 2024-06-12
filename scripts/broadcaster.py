import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
import numpy as np

class StaticFramePublisher(Node):

    def __init__(self):
        super().__init__('static_tf_broadcaster')
        self._tf_publisher = StaticTransformBroadcaster(self)
        self.publish_static_transforms()

    def publish_static_transforms(self):
        transforms = [
            ('world', 'link_base', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('link_base', 'link1', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link1', 'link2', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link2', 'link3', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link3', 'link4', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link4', 'link5', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link5', 'link6', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link6', 'link_eef', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('link_eef', 'camera_link', [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ('camera_link', 'camera_depth_frame', [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]),
            ('camera_depth_frame', 'camera_depth_optical_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_link', 'camera_color_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_color_frame', 'camera_color_optical_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_link', 'camera_left_ir_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_left_ir_frame', 'camera_left_ir_optical_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_link', 'camera_right_ir_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('camera_right_ir_frame', 'camera_right_ir_optical_frame', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('link_eef', 'uflite_gripper_link', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ('uflite_gripper_link', 'link_tcp', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        ]

        for parent, child, translation, rotation in transforms:
            static_transform_stamped = TransformStamped()
            static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
            static_transform_stamped.header.frame_id = parent
            static_transform_stamped.child_frame_id = child
            static_transform_stamped.transform.translation.x = float(translation[0])
            static_transform_stamped.transform.translation.y = float(translation[1])
            static_transform_stamped.transform.translation.z = float(translation[2])

            quat = self.euler_to_quaternion(float(rotation[0]), float(rotation[1]), float(rotation[2]))
            static_transform_stamped.transform.rotation = quat

            self._tf_publisher.sendTransform(static_transform_stamped)

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

def main(args=None):
    rclpy.init(args=args)
    node = StaticFramePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

