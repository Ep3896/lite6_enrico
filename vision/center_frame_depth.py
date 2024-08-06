import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
from std_msgs.msg import Float32

import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener(Node):
    def __init__(self, depth_image_topic, depth_info_topic):
        node_name = os.path.basename(sys.argv[0]).split('.')[0]
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(msg_Image, depth_image_topic, self.imageDepthCallback, 30)
        self.sub_info = self.create_subscription(CameraInfo, depth_info_topic, self.imageDepthInfoCallback, 30)
        self.intrinsics = None
        self.pix = (320, 240)
        self.pix_grade = None


        self.center_depth_pub = self.create_publisher(Float32, '/control/depth_at_centroid', 30)
    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # pick one pixel among all the pixels with the closest range:
            #indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
            #pix = (indices[1], indices[0])
            #self.pix = pix
            pix = self.pix
            line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])

            if self.intrinsics:
                depth = cv_image[pix[1], pix[0]]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                line += '  Coordinate: %8.2f %8.2f %8.2f.' % (result[0], result[1], result[2])
                msg = Float32()
                msg.data = result[2]/1000
                print("depth is:",msg.data)
                self.center_depth_pub.publish(msg)

            if (not self.pix_grade is None):
                line += ' Grade: %2d' % self.pix_grade
            line += '\r'
            sys.stdout.write(line)
            sys.stdout.flush()

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return


    def imageDepthInfoCallback(self, cameraInfo):
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
            print(e)
            return

def main():
    depth_image_topic = '/camera/camera/depth/image_rect_raw'
    depth_info_topic = '/camera/camera/depth/camera_info'

    listener = ImageListener(depth_image_topic, depth_info_topic)
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    main()