import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool
import cv2
import numpy as np

class TemplateMatchingNode(Node):

    def __init__(self):
        super().__init__('template_matching_node')
        self.template_path = '/home/lite6/Desktop/Piacenti/2.Vision/3.Training/9.Training_10_07/Shape_Detector/pos_real_logo1.png'
        self.webcam_channel = 4
        self.threshold = 0.7
        self.scales = np.linspace(0.5, 2.0, 30)
        
        self.depth_pub = self.create_publisher(Float32, '/control/depth_at_centroid', 10)
        self.bbox_center_pub = self.create_publisher(Float32MultiArray, '/control/bbox_center', 10)
        self.moving_along_y_pub = self.create_publisher(Bool, '/control/start_moving_along_y', 10)
        
        self.detect_logo_from_webcam()

    def detect_logo_from_webcam(self):
        template = cv2.imread(self.template_path, 0)
        if template is None:
            self.get_logger().error(f"Unable to load template image from {self.template_path}")
            return

        template_height, template_width = template.shape
        template = cv2.GaussianBlur(template, (3, 3), 0) # it was 7,7

        cap = cv2.VideoCapture(self.webcam_channel)
        if not cap.isOpened():
            self.get_logger().error(f"Could not open webcam with index {self.webcam_channel}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("Failed to read frame from webcam")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0) # It was 7,7
            frame_height, frame_width = gray_frame.shape

            best_match_val = -1
            best_match_loc = None
            best_match_scale = None

            for scale in self.scales:
                resized_template = cv2.resize(template, (int(template_width * scale), int(template_height * scale)))
                if resized_template.shape[0] > frame_height or resized_template.shape[1] > frame_width:
                    continue

                res = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_scale = scale

            if best_match_val >= self.threshold:
                top_left = best_match_loc
                bottom_right = (int(top_left[0] + template_width * best_match_scale), int(top_left[1] + template_height * best_match_scale))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                centroid_x = (top_left[0] + bottom_right[0]) / 2
                centroid_y = (top_left[1] + bottom_right[1]) / 2
                bbox_center = Float32MultiArray(data=[centroid_x, centroid_y])
                self.bbox_center_pub.publish(bbox_center)

                frame_center_x = frame_width / 2
                frame_center_y = frame_height / 2
                distance_x = frame_center_x - centroid_x
                distance_y = frame_center_y - centroid_y
                adjustment = 0.001 * np.array([distance_x, distance_y])  # Scale as needed
                self.adjust_robot_position(adjustment)

                # Commenting out depth_at_centroid until get_depth_at_centroid is implemented
                # depth_at_centroid = self.get_depth_at_centroid(centroid_x, centroid_y)
                # self.depth_pub.publish(Float32(data=depth_at_centroid))

                # Publish False to /control/start_moving_along_y once bounding box is generated
                self.moving_along_y_pub.publish(Bool(data=False))

            cv2.imshow('Detected Logo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    """
    def get_depth_at_centroid(self, x, y):
        # Implement depth extraction logic here
        depth = 0.0  # Replace with actual depth value
        return depth
    """

    def adjust_robot_position(self, adjustment):
        # Implement robot position adjustment here
        pass

def main(args=None):
    rclpy.init(args=args)
    node = TemplateMatchingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
