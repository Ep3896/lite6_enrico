import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# Initialize video capture for the webcam (channel 4)
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

class CardEdgeDetection(Node):
    def __init__(self):
        super().__init__('card_edge_detection')

        low_latency_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.publisher_ = self.create_publisher(Float32MultiArray, '/control/bbox_center', 10)
        self.status_publisher_ = self.create_publisher(Bool, '/control/alignment_status', 10)
        self.line_is_far_publisher_ = self.create_publisher(Bool, '/control/line_is_far', 10)

    def publish_bbox_center(self, x, y):
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.publisher_.publish(msg)

    def publish_alignment_status(self, status):
        msg = Bool()
        msg.data = status
        self.status_publisher_.publish(msg)

    def publish_line_is_far(self, status):
        msg = Bool()
        msg.data = status
        self.line_is_far_publisher_.publish(msg)

rclpy.init()
node = CardEdgeDetection()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Get the center of the frame
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    # Convert into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image processing (smoothing)
    blur = cv2.blur(gray, (3, 3))

    # Apply logarithmic transform
    img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255
    img_log = np.array(img_log, dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    # Canny Edge Detection with lower thresholds
    edges = cv2.Canny(bilateral, 30, 100)

    # Initialize the edges_with_bounding_box to avoid NameError
    edges_with_bounding_box = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for drawing

    # Find coordinates of white pixels in the edge-detected image
    white_pixels = np.column_stack(np.where(edges > 0))

    if white_pixels.size > 0:
        # Identify the endpoint with the maximum y value
        max_y_index = np.argmax(white_pixels[:, 0])
        max_y_point = (white_pixels[max_y_index, 1], white_pixels[max_y_index, 0])  # (x, y)

        # Draw a cross at the center of the frame
        cross_size = 10
        cv2.line(edges_with_bounding_box, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 255, 0), 2)
        cv2.line(edges_with_bounding_box, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 255, 0), 2)

        # Draw a smaller red cross at the endpoint with the maximum y value
        cross_size_red = 8
        cv2.line(edges_with_bounding_box, (max_y_point[0] - cross_size_red, max_y_point[1]), 
                 (max_y_point[0] + cross_size_red, max_y_point[1]), (0, 0, 255), 2)
        cv2.line(edges_with_bounding_box, (max_y_point[0], max_y_point[1] - cross_size_red), 
                 (max_y_point[0], max_y_point[1] + cross_size_red), (0, 0, 255), 2)

        print(f"Red cross drawn at the endpoint with maximum y value: {max_y_point}")  # Debug print

        # Publish True to /control/line_is_far if max y < 260, else publish False
        if max_y_point[1] < 260: # a bit more that 240 but it is to be safe that the robot grasp the card 
            node.publish_line_is_far(True)
        else:
            node.publish_line_is_far(False)

        # Calculate and publish the center of the detected line
        node.publish_bbox_center(max_y_point[0], max_y_point[1])
    else:
        print("No white pixels found.")

    # Display the resulting frames
    cv2.imshow('Original', frame)
    cv2.imshow('Canny Edges with Bounding Box', edges_with_bounding_box)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
rclpy.shutdown()
