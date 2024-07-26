import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String,Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# Initialize video capture for the webcam (channel 4)
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def get_line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def does_line_cross_center(x1, y1, x2, y2, center_x, center_y):
    # Check if the line segment crosses the center of the frame
    if x1 == x2:  # Vertical line
        return center_x == x1 and min(y1, y2) <= center_y <= max(y1, y2)
    else:
        # Calculate line equation y = mx + c
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        # Check if the center point satisfies the line equation
        y_at_center_x = m * center_x + c
        x_at_center_y = (center_y - c) / m
        return (min(y1, y2) <= center_y <= max(y1, y2) and abs(x_at_center_y - center_x) < 1) or \
               (min(x1, x2) <= center_x <= max(x1, x2) and abs(y_at_center_x - center_y) < 1)

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



    def publish_bbox_center(self, x, y):
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.publisher_.publish(msg)

    def publish_alignment_status(self, status):
        msg = Bool()
        msg.data = status
        self.status_publisher_.publish(msg)

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
    distance_threshold = 100  # Distance threshold in pixels

    # Convert into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Image processing (smoothing)
    # Averaging
    blur = cv2.blur(gray, (3, 3))

    # Apply logarithmic transform
    img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255

    # Specify the data type
    img_log = np.array(img_log, dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    # Canny Edge Detection with lower thresholds
    edges = cv2.Canny(bilateral, 30, 100)

    # Hough Line Transform with lower threshold for more sensitivity
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    longest_line = None
    max_length = 0

    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Filter for almost vertical lines
            if theta < np.pi / 18 or theta > 17 * np.pi / 18:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                length = get_line_length(x1, y1, x2, y2)
                if length > max_length :
                    longest_line = (x1, y1, x2, y2)
                    max_length = length

    # Draw the bounding box or cross on the detected line near the center
    edges_with_bounding_box = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to BGR for drawing

    # Draw a cross at the center of the frame
    cross_size = 10
    cv2.line(edges_with_bounding_box, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 255, 0), 2)
    cv2.line(edges_with_bounding_box, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 255, 0), 2)

    if longest_line is not None:
        x1, y1, x2, y2 = longest_line
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        box_size = 10
        cv2.rectangle(edges_with_bounding_box, (mid_x - box_size, mid_y - box_size), 
                      (mid_x + box_size, mid_y + box_size), (0, 0, 255), 2)
        if does_line_cross_center(x1, y1, x2, y2, center_x, center_y):
            print("OK")
            node.publish_alignment_status(True)
        else:
            node.publish_alignment_status(False)
        if abs(mid_x - center_x) <= box_size and abs(mid_y - center_y) <= box_size:
            print("DONE")
        # Publish the center coordinates of the bounding box
        node.publish_bbox_center(mid_x, mid_y)

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
