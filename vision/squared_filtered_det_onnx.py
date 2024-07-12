import cv2
import numpy as np
from pathlib import Path
from with_onnx import OnnxObjectDetector

# Define paths to ONNX models
PREPROCESSING_PATH = Path("models/preprocessing.onnx")
YOLO_PATH = Path("models/yolo.onnx")
NMS_PATH = Path("models/nms.onnx")
POSTPROCESSING_PATH = Path("models/postprocessing.onnx")

# Initialize the object detector
detector = OnnxObjectDetector(
    preprocessing_path=PREPROCESSING_PATH,
    yolo_path=YOLO_PATH,
    nms_path=NMS_PATH,
    postprocessing_path=POSTPROCESSING_PATH
)

# Open the camera with the specified index
camera_index = 4
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open video capture device at index {camera_index}.")
    exit()

# Set input size expected by the YOLO model
input_h, input_w = 256, 256

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reverse the colors: convert black to white and white to black
    reversed_frame = cv2.bitwise_not(frame)

    # Filter the frame to keep only nearly black pixels
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([100, 100, 100])  # Adjust this upper bound to include more shades of black, 75,75,75 seems good btw
    black_mask = cv2.inRange(reversed_frame, lower_black, upper_black)
    filtered_frame = cv2.bitwise_and(reversed_frame, reversed_frame, mask=black_mask)
    filtered_frame[black_mask == 0] = 255  # Set non-black pixels to white

    # Resize the frame to the expected input size
    resized_frame = cv2.resize(filtered_frame, (input_w, input_h))

    # Preprocess the frame
    preprocessed_data = detector.preprocessing(resized_frame, input_h=input_h, input_w=input_w, fill_value=128)
    preprocessed_img = preprocessed_data["preprocessed_img"]

    # Add a batch dimension to the input tensor
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    print(f"Preprocessed image shape: {preprocessed_img.shape}")

    # Run YOLO model inference
    yolo_outputs = detector.yolo(preprocessed_img)
    print(f"YOLO outputs: {yolo_outputs}")

    # Apply Non-Maximum Suppression
    nms_results = detector.nms(yolo_outputs['output0'], max_output_boxes_per_class=10, iou_threshold=0.5, score_threshold=0.5)
    print(f"NMS results: {nms_results}")

    # Postprocess the results
    postprocessed_results = detector.postprocessing(
        input_h=input_h,
        input_w=input_w,
        boxes_xywh=nms_results["selected_boxes_xywh"],
        padding_tlbr=preprocessed_data["padding_tlbr"]
    )

    # Draw the detection results on the filtered frame (resized)
    for box, score, class_id in zip(
            postprocessed_results["boxes_xywhn"],
            nms_results["selected_class_scores"],
            nms_results["selected_class_ids"]
    ):
        x, y, w, h = box
        side_length = min(w, h)
        x1 = int((x - side_length / 2) * frame.shape[1])
        y1 = int((y - side_length / 2) * frame.shape[0])
        x2 = int((x + side_length / 2) * frame.shape[1])
        y2 = int((y + side_length / 2) * frame.shape[0])
        cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(filtered_frame, f"Digit: {class_id}, Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the filtered frame
    cv2.imshow("Digit Detection", filtered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

