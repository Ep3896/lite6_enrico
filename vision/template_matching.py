import cv2
import numpy as np

def detect_logo_from_webcam(template_path, webcam_channel, threshold, scales):
    # Load the template image
    template = cv2.imread(template_path, 0)  # Template image in grayscale
    if template is None:
        print(f"Error: Unable to load template image from {template_path}")
        return

    template_height, template_width = template.shape

    # Apply Gaussian Blur to the template to make it less precise
    template = cv2.GaussianBlur(template, (7, 7), 0)

    # Open the webcam
    cap = cv2.VideoCapture(webcam_channel)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {webcam_channel}.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to the frame to make it less precise
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        frame_height, frame_width = gray_frame.shape

        best_match_val = -1
        best_match_loc = None
        best_match_scale = None

        # Loop over different scales of the template
        for scale in scales:
            resized_template = cv2.resize(template, (int(template_width * scale), int(template_height * scale)))
            resized_template_height, resized_template_width = resized_template.shape

            # Ensure the resized template is smaller than the frame
            if resized_template_height > frame_height or resized_template_width > frame_width:
                continue

            res = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_match_scale = scale

        # Only draw bounding boxes for scales >= 0.9
        if best_match_val >= threshold and best_match_scale >= 0.9:
            top_left = best_match_loc
            bottom_right = (int(top_left[0] + template_width * best_match_scale), int(top_left[1] + template_height * best_match_scale))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f'Scale: {best_match_scale:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detected logos
        cv2.imshow('Detected Logo', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    template_path = '/home/lite6/Desktop/Piacenti/2.Vision/3.Training/9.Training_10_07/Shape_Detector/pos_real_logo1.png'
    webcam_channel = 4  # You may need to adjust this index based on your setup
    threshold = 0.5
    scales = np.linspace(0.5, 2.0, 30)
    detect_logo_from_webcam(template_path, webcam_channel, threshold, scales)

