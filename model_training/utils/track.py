import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Function to clear the contents of the info file
def clear_info_file(file_path):
    with open(file_path, 'w') as f:
        f.truncate(0)

# Focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    """
    Calculate the focal length (in pixels) using the measured distance, real width,
    and width of the object in the reference image.
    """
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# Distance estimation function
def distance_finder(focal_length, real_width, width_in_frame):
    """
    Estimate the distance to the object (drone) using the focal length, real width,
    and width of the object in the current frame.
    """
    distance = (real_width * focal_length) / width_in_frame
    return distance / 100  # Convert from centimeters to meters

# Function to calculate the angle from the camera's perspective
def angle_from_camera(object_center_x, frame_width):
    """
    Calculate the angle of the object relative to the camera's position.

    Parameters:
        object_center_x (float): X-coordinate of the object's center in the frame.
        frame_width (int): Width of the frame in pixels.

    Returns:
        float: Angle of the object relative to the camera's centerline.
    """
    # Calculate the horizontal distance of the object from the center of the frame
    distance_from_center = object_center_x - (frame_width / 2)
    
    # Calculate the angle using trigonometry
    if distance_from_center == 0:
        return 90.0  # Object is exactly in the center
    
    angle = np.arctan(distance_from_center / (frame_width / 2)) * (180 / np.pi)
    
    # Adjust the angle to be relative to the camera's position
    if distance_from_center < 0:
        angle = 90 - angle  # Object is to the left of the center
    else:
        angle = 90 + angle  # Object is to the right of the center
    
    return angle

# Function to save detection info to a file
def save_detection_info(file_path, label):
    with open(file_path, 'a') as f:
        f.write(label + '\n')

# Function to track and display drone detection
def track(custom_model, drone_width_pixels, KNOWN_DISTANCE, KNOWN_WIDTH, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, video, info_file):

    # Loading pretrained model.
    model = YOLO(custom_model)

    # Estimate focal length in pixels
    focal_length_value = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, drone_width_pixels)
    
    print("Estimated focal length (in pixels):", focal_length_value)

    # Initialize video capture
    if video == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video)
        
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)

    print("============ Opened OpenCV and model ============")

    # Clear the contents of the info file
    clear_info_file(info_file)
    
    while True:
        _, img = cap.read()
        # BGR to RGB conversion is performed under the hood
        results = model.predict(img, tracker="bytetrack.yaml")

        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
                c = box.cls

                # Calculate bounding box width in pixels
                bounding_box_width_pixels = b[2] - b[0]
                
                # Estimate distance using focal length and object width
                estimated_distance_m = distance_finder(focal_length_value, KNOWN_WIDTH, bounding_box_width_pixels)

                # Calculate the angle of the object relative to the camera's position
                object_center_x = (b[2] + b[0]) / 2  # X-coordinate of the object's center
                angle = angle_from_camera(object_center_x, RESOLUTION_WIDTH)
                
                label = f"{model.names[int(c)]}, Distance: {estimated_distance_m:.2f} m, Angle: {angle:.2f} degrees"
                annotator.box_label(b, label, color=(0, 0, 255))  # Red color in RGB format

                # Save detection info to the file
                save_detection_info(info_file, label)
          
        img = annotator.result()  

        # Resize the frame to fit the window without changing the window size
        resized_img = cv2.resize(img, (640, 480))

        cv2.imshow('Drone Detection', resized_img)     

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
