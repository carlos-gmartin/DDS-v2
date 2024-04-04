import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Constants
# Distance from camera to object (drone) measured in centimeters
KNOWN_DISTANCE = 100
# Width of the drone in the real world measured in centimeters
KNOWN_WIDTH = 30

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

def main():
    # Load custom YOLO model
    custom_model = "./runs/detect/train4/weights/best.pt"
    model = YOLO(custom_model)

    # Measure drone width in pixels
    drone_width_pixels = 490  # Replace with your measured width

    # Estimate focal length in pixels
    focal_length_value = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, drone_width_pixels)
    
    print("Estimated focal length (in pixels):", focal_length_value)

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    print("============ Opened OpenCV ============")
    
    while True:
        _, img = cap.read()
        
        print(" ============ Opened MODEL ============")
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

                label = f"{model.names[int(c)]}, Distance: {estimated_distance_m:.2f} m"  # Combine object label with distance
                annotator.box_label(b, label, color=(0, 0, 255))  # Red color in RGB format
          
        img = annotator.result()  

        # Resize the frame to fit the window without changing the window size
        resized_img = cv2.resize(img, (640, 480))

        cv2.imshow('Object Detection', resized_img)     

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
