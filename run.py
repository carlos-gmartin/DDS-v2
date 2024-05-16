import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Set the width and height of the radar grid
width = 1335
height = 800
radius = 30

# Distance of each radius. 100 pixels ~ 20m
RADIUS = 100

# Global variables
point_coordinates = []
drone_coordinates = []
points_added = 0
squares = []  # List to store coordinates of squares

def pos_angle(base_angle, length):
    """
    Calculate the position of a point at a given angle and distance from the center.

    Parameters:
        base_angle (float): The base angle from the center (in degrees).
        length (float): The distance from the center.

    Returns:
        tuple: The coordinates (x, y) of the point.
    """
    # Calculate the coordinates of the point using trigonometry
    pt = (width // 2 + int(length * np.sin(np.radians(base_angle))), height + int(length * np.cos(np.radians(base_angle))))
    return pt

def put_degree(txt, shift=0, level=0, txt_shift=0, txt_level=0):
    """
    Add degree labels to the radar grid.

    Parameters:
        txt (str): The degree label.
        shift (int): Shift the label horizontally.
        level (int): Shift the label vertically.
        txt_shift (int): Shift the text horizontally.
        txt_level (int): Shift the text vertically.
    """
    # Calculate the position for the degree label and draw it on the image
    cv2.putText(img_radar, txt, org=pos_angle(int(txt) + 90 + txt_shift, 353 + txt_level), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    # Draw a small circle at the degree label position
    cv2.circle(img_radar, center=pos_angle(int(txt) + 90 + shift, 378 + level), radius=2, color=(0, 255, 0))

def add_dot(x, y, angle):
    """
    Add a point to the radar grid.

    Parameters:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        angle (float): The angle of the point from the center (in degrees).
    """
    global img_radar, points_added, squares
    pt1 = (width // 2, height)  # Center of the radar grid
    pt2 = (x, y)  # Coordinates of the clicked point
    # Draw a circle at the clicked point
    cv2.circle(img_radar, center=pt2, radius=5, color=[0, 0, 250], thickness=-1)
    # Draw a line from the center to the clicked point
    # cv2.line(img_radar, pt1=pt1, pt2=pt2, color=[8,255,8], thickness=1)
    # Show the updated radar grid
    cv2.imshow('Radar', img_radar)
    # Append the point coordinates to the list
    point_coordinates.append(pt2)
    points_added += 1
    print(f"Added dot at ({x}, {y}), Angle: {angle}")

    if points_added == 4:
        # If four points have been added, reset the counter for the next square
        points_added = 0
        print("Square complete")
        # Extract the coordinates of the square
        square_coordinates = point_coordinates[-4:]
        # Draw the no-fly zone square
        add_no_fly_zone(square_coordinates)
        for i in range(0, len(squares)):
            print(f"Added Square {squares[i]}")

def add_no_fly_zone(square_coordinates):
    """
    Create a square around the drawn points to define a no-fly zone.
    """
    global img_radar

    # Find the minimum and maximum x and y coordinates of the drawn points
    min_x = min(square_coordinates, key=lambda x: x[0])[0]
    max_x = max(square_coordinates, key=lambda x: x[0])[0]
    min_y = min(square_coordinates, key=lambda x: x[1])[1]
    max_y = max(square_coordinates, key=lambda x: x[1])[1]

    # Add points to define the square
    square_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    squares.append(square_points)  # Save the coordinates of the square

    # Draw the square on the grid
    cv2.polylines(img_radar, [np.array(square_points)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow('Radar', img_radar)

def check_no_fly_zone(drone_coordinates):
    """
    Check if the drone is in the no-fly zone.

    Parameters:
        drone_coordinates (tuple): The coordinates (x, y) of the drone.

    Returns:
        bool: True if the drone is in the no-fly zone, False otherwise.
    """
    for square in squares:
        min_x = min(square, key=lambda x: x[0])[0]
        max_x = max(square, key=lambda x: x[0])[0]
        min_y = min(square, key=lambda x: x[1])[1]
        max_y = max(square, key=lambda x: x[1])[1]

        # Check if drone coordinates fall within the square boundaries
        if min_x <= drone_coordinates[0] <= max_x and min_y <= drone_coordinates[1] <= max_y:
            return True  # Drone is in the no-fly zone

    return False  # Drone is not in the no-fly zone

def add_drone(distance, angle):
    """
    Add a single drone at the specified distance and angle from the center of the radar grid.
    """
    global img_radar

    # Clear the previous drone circle
    img_copy = img_radar.copy()  # Make a copy of the image to preserve the original

    # Calculate the coordinates of the new drone
    x = width // 2 + int(distance / 20 * RADIUS * np.cos(np.radians(angle)))  # Calculate x coordinate
    y = height - int(distance / 20 * RADIUS * np.sin(np.radians(angle)))  # Calculate y coordinate (adjust for alignment)

    # Check if the drone is in the no-fly zone
    in_no_fly_zone = check_no_fly_zone((x, y))

    # Draw the new drone on the radar grid
    pt1 = (width // 2, height)  # Center of the radar grid
    pt2 = (x, y)  # Coordinates of the drone

    if in_no_fly_zone:
        # If the drone is in the no-fly zone, draw it in red and display a warning message
        color = [0, 0, 255]  # Red color

        # Display the warning message above the drone
        warning_text = "WARNING: Drone in No-Fly Zone!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x - text_size[0] // 2
        text_y = y - 20  # Adjust this value to set the vertical position of the warning message
        cv2.putText(img_copy, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # If the drone is not in the no-fly zone, draw it in gray
        color = [50, 50, 50]  # Gray color

    cv2.circle(img_copy, center=pt2, radius=5, color=color, thickness=-1)  # Draw a circle at the drone position
    cv2.line(img_copy, pt1=pt1, pt2=pt2, color=color, thickness=1)  # Draw a line from the center to the drone position

    # Rectangle
    box_size = 20  # Size of the box
    pt1_rect = (x - box_size // 2, y - box_size // 2)  # Top-left corner of the rectangle
    pt2_rect = (x + box_size // 2, y + box_size // 2)  # Bottom-right corner of the rectangle
    cv2.rectangle(img_copy, pt1=pt1_rect, pt2=pt2_rect, color=[0, 255, 0], thickness=1)  # Draw the rectangle

    # Display the distance of the drone on the image
    cv2.putText(img_copy, f"Distance: {distance}m", (x - 40, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Radar', img_copy)  # Show the updated radar grid

    # Clear the previous drone coordinates
    drone_coordinates.clear()

    # Append the new drone coordinates to the list
    drone_coordinates.append(pt2)

def on_mouse(event, x, y, flags, param):
    """
    Handle mouse events.

    Parameters:
        event: The type of mouse event (e.g., mouse click).
        x (int): The x-coordinate of the mouse cursor.
        y (int): The y-coordinate of the mouse cursor.
        flags: Any relevant flags associated with the mouse event.
        param: Any additional parameters passed to the function.
    """
    # If the left mouse button is pressed, add a dot to the radar grid
    if event == cv2.EVENT_LBUTTONDOWN:
        add_dot(x, y, np.arctan2(y - height, x - (width // 2)) * 180 / np.pi)
    # If the right mouse button is pressed, define the no-fly zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        add_no_fly_zone()

def start_grid(example_img):
    global img_radar

    # Load the example grid image
    img_radar = cv2.imread(example_img)

    # Set mouse callback function
    cv2.namedWindow('Radar')
    cv2.setMouseCallback('Radar', on_mouse)

    for base_angle in range(90, 300, 30):
        pt1 = (width // 2 + int(350 * np.sin(np.radians(base_angle))), height + int(350 * np.cos(np.radians(base_angle))))
        cv2.line(img_radar, pt1=pt1, pt2=(width // 2, height), color=[0, 255, 0], thickness=2)
    
    # Draw the distance lines.
    for i in range(1, 1000, RADIUS):
        cv2.circle(img_radar, (width // 2, height), i, color=[0, 255, 0])

    put_degree('30')
    put_degree('60', -2, -2)
    put_degree('90', -4, -13)
    put_degree('120', -3.5, -18, 2, 15)
    put_degree('150', -2, -20, 2, 28)

    for n, i in enumerate(range(0, 1000, 100)):
        cv2.putText(img_radar, f'{20*(n+1)}m', org=pos_angle(90, 50+i), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,color=(0, 255, 0), thickness=1)

    cv2.imshow('Radar',img_radar)

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

def run_radar_project(custom_model, radar_image, video, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, KNOWN_DISTANCE, KNOWN_WIDTH, drone_width_pixels):
    global img_radar

    # Start the radar grid
    start_grid(radar_image)

    # While loop for audio detection. 

    # If the drone audio is detected, run the project:

    

    # Initialize the object detection model
    model = YOLO(custom_model)

    # Estimate focal length in pixels
    focal_length_value = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, drone_width_pixels)
    
    print("Estimated focal length (in pixels):", focal_length_value)
    # Initialize video capture
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)

    # Create a separate window for drone detection
    cv2.namedWindow('Drone Detection')

    while True:
        _, img = cap.read()
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Perform object detection using YOLO
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

                # Add the drone to the radar map
                add_drone(estimated_distance_m, angle)
        
        img = annotator.result()  

        # Resize the frame to fit the window without changing the window size
        resized_img = cv2.resize(img, (1280, 720))

        cv2.imshow('Drone Detection', resized_img)     

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # Close windows on ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the radar project
if __name__ == "__main__":

    # Initialize the object detection model
    custom_model = "./model_training/model/train4/weights/best.pt"  # Pretrained YOLO model
    #video = "./model_training/test/real_test_new.mp4"
    video = 0
    radar_image = "radar_example.jpg"

    # Camera resolution
    RESOLUTION_WIDTH = 1080	
    RESOLUTION_HEIGHT = 1920

    # Constants for testing (DEFAULT VARIABLES)
    KNOWN_DISTANCE = 100 # Distance from camera to object (drone) measured in centimeters 
    KNOWN_WIDTH = 30 # Width of the drone in the real world measured in centimeters

    # Measured drone width using camera calibration
    drone_width_pixels = 150  # pixels

    run_radar_project(custom_model, "radar_example.jpg", video, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, KNOWN_DISTANCE, KNOWN_WIDTH, drone_width_pixels)
