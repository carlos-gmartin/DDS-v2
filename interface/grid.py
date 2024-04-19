import cv2
import numpy as np
import os
import time

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
    cv2.putText(img, txt, org=pos_angle(int(txt) + 90 + txt_shift, 353 + txt_level), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    # Draw a small circle at the degree label position
    cv2.circle(img, center=pos_angle(int(txt) + 90 + shift, 378 + level), radius=2, color=(0, 255, 0))

def add_dot(x, y, angle):
    """
    Add a point to the radar grid.

    Parameters:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        angle (float): The angle of the point from the center (in degrees).
    """
    global img, points_added, squares
    pt1 = (width // 2, height)  # Center of the radar grid
    pt2 = (x, y)  # Coordinates of the clicked point
    # Draw a circle at the clicked point
    cv2.circle(img, center=pt2, radius=5, color=[0, 0, 250], thickness=-1)
    # Draw a line from the center to the clicked point
    # cv2.line(img, pt1=pt1, pt2=pt2, color=[8,255,8], thickness=1)
    # Show the updated radar grid
    cv2.imshow('Radar', img)
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
    global img

    # Find the minimum and maximum x and y coordinates of the drawn points
    min_x = min(square_coordinates, key=lambda x: x[0])[0]
    max_x = max(square_coordinates, key=lambda x: x[0])[0]
    min_y = min(square_coordinates, key=lambda x: x[1])[1]
    max_y = max(square_coordinates, key=lambda x: x[1])[1]

    # Add points to define the square
    square_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    squares.append(square_points)  # Save the coordinates of the square

    # Draw the square on the grid
    cv2.polylines(img, [np.array(square_points)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow('Radar', img)

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
    global img

    # Clear the previous drone circle
    img_copy = img.copy()  # Make a copy of the image to preserve the original

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
    global img

    # Load the example grid image
    img = cv2.imread(example_img)

    # Set mouse callback function
    cv2.namedWindow('Radar')
    cv2.setMouseCallback('Radar', on_mouse)

    for base_angle in range(90, 300, 30):
        pt1 = (width // 2 + int(350 * np.sin(np.radians(base_angle))), height + int(350 * np.cos(np.radians(base_angle))))
        cv2.line(img, pt1=pt1, pt2=(width // 2, height), color=[0, 255, 0], thickness=2)
    
    # Draw the distance lines.
    for i in range(1, 1000, RADIUS):
        cv2.circle(img, (width // 2, height), i, color=[0, 255, 0])

    put_degree('30')
    put_degree('60', -2, -2)
    put_degree('90', -4, -13)
    put_degree('120', -3.5, -18, 2, 15)
    put_degree('150', -2, -20, 2, 28)

    for n, i in enumerate(range(0, 1000, 100)):
        cv2.putText(img, f'{20*(n+1)}m', org=pos_angle(90, 50+i), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,color=(0, 255, 0), thickness=1)

    cv2.imshow('Radar',img)

def run_radar_project(image, detection_file):
    global img

    # Start default grid.
    start_grid(image)
    
    # Restarting the count
    index = 0 

    # Read drone parameters from the file
    drone_params_generator = get_drone_params(index, detection_file)

    while True:
        drone_params_generator = get_drone_params(index, detection_file)
        while True:
            # Check if the detection file exists
            if os.path.exists(detection_file):
                try:
                    # Attempt to get the next drone parameters
                    distance, angle = next(drone_params_generator)
                    add_drone(distance, angle)
                    index += 1
                except StopIteration:
                    # If there are no more drone parameters, wait for a while before checking again
                    print("No more drone parameters. Waiting for new ones...")
                    time.sleep(5)  # Adjust the delay time as needed
                    break  # Exit the inner loop to reset the timer and try again
            else:
                # If the detection file doesn't exist, wait for a while before checking again
                print("Detection file not found. Waiting for it to be created...")
                time.sleep(1)  # Adjust the delay time as needed
                break  # Exit the inner loop to reset the timer and try again

            # Check for user input
            key = cv2.waitKey(100)
            if key == 27:  # Exit on ESC key
                break  # Exit the inner loop and stop the program

        if key == 27:  # Exit on ESC key
            break  # Exit the outer loop and stop the program

    cv2.destroyAllWindows()

def get_drone_params(start_index, file_path):
    """
    Read drone parameters from a file.

    Parameters:
        start_index (int): The line number to start reading from.
        file_path (str): The path to the file containing drone parameters.

    Yields:
        tuple: A tuple containing distance and angle of each drone.
    """

    with open(file_path, 'r') as file:
        current_line = 0
        for line in file:
            if current_line >= start_index:
                parts = line.strip().split(',')
                distance = float(parts[1].split(':')[1].strip().split()[0])  # Extract distance from line
                angle = float(parts[2].split(':')[1].strip().split()[0])  # Extract angle from line
                yield distance, angle  # Yield the drone parameters one by one
            current_line += 1

# # Run the radar project
# run_radar_project("example_grid", "detection_info.txt")