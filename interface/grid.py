import numpy as np
import cv2

# Set the width and height of the radar grid
width = 1335
height = 800
radius = 30

# Distance of each radius. 100 pixels ~ 20m
RADIUS = 100

# Global variables
point_coordinates = []
drone_coordinates = []

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
    global img
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
    print(f"Added dot at ({x}, {y}), Angle: {angle}")

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

    # Draw the new drone on the radar grid
    pt1 = (width // 2, height)  # Center of the radar grid
    pt2 = (x, y)  # Coordinates of the drone
    cv2.circle(img_copy, center=pt2, radius=5, color=[50, 50, 50], thickness=-1)  # Draw a circle at the drone position
    cv2.line(img_copy, pt1=pt1, pt2=pt2, color=[50, 50, 50], thickness=1)  # Draw a line from the center to the drone position

    # Draw a green rectangle around the drone
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

    #print(f"Added drone at ({x}, {y}), Angle: {angle}")

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

def run_project(get_drone_params):
    global img

    # Load the example grid image
    img = cv2.imread('example_grid.jpg')

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

    # Display the radar grid and allow manual dot addition
    cv2.imshow('Radar',img)

    for distance, angle in get_drone_params:
        add_drone(distance, angle)

        # Check for user input
        key = cv2.waitKey(100)
        if key == 27:  # Exit on ESC key
            break

    cv2.destroyAllWindows()


# # Testing the function
# if __name__ == "__main__":
#     # Example function to get drone parameters (distance, angle)
#     def get_drone_params():
#         for distance in range(0, 1000, 10):
#             angle = -70  # Constant angle
#             yield (distance, angle)

#     # To run the grid system.
#     run_project(get_drone_params)

