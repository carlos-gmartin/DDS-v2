import numpy as np
import cv2
import time

# Set the width and height of the radar grid
width = 1335
height = 800
radius = 30

# Load the example grid image
img = cv2.imread('example_grid.jpg')

def pos_angle(base_angle, length):
    pt = (width // 2 + int(length * np.sin(np.radians(base_angle))), height + int(length * np.cos(np.radians(base_angle))))
    return pt

def put_degree(txt, shift=0, level=0, txt_shift=0, txt_level=0):
    cv2.putText(img, txt, org=pos_angle(int(txt) + 90 + txt_shift, 353 + txt_level), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.circle(img, center=pos_angle(int(txt) + 90 + shift, 378 + level), radius=2, color=(0, 255, 0))

for base_angle in range(90, 300, 30):
    pt1 = (width // 2 + int(350 * np.sin(np.radians(base_angle))), height + int(350 * np.cos(np.radians(base_angle))))
    cv2.line(img, pt1=pt1, pt2=(width // 2, height), color=[0, 255, 0], thickness=2)

for i in range(1, 1000, 78):
    cv2.circle(img, (width // 2, height), i, color=[0, 255, 0])

put_degree('30')
put_degree('60', -2, -2)
put_degree('90', -4, -13)
put_degree('120', -3.5, -18, 2, 15)
put_degree('150', -2, -20, 2, 28)

for n, i in enumerate(range(0, 300, 78)):
    cv2.putText(img, f'{10*(n+1)}cm', org=pos_angle(90, 50+i), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,color=(0, 255, 0), thickness=1)

def add_object(distance, angle, img_copy):
    if distance < 40:
        distance = np.interp(distance, [0, 40], [0, 350])
        pt1 = (width // 2 + int(distance * np.sin(np.radians(angle))), height + int(distance * np.cos(np.radians(angle))))
        cv2.circle(img_copy, center=pt1, radius=5, color=[255, 0, 0], thickness=-1)
        return pt1

# Initialize the flag to track whether the mouse button is pressed
mouse_pressed = False

# Initialize the point coordinates list
point_coordinates = []

# Define the callback function for mouse events
def on_mouse(event, x, y, flags, param):
    global mouse_pressed, point_coordinates

    # If the left mouse button is pressed, save the point coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        point_coordinates.append((x, y))

    # If the left mouse button is released, add the point to the radar grid
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        if point_coordinates:
            pt1 = add_object(np.linalg.norm(np.array(point_coordinates[-1])- np.array((width // 2, height))),
                              np.arctan2(point_coordinates[-1][1] - height, point_coordinates[-1][0] - (width // 2)) * 180 / np.pi,
                              img.copy())
            point_coordinates.append(pt1)

# Display the radar grid and the added points
cv2.imshow('Radar',img)

# Set the callback function for the 'Radar' window
cv2.setMouseCallback('Radar', on_mouse)

while True:
    # Create a copy of the img variable
    img_copy = img.copy()

    # Add objects dynamically
    for pt in point_coordinates:
        cv2.circle(img_copy, center=pt, radius=5, color=[255, 0, 0], thickness=-1)

    # Display the radar grid and the added points
    cv2.imshow('Radar', img_copy)

    # Check for user input
    key = cv2.waitKey(100)
    if key == 27:  # Exit on ESC key
        break

cv2.destroyAllWindows()