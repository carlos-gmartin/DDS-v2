import cv2
import numpy as np

# Function to calculate and draw the angle on the image
def draw_calculated_angle(img, center, distance_from_center, frame_width, length=100, color=(0, 255, 0), thickness=2):
    angle_radians = np.arctan(distance_from_center / (frame_width / 2))
    angle_degrees = np.degrees(angle_radians)  # Convert to degrees
    x_end = int(center[0] + length * np.cos(angle_radians))
    y_end = int(center[1] - length * np.sin(angle_radians))
    cv2.line(img, center, (x_end, y_end), color, thickness)
    cv2.putText(img, f"{angle_degrees:.2f} degrees", (x_end, y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# Load the image
img = cv2.imread('drone_distance.jpg')

# Define the center of the frame (usually the center of the image)
center = (img.shape[1] // 2, img.shape[0] // 2)

# Frame width
frame_width = img.shape[1]

# Example distances from center to calculate angles
distances_from_center = [0, 50, 100, 150, 200, 250, 300]

# Draw calculated angles
for distance in distances_from_center:
    draw_calculated_angle(img, center, distance, frame_width)

# Display the image
cv2.imshow('Image with Calculated Angles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('image_with_calculated_angles.jpg', img)
