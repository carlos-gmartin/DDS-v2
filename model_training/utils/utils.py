import cv2
import math

def measure_drone_width(image):
    # Display the image
    cv2.imshow('Image', image)
    
    # Define a callback function to get mouse click events
    def on_mouse(event, x, y, flags, param):
        nonlocal click_count, points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_count < 2:
                # Record the mouse click coordinates
                points.append((x, y))
                click_count += 1
                print(f"Point {click_count} selected: ({x}, {y})")
                
                # Draw a circle at the clicked point
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Image', image)
                
                if click_count == 2:
                    # Calculate the distance between the two points
                    width_pixels = abs(points[1][0] - points[0][0])
                    print(f"Width of the drone in pixels: {width_pixels}")
                    
                    # Close the image window
                    cv2.destroyAllWindows()
        
    # Initialize variables
    click_count = 0
    points = []
    
    # Set mouse callback function
    cv2.setMouseCallback('Image', on_mouse)
    
    # Wait for the user to select two points
    cv2.waitKey(0)

def take_picture():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        return
    
    # Release camera
    cap.release()
    
    return frame

def check_camera_resolution():
    """
    Check the resolution of the default camera.
    """
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None

    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame")
        cap.release()
        return None

    # Get the resolution of the captured frame
    height, width, _ = frame.shape

    # Release the camera
    cap.release()

    return width, height


if __name__ == "__main__":
    resolution = check_camera_resolution()
    if resolution:
        print(f"Camera resolution: {resolution[0]}x{resolution[1]}")
