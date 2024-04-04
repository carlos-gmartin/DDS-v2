import cv2

def measure_drone_width(image):
    """
    Getting the pixels distance between two points chosen by the user.
    """
    # Display the image
    cv2.imshow('Image', image)
    
    # Initialize variables
    click_count = 0
    points = []
    width_pixels = None
    
    # Define a callback function to get mouse click events
    def on_mouse(event, x, y, flags, param):
        nonlocal click_count, points, width_pixels
        
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
    
    # Set mouse callback function
    cv2.setMouseCallback('Image', on_mouse)
    
    # Wait for the user to select two points
    cv2.waitKey(0)
    
    # Return the measured width
    return width_pixels

def check_camera_resolution():
    """
    Check the resolution of the default camera. Without making any changes.
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

def take_picture(width, height):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Capture frame
    print("Press space bar to take the picture.")
    print(" * DISCLAMER * Image appearing on screen is not 1:1 to the chosen resolution.")
    while True:
        ret, frame = cap.read()  
        resized_img = cv2.resize(frame, (640, 480))

        cv2.imshow('Press Space Bar to Capture', resized_img)
        if cv2.waitKey(1) == ord(' '):
            break

    # Release camera
    cv2.destroyAllWindows()
    cap.release()
    return frame

def setup_program():
    """
    Complete setup returning the neccesary calibration for model tracking.
    """
    width, height = 640, 480

    # Prompt the user to select the desired resolution
    print("Please select the resolution for capturing the picture:")
    print("1. 640x480")
    print("2. 1280x720")
    print("3. 1920x1080")
    print("4. 2560x1440")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == '1':
        width, height = 640, 480
    elif choice == '2':
        width, height = 1280, 720
    elif choice == '3':
        width, height = 1920, 1080
    elif choice == '4':
        width, height = 2560, 1440
    else:
        print("Invalid choice. Using default resolution (640x480).")
        width, height = 640, 480
    
    print("Please hold up drone at a distance of 1 meter. Press space bar to take the picture.")

    image = take_picture(width, height)
    if image is not None:
        print("Picture taken successfully.")
        # Measure drone width in the picture
        drone_width = measure_drone_width(image)
        return drone_width, width, height
    
if __name__ == "__main__":
    drone_width, width, height = setup_program()
    
    if drone_width is not None:
        print(f"Drone width: {drone_width} pixels")
        print(f"Chosen resolution: {width}x{height}")
