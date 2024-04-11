import cv2

def measure_distance(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Display the image
    cv2.imshow('Image', image)
    
    # Initialize variables
    click_count = 0
    points = []
    distance_pixels = None
    
    # Define a callback function to get mouse click events
    def on_mouse(event, x, y, flags, param):
        nonlocal click_count, points, distance_pixels
        
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
                    distance_pixels = ((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2) ** 0.5
                    print(f"Distance between the two points in pixels: {distance_pixels}")
                    # Close the image window
                    cv2.destroyAllWindows()
    
    # Set mouse callback function
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', on_mouse)
    
    # Wait for the user to select two points
    cv2.waitKey(0)
    
    # Return the measured distance
    return distance_pixels

if __name__ == "__main__":
    # Provide the path to the image
    image_path = 'distance.jpg'  # Change this to the path of your image
    distance = measure_distance(image_path)
    if distance is not None:
        print(f"Distance between the two points: {distance} pixels")