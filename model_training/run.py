from utils.track import track
from utils.setup_camera import setup_program

"""
    - Constants for TRUST WEBCAM 2K
    KNOWN_DISTANCE = 100 # Distance from camera to object (drone) measured in centimeters 
    KNOWN_WIDTH = 30 # Width of the drone in the real world measured in centimeters

    - Camera resolution
    RESOLUTION_WIDTH = 2560
    RESOLUTION_HEIGHT = 1440

    - Measured drone width in pixels from camera.
    drone_width_pixels = 490  # pixels
"""

if __name__ == "__main__":

    # Prompt the user to enter width of the drone in real life. CM
    print("Please enter width of the drone in real life")
    choice = int(input("Size of drone in CM: "))

    # Constants for testing (DEFAULT VARIABLE)
    KNOWN_DISTANCE = 100 # Distance from camera to object (drone) measured in centimeters 
    KNOWN_WIDTH = choice # Width of the drone in the real world measured in centimeters

    # Setting up and calibration.
    drone_width_pixels, width, height = setup_program()

    custom_model = "./model/train4/weights/best.pt"
    
    if drone_width_pixels is not None:
        print(f"Drone width: {drone_width_pixels} pixels")
        print(f"Chosen resolution: {width}x{height}")

    track(custom_model, drone_width_pixels, KNOWN_DISTANCE, KNOWN_WIDTH,  width, height)

