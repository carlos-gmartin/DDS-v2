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

    FOV is 90 degrees.
"""

"""
    - Constants for Google pixel 8 ULTRAWIDE LENS
    KNOWN_DISTANCE = 100 # Distance from camera to object (drone) measured in centimeters 
    KNOWN_WIDTH = 30 # Width of the drone in the real world measured in centimeters

    - Camera resolution
    RESOLUTION_WIDTH = 2560
    RESOLUTION_HEIGHT = 1440

    - Measured drone width in pixels from camera.
    drone_width_pixels = 52  # pixels

    FOV is 125 degrees.
"""


if __name__ == "__main__":
    # # Prompt the user to enter width of the drone in real life. CM
    # print("Please enter width of the drone in real life")
    # choice = int(input("Size of drone in CM: "))

    # # Setting up and calibration.
    # drone_width_pixels, width, height = setup_program()
    
    # if drone_width_pixels is not None:
    #     print(f"Drone width: {drone_width_pixels} pixels")
    #     print(f"Chosen resolution: {width}x{height}")

    # Constants for testing (DEFAULT VARIABLES)
    KNOWN_DISTANCE = 100 # Distance from camera to object (drone) measured in centimeters 
    KNOWN_WIDTH = 30 # Width of the drone in the real world measured in centimeters

    # Camera resolution
    RESOLUTION_WIDTH = 2560
    RESOLUTION_HEIGHT = 1440

    # Measured drone width using camera calibration
    drone_width_pixels = 150  # pixels

    # Load custom YOLO model
    custom_model = "./model/train4/weights/best.pt"

    video = "./test/real_test_new.mp4"
    #video = 0

    # Save tracking file path
    info_file = "../interface/detection_info.txt"

    # Running the model.
    track(custom_model, drone_width_pixels, KNOWN_DISTANCE, KNOWN_WIDTH, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, video, info_file)


