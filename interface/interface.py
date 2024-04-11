from grid import run_project
import time
import random

def get_drone_params():
    """
    Generator function to simulate real-time updates of drone parameters.
    This function yields random distance and angle values.
    """
    while True:
        # Simulate some processing time
        time.sleep(1)
        # Generate random distance (between 0 and 180 meters) and angle (between -180 and 180 degrees)
        distance = random.randint(0, 180)
        angle = random.randint(0, 180)
        yield (distance, angle)


if __name__ == "__main__":
    # Running radar
    run_project(get_drone_params)
