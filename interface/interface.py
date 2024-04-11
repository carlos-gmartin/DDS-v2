from grid import run_project

def get_drone_params(file_path):
    """
    Read drone parameters from a file.

    Parameters:
        file_path (str): The path to the file containing drone parameters.

    Returns:
        list: A list of tuples containing distance and angle of each drone.
    """
    drone_params = []  # Initialize an empty list to store drone parameters

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            distance = float(parts[1].split(':')[1].strip().split()[0])  # Extract distance from line
            angle = float(parts[2].split(':')[1].strip().split()[0])  # Extract angle from line
            drone_params.append((distance, angle))  # Append the drone parameters to the list

    return drone_params

if __name__ == "__main__":
    file_path = "detection_info.txt"  # Replace this with the actual path to your file
    drone_params = get_drone_params(file_path)
    run_project(drone_params)



