import os

def rename_sound_files(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter out only sound files (you may need to adjust this depending on your file extensions)
    sound_files = [file for file in files if file.endswith('.mp3') or file.endswith('.wav')]

    # Sort the sound files to ensure linear incrementation
    sound_files.sort()

    # Rename each sound file with "Drone" and linear incrementation
    for i, file in enumerate(sound_files):
        # Extract the file extension
        _, ext = os.path.splitext(file)

        # Construct the new file name
        new_name = f'Drone{i + 1}{ext}'

        # Rename the file
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
        print(f'Renamed {file} to {new_name}')

# Specify the directory containing the sound files
directory = './soundDataset/'

# Call the function to rename the sound files
rename_sound_files('drone/')