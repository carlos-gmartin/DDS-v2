import os
import random
import shutil
import xml.etree.ElementTree as ET

def prepareTrain(train_i, img_paths, ano_paths, dataset, train_path):
    for i in train_i:
        img_path = img_paths[i]
        ano_path = ano_paths[i]
        try:
            shutil.copy(ano_path, train_path)
            shutil.copy(img_path, train_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return len(os.listdir(train_path))

def prepareValid(valid_i, img_paths, ano_paths, dataset, valid_path):
    for i in valid_i:
        img_path = img_paths[i]
        ano_path = ano_paths[i]
        try:
            shutil.copy(ano_path, valid_path)
            shutil.copy(img_path, valid_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return len(os.listdir(valid_path))

def prepareTest(test_i, img_paths, ano_paths, dataset, test_path):
    for i in test_i:
        img_path = img_paths[i]
        ano_path = ano_paths[i]
        try:
            shutil.copy(ano_path, test_path)
            shutil.copy(img_path, test_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return len(os.listdir(test_path))

def prepareData(dataset, train_path, valid_path, test_path):
    img_paths=[]
    ano_paths=[]
    for dirname, _, filenames in os.walk(dataset):
        for filename in filenames:
            if filename[-4:] == '.jpg':
                img_paths.append(os.path.join(dirname, filename))
            elif filename[-4:] == '.txt':
                ano_paths.append(os.path.join(dirname, filename))
                
    n = min(len(img_paths), len(ano_paths))
    print(n)
    N = list(range(n))
    random.shuffle(N)
    
    # Training split 70/20/10
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1
    
    train_size = int(train_ratio * n)
    valid_size = int(valid_ratio * n)
    
    train_i = N[:train_size]
    valid_i = N[train_size:train_size + valid_size]
    test_i = N[train_size + valid_size:]
    
    print('Train: ' + str(prepareTrain(train_i, img_paths, ano_paths, dataset, train_path)) +
          ' Valid: ' + str(prepareValid(valid_i, img_paths, ano_paths, dataset, valid_path)) +
          ' Test: ' + str(prepareTest(test_i, img_paths, ano_paths, dataset, test_path)))

# Example usage:
dataset = './archive'
train_path = './datasets/train'
valid_path = './datasets/valid'
test_path = './datasets/test'

def fix_annotations(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            annotation_file = os.path.join(folder_path, filename)
            lines_to_keep = []
            with open(annotation_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_index = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            if all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                                lines_to_keep.append(line)
                        except ValueError:
                            pass
            
            # Write the valid lines back to the annotation file
            with open(annotation_file, 'w') as file:
                for line in lines_to_keep:
                    file.write(line)

def fix_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.JPEG') or filename.endswith('.jpg') or filename.endswith('.png'):
            image_file = os.path.join(folder_path, filename)
            # Check if the image file is corrupted or truncated (optional)
            # If necessary, remove or replace the corrupted image file
            # Perform any other required operations on the image files
            
    # Fix the annotations in the folder
    fix_annotations(folder_path)

def convert_txt_files_in_folder(root_folder):
    # Traverse through all folders and subfolders
    for root, _, files in os.walk(root_folder):
        # Iterate through each file
        for file in files:
            # Check if the file is a TXT file
            if file.endswith('.txt'):
                # Get the full path to the TXT file
                txt_file = os.path.join(root, file)
                
                # Read the content of the TXT file
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                # Modify the content to match the desired format
                modified_lines = []
                for line in lines:
                    # Split the line into values
                    values = line.strip().split()
                    
                    # Check if the line contains exactly four values
                    if len(values) == 4:
                        # Parse the existing bounding box coordinates
                        x_center, y_center, box_width, box_height = map(float, values)
                        
                        # Convert the bounding box coordinates to the desired format
                        x_center = 0
                        y_center = x_center + x_center * x_center
                        box_width = y_center + y_center * y_center
                        box_height = box_width + box_width * box_width
                        
                        # Append the modified line to the list
                        modified_lines.append(f"{x_center} {y_center} {box_width} {box_height}\n")
                    else:
                        # Handle lines with incorrect format (optional)
                        print(f"Ignoring line: {line.strip()}")
                
                # Overwrite the TXT file with the modified content
                with open(txt_file, 'w') as f:
                    f.writelines(modified_lines)

def convert_to_yolo_format(xml_content):
    root = ET.fromstring(xml_content)
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    objects = root.findall('object')
    
    yolo_format_lines = []
    for obj in objects:
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        yolo_format_lines.append(f"{name} {x_center} {y_center} {box_width} {box_height}")
    
    return '\n'.join(yolo_format_lines)

def convert_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Check if the file content matches the XML format
            if '<annotation>' in content and '</annotation>' in content:
                # Convert to YOLO format
                yolo_content = convert_to_yolo_format(content)
                
                # Write YOLO format content to the same file
                with open(file_path, 'w') as txt_file:
                    txt_file.write(yolo_content)
                    
                print(f"Converted {filename} to YOLO format.")

def replace_drone_with_0_in_directory(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            # Apply replacement function to each text file
            replace_drone_with_0_in_file(file_path)

def replace_drone_with_0_in_file(file_path):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Modify the lines containing 'drone'
    for i in range(len(lines)):
        words = lines[i].split()  # Split the line into words
        for j in range(len(words)):
            if words[j] == 'drone':
                words[j] = '0'  # Replace 'drone' with '0'
        lines[i] = ' '.join(words)  # Join the modified words back into a line

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the lines containing 'drone'
    for i in range(len(lines)):
        words = lines[i].split()  # Split the line into words
        for j in range(len(words)):
            if words[j] == 'drone':
                words[j] = '0'  # Replace 'drone' with '0'
        lines[i] = ' '.join(words)  # Join the modified words back into a line

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Provide the path to the folder containing the images and annotation files
folder_path = 'archive'

if __name__ == "__main__":
    prepareData(dataset, train_path, valid_path, test_path)
    #replace_drone_with_0_in_directory(folder_path)
    #fix_folder(folder_path)
    #convert_txt_files_in_folder('archive')

    # Convert the .txt files
    #convert_txt_files(folder_path)
