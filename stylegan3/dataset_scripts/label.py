import os
import json

# Path to your image folder
image_folder_path = '/localdisk0/JHU_UCF_NWPU/'

# Dummy label with all zeros except the first index is 1
dummy_label = [0] * 12
dummy_label[0] = 1

# List to store image labels
image_labels = []

# Iterate over the image folder
for root, dirs, files in os.walk(image_folder_path):
    for file in sorted(files):
        if file.lower().endswith(('.jpg', '.jpeg')):
            # Create the image path
            image_path = os.path.join(root, file)
            
            # Append the image path and dummy label to the list
            image_labels.append([file, dummy_label.copy()])

# Create the JSON structure
json_data = {"labels": image_labels}

# Save the JSON to a file
json_filename = 'image_labels.json'
with open(json_filename, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print(f"JSON file '{json_filename}' created successfully.")