import os
import shutil
from PIL import Image

def resize_image(input_path, output_path, size=(256, 256)):
    # Open the image file
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        # Resize the image
        resized_img = img.resize(size)
        # Save the resized image
        resized_img.save(output_path)

def copy_and_rename_images(src_directory, dest_directory, start_iterator=1):
    # Ensure the destination directory exists
    os.makedirs(dest_directory, exist_ok=True)
    
    iterator = start_iterator

    # Iterate through the source directory
    for filename in os.listdir(src_directory):
        if filename.endswith('.jpg'):
            # Generate the new filename with a zero-padded iteration value
            new_filename = f"{iterator:05d}.jpg"
            
            # Create the full paths for source and destination
            src_path = os.path.join(src_directory, filename)
            dest_path = os.path.join(dest_directory, new_filename)

            # Copy the file to the destination with the new name
            resize_image(src_path, dest_path)

            # Increment the iterator for the next file
            iterator += 1
        
    return iterator

if __name__ == "__main__":
    # List of source directories
    source_directories = [
        "/localdisk0/UCF-QNRF_ECCV18/Test",
        "/localdisk0/UCF-QNRF_ECCV18/Train",
        "/localdisk0/NWPU-Crowd/images",
        "/localdisk0/jhu_crowd_v2.0/test/images",
        "/localdisk0/jhu_crowd_v2.0/train/images",
        "/localdisk0/jhu_crowd_v2.0/val/images"
    ]

    # Destination directory
    dest_directory = "/localdisk0/JHU_UCF_NWPU/"

    start_iterator = 0
    # Iterate through each source directory
    for source_directory in source_directories:
        
        print("Copying from: ", source_directory)

        # Copy and rename images from the source to the destination subdirectory
        start_iterator = copy_and_rename_images(source_directory, dest_directory, start_iterator=start_iterator)

    print("Images copied and renamed successfully.")
