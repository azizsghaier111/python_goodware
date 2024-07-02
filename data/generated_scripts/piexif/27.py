# Import necessary libraries
import os
import piexif
import piexif.helper
import numpy as np
from PIL import Image

# Function for checking if the file exists or not
def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

# Function for reading EXIF data from the image
def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

# Function for writing EXIF data to the image
def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)  # Dumping the EXIF data
    img.save(file_path, exif=exif_bytes)  # Saving the image with new EXIF data

# Function to get the focal length from EXIF data
def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]
    return focal_length

# Function to clone EXIF data
def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    exif_dict = read_exif(source_file_path)  # Reading EXIF data from source
    write_exif(target_file_path, exif_dict)  # Writing EXIF data to target

# Function to remove all EXIF data
def remove_exif_data(file_path):
    check_file(file_path)
    piexif.remove(file_path)  # Removing all EXIF data

# Function to remove specific EXIF data
def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    exif_dict = read_exif(file_path)  # Reading EXIF data
    for ifd in ('0th', 'Exif', 'GPS', '1st'):
        if exif_key in exif_dict[ifd]:
            del exif_dict[ifd][exif_key]  # Deleting specific EXIF data
    write_exif(file_path, exif_dict)  # Writing the new EXIF data back to the image

# Function to get GPS info from EXIF data
def get_gps_info(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)  # Reading EXIF data
    gps_info = piexif.helper.UserComment.load(exif_dict.get('Exif').get(piexif.ExifIFD.UserComment)).get('GPS')
    return gps_info

if __name__ == '__main__':
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'

    # Load EXIF data
    exif_dict = read_exif(source_file_path)

    # Write EXIF data back to the file
    write_exif(source_file_path, exif_dict)

    # Get EXIF focal length
    focal_length = get_exif_focal_length(source_file_path)
    print(f"Focal length: {focal_length}")

    # Clone EXIF data
    clone_exif_data(source_file_path, target_file_path)

    # Remove EXIF data
    remove_exif_data(target_file_path)

    # Use numpy for generating a random EXIF key for demonstration
    exif_key = np.random.choice(list(exif_dict['0th'].keys()))

    # Remove specific EXIF data
    remove_specific_exif_data(target_file_path, exif_key)

    # Get GPS info from EXIF data
    gps_info = get_gps_info(source_file_path)
    print(f"GPS info: {gps_info}")