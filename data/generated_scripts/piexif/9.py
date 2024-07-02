import os
import piexif
import piexif.helper
import numpy as np
import random
from PIL import Image

# Initial check if the file is valid
def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

# Extracting EXIF data from the given image file
def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

# Overwriting or adding EXIF data to the given image file
def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)

# Getting the focal length from the EXIF data of the given image file
def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
    
    # Unpack the focal length if it's a tuple
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]
    return focal_length

# Testing all the functions above with the given source file and a target file
def test_functions(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    exif_dict = read_exif(source_file_path)
    write_exif(source_file_path, exif_dict)
    focal_length = get_exif_focal_length(source_file_path)
    print(f"Focal length: {focal_length}")
    clone_exif_data(source_file_path, target_file_path)
    remove_exif_data(target_file_path)
    exif_key = np.random.choice(list(exif_dict['0th'].keys()))
    remove_specific_exif_data(target_file_path, exif_key)

# Cloning EXIF data from the source to the target
def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    exif_dict = read_exif(source_file_path)
    write_exif(target_file_path, exif_dict)

# Removing EXIF data from the given image file
def remove_exif_data(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    piexif.remove(file_path)

# Specific EXIF data removal from the given image
def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for ifd_name in exif_dict:
        if exif_key in exif_dict[ifd_name]:
            del exif_dict[ifd_name][exif_key]
    write_exif(file_path, exif_dict)

if __name__ == '__main__':
    source_file = 'path_to_source_file.jpg'
    target_file = 'path_to_target_file.jpg'
    test_functions(source_file, target_file)