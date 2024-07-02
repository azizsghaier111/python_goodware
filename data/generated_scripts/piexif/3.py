import os
import piexif
import piexif.helper
import numpy as np
from PIL import Image

# Read EXIF data from image file
def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])

    return exif_dict

# Write EXIF data to image file
def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)

    # Dump and encode exif_dict to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save image with new EXIF data
    img.save(file_path, exif=exif_bytes)

# Get specific value from EXIF data
def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)

    # Use piexif's predefined constant for FocalLength
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)

    # FocalLength is usually a tuple like (focal_length, 1)
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]

    return focal_length

..
..
..

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
    exif_key = np.random.choice(list(piexif.TAGS.keys()))

    # Remove specific EXIF data
    remove_specific_exif_data(target_file_path, exif_key)