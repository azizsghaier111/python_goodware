import os
import numpy as np
import piexif
import mock
import pytorch_lightning
from PIL import Image
import copy

# Load image function
def load_image(file_path):
    img = Image.open(file_path)
    return img

def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')

def clone_exif_data(source_file, target_file):
    check_file(source_file)
    check_file(target_file)

    # Load source image and get its EXIF data
    source_image = load_image(source_file)
    source_exif_data = piexif.load(source_image.info['exif'])

    # Load target image
    target_image = load_image(target_file)

    # Generate new EXIF bytes
    new_exif_bytes = piexif.dump(source_exif_data)

    # Update target image with cloned EXIF data
    target_image.save(target_file, exif=new_exif_bytes)

def remove_exif_data(file_path):
    check_file(file_path)

    # Load image
    image = load_image(file_path)

    # Generate new EXIF bytes with empty data to remove EXIF
    new_exif_bytes = piexif.dump({})

    # Save image with new EXIF bytes
    image.save(file_path, exif=new_exif_bytes)

def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)

    # Load image and get its EXIF data
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])

    # Loop over EXIF items and remove specific keys
    for ifd in exif_data:
        if exif_key in exif_data[ifd]:
            del exif_data[ifd][exif_key]

    # Generate new EXIF bytes
    new_exif_bytes = piexif.dump(exif_data)

    # Save image with new EXIF bytes
    image.save(file_path, exif=new_exif_bytes)

# Extract thumbnail from EXIF data
def extract_thumbnail(file_path):
    check_file(file_path)

    # Load image and get its EXIF data
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])

    # Extract thumbnail
    thumbnail = exif_data.pop("thumbnail")

    # Save thumbnail
    if thumbnail is not None:
        with open(f"{file_path}_thumbnail.jpg", "wb") as f:
            f.write(thumbnail)

# Read EXIF data
def read_exif_data(file_path):
    check_file(file_path)

    # Load image and get its EXIF data
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])

    return exif_data

# Write EXIF data
def write_exif_data(file_path, exif_data):
    check_file(file_path)

    # Load image
    image = load_image(file_path)

    # Generate new EXIF bytes
    new_exif_bytes = piexif.dump(exif_data)

    # Save image with new EXIF bytes
    image.save(file_path, exif=new_exif_bytes)

if __name__ == '__main__':
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'

    # Clone EXIF data
    clone_exif_data(source_file_path, target_file_path)

    # Remove EXIF data
    remove_exif_data(target_file_path)

    # Use numpy for generating a random exif key for demo
    exif_key = np.random.choice(list(piexif.TAGS.keys()))

    # Remove specific EXIF data
    remove_specific_exif_data(target_file_path, exif_key)

    # Extract thumbnail
    extract_thumbnail(source_file_path)

    # Read EXIF data
    exif_data = read_exif_data(source_file_path)

    # Modify EXIF data for demo
    exif_data["0th"][piexif.ImageIFD.Artist] = u"Your Name"

    # Write EXIF data
    write_exif_data(target_file_path, exif_data)