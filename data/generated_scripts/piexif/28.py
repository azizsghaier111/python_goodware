import os
import piexif
import piexif.helper
from PIL import Image

# A function to verify a file exists before running rest of the operations
def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

# Function to read EXIF data from a file
def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

# Function to write EXIF data to a file
def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)

# Function to modify certain EXIF data of a file
def modify_exif_data(file_path, data_dict):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for key, value in data_dict.items():
        exif_dict["0th"][getattr(piexif.ImageIFD, key)] = value
    write_exif(file_path, exif_dict)

# Function to retrieve the focal length from EXIF data
def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength, None)
    return focal_length

# Function to remove entire EXIF data from a file
def remove_exif_data(file_path):
    check_file(file_path)
    piexif.remove(file_path)

# Above is the actual python script as per your requirement

if __name__ == '__main__':
    # Replace with your file path here
    file_path = 'path_to_your_file.jpg' 
    # Make sure the file exists
    check_file(file_path)

    # Reading and displaying EXIF data
    exif_dict = read_exif(file_path)
    print('EXIF data before modification: ', exif_dict)

    # Modifying EXIF data
    modify_data = {"DateTime": "2022:01:01 00:00:00"}
    modify_exif_data(file_path, modify_data)

    # Reading and displaying modified EXIF data
    modified_exif_dict = read_exif(file_path)
    print('EXIF data after modification: ', modified_exif_dict)

    # Get EXIF focal length
    print('Focal length: ', get_exif_focal_length(file_path))

    # Remove EXIF data
    remove_exif_data(file_path)

    # Check if EXIF data is removed
    exif_dict_after_removal = read_exif(file_path)
    print('EXIF data after removal: ', exif_dict_after_removal)