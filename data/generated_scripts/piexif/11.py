import os
import piexif
import piexif.helper
from PIL import Image
import numpy as np
import datetime

# Check if the file exists and is valid
def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

# Load EXIF data from the image file
def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

# Write EXIF data to the image file
def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, "jpeg", exif=exif_bytes)

# Get specific focal length from EXIF data
def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]
    return focal_length

# Get specific date and time from EXIF data
def get_exif_datetime(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    datetime_info = exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, "")
    return datetime.datetime.strptime(datetime_info.decode("utf-8"), '%Y:%m:%d %H:%M:%S')

# Copy EXIF data from one file to another
def clone_exif_data(src_file_path, tgt_file_path):
    check_file(src_file_path)
    check_file(tgt_file_path)
    exif_dict = read_exif(src_file_path)
    write_exif(tgt_file_path, exif_dict)

# Remove EXIF data from the image file
def remove_exif_data(file_path):
    check_file(file_path)
    piexif.remove(file_path)

# Remove specific EXIF data
def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for ifd_name in exif_dict.keys():
        if piexif.is_exist(exif_key, exif_dict[ifd_name]):
            del exif_dict[ifd_name][exif_key]
    write_exif(file_path, exif_dict)

# Read an image
file_path = 'file_path'

try:
    exif_dict = read_exif(file_path)
    focal_length = get_exif_focal_length(file_path)
    exif_datetime = get_exif_datetime(file_path)
    print(f"Focal Length: {focal_length}\nDatetime: {exif_datetime}")
    clone_exif_data(file_path, 'new_file_path')
    remove_exif_data(file_path)
    remove_specific_exif_data(file_path, piexif.ExifIFD.FocalLength)
except FileNotFoundError as err:
    print(err)
except Exception as e:
    print(f"Unexpected Error: {e}")