import os
import piexif
from PIL import Image
import numpy as np

def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

def read_exif(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)

def modify_exif_data(file_path, data_dict):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for key, value in data_dict.items():
        exif_dict["0th"][getattr(piexif.ImageIFD, key)] = value
    write_exif(file_path, exif_dict)

def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]
    return focal_length

def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    exif_dict = read_exif(source_file_path)
    write_exif(target_file_path, exif_dict)

def remove_exif_data(file_path):
    check_file(file_path)
    piexif.remove(file_path)

def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    # Remove the exif_key from all IFDs
    for ifd in ['0th', 'Exif', 'GPS', '1st']:
        if exif_key in exif_dict[ifd]:
            del exif_dict[ifd][exif_key]
    write_exif(file_path, exif_dict)

def list_all_exif_data(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag in exif_dict[ifd]:
            print(f"{piexif.TAGS[ifd][tag]['name']}, Value: {exif_dict[ifd][tag]}")

# Function to add thumbnail to EXIF Data
def add_thumbnail_to_exif(file_path, thumb_path):
    check_file(file_path)
    check_file(thumb_path)
    thumbnail_img = Image.open(thumb_path)
    thumbnail_img.thumbnail((100, 100))
    thumb_bytes = thumbnail_img.tobytes("jpeg", "RGB")
    exif_dict = read_exif(file_path)
    exif_dict['thumbnail'] = thumb_bytes
    write_exif(file_path, exif_dict)

if __name__ == '__main__':
    source_file = 'path_to_source_file.jpg'
    target_file = 'path_to_target_file.jpg'
    exif_data = {
        "DateTime": "2022:01:01 00:00:00",
        "XPAuthor": 'Author Name'
    }
    thumbnail_path = 'path_to_thumbnail.jpg'
    check_file(source_file)
    check_file(target_file)
    check_file(thumbnail_path)
    exif_dict = read_exif(source_file)
    write_exif(source_file, exif_dict)
    focal_length = get_exif_focal_length(source_file)
    print(f"Focal length: {focal_length}")
    modify_exif_data(source_file, exif_data)
    clone_exif_data(source_file, target_file)
    remove_exif_data(target_file)
    exif_key = np.random.choice(list(exif_dict['0th'].keys()))
    remove_specific_exif_data(target_file, exif_key)
    list_all_exif_data(source_file)
    thumbnail_path = 'path_to_thumbnail.jpg'
    add_thumbnail_to_exif(target_file, thumbnail_path)
    list_all_exif_data(target_file)