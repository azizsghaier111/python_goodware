import os
import piexif
import piexif.helper
import numpy as np
from PIL import Image

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
    if exif_key in exif_dict['0th']:
        del exif_dict['0th'][exif_key]
    write_exif(file_path, exif_dict)

def list_all_exif_data(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag in exif_dict[ifd]:
            print(f"{piexif.TAGS[ifd][tag]['name']}, Value: {exif_dict[ifd][tag]}")

if __name__ == '__main__':
    source_file = 'path_to_source_file.jpg'
    target_file = 'path_to_target_file.jpg'
    exif_data = {
        "DateTime": "2022:01:01 00:00:00",
        "XPAuthor": 'Author Name'
    }
    check_file(source_file)
    check_file(target_file)
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
    list_all_exif_data(target_file)