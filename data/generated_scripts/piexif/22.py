import os
import numpy as np
import piexif
from PIL import Image
import pytorch_lightning as pl
import mock

# Validation functions
def is_valid_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')

def was_a_jpeg(file_path):
    try: 
        img = Image.open(file_path)
        img.verify() 
        img.close()
        if img.format != 'JPEG':
            raise ValueError(f'File {file_path} is not a valid JPEG')
    except (IOError, SyntaxError) as e:
        raise ValueError(f'File {file_path} is not a valid image')

# Image functions      
def load_image(file_path):
    is_valid_file(file_path)
    was_a_jpeg(file_path)    
    img = Image.open(file_path)
    return img

def save_image(img, file_path, exif_bytes):
    is_valid_file(file_path)
    was_a_jpeg(file_path)    
    img.save(file_path, exif=exif_bytes)

# Exif functions
def get_exif_from_img(img):
    return piexif.load(img.info['exif'])

def get_exif_from_img_path(img_path):
    img = load_image(img_path)
    return get_exif_from_img(img)

def remove_specific_key_from_exif(exif_data, exif_key):
    for ifd in exif_data:
        if isinstance(exif_data[ifd], dict):
            if exif_key in exif_data[ifd]:
                del exif_data[ifd][exif_key]
    return exif_data

def get_bytes_from_exif(exif_data):
    return piexif.dump(exif_data)

def get_camera_model(exif_data):
    try:
        return exif_data['0th'][272].decode('utf-8')
    except KeyError:
        raise Exception("Camera model not found in EXIF data")

def get_focal_length(exif_data):
    try:
        return exif_data['Exif'][37386][0]/exif_data['Exif'][37386][1]
    except KeyError:
        raise Exception("Focal length not found in EXIF data")

# top-level functions
def clone_exif_data(source_file_path, target_file_path):
    source_exif_data = get_exif_from_img_path(source_file_path)
    target_image = load_image(target_file_path)
    new_exif_bytes = get_bytes_from_exif(source_exif_data)
    save_image(target_image, target_file_path, new_exif_bytes)

def remove_exif_data(file_path):
    empty_exif_data = piexif.load({})
    exif_bytes = get_bytes_from_exif(empty_exif_data)
    img = load_image(file_path)
    save_image(img, file_path, exif_bytes)

def remove_specific_exif_data(file_path, exif_key):
    img = load_image(file_path)
    exif_data = get_exif_from_img(img)
    exif_data = remove_specific_key_from_exif(exif_data, exif_key)
    exif_bytes = get_bytes_from_exif(exif_data)
    save_image(img, file_path, exif_bytes)

# Main code
if __name__ == '__main__':
    input_file_path = 'path_to_your_input_file'
    target_file_path = 'path_to_your_target_file'
    exif_data = get_exif_from_img_path(input_file_path)

    # print camera model
    camera_model = get_camera_model(exif_data)
    print(f'Camera Model: {camera_model}')

    # print focal length
    focal_length = get_focal_length(exif_data)
    print(f'Focal Length: {focal_length}')

    # clone EXIF data
    clone_exif_data(input_file_path, target_file_path)

    # remove EXIF data
    remove_exif_data(target_file_path)

    # remove specific EXIF data
    remove_specific_exif_data(target_file_path, 42036)