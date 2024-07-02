import os
import numpy as np
import piexif
from PIL import Image

def load_image(file_path):
    img = Image.open(file_path)
    return img

def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')

def clone_exif_data(source_file, target_file):
    check_file(source_file)
    check_file(target_file)
    source_image = load_image(source_file)
    source_exif_data = piexif.load(source_image.info['exif'])
    target_image = load_image(target_file)
    new_exif_bytes = piexif.dump(source_exif_data)
    target_image.save(target_file, exif=new_exif_bytes)

def remove_exif_data(file_path):
    check_file(file_path)
    image = load_image(file_path)
    new_exif_bytes = piexif.dump({})
    image.save(file_path, exif=new_exif_bytes)

def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])
    for ifd in exif_data:
        if exif_key in exif_data[ifd]:
            del exif_data[ifd][exif_key]
    new_exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=new_exif_bytes)

def extract_thumbnail(file_path):
    check_file(file_path)
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])
    thumbnail = exif_data.pop("thumbnail")
    if thumbnail is not None:
        with open(f"{file_path}_thumbnail.jpg", "wb") as f:
            f.write(thumbnail)

def read_exif_data(file_path):
    check_file(file_path)
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])
    return exif_data

def get_camera_model(file_path):
    check_file(file_path)
    image = load_image(file_path)
    exif_data = piexif.load(image.info['exif'])
    camera_model = exif_data['0th'][piexif.ImageIFD.Model]
    return camera_model

def write_exif_data(file_path, exif_data):
    check_file(file_path)
    image = load_image(file_path)
    new_exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=new_exif_bytes)

if __name__ == '__main__':
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'

    clone_exif_data(source_file_path, target_file_path)
    remove_exif_data(target_file_path)
    
    exif_key = np.random.choice(list(piexif.TAGS.keys()))
    remove_specific_exif_data(target_file_path, exif_key)
    
    extract_thumbnail(source_file_path)
    
    exif_data = read_exif_data(source_file_path)
    exif_data["0th"][piexif.ImageIFD.Artist] = u"Your Name"
    write_exif_data(target_file_path, exif_data)
    
    camera_model = get_camera_model(source_file_path)
    print(f"The camera model is {camera_model}")