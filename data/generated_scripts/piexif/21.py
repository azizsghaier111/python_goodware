import os
import numpy as np
import piexif
from PIL import Image
from typing import Union


def load_image(file_path):
    return Image.open(file_path)


def check_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found')


def clone_exif_data(source_file: str, target_file: str):
    check_file(source_file)
    check_file(target_file)
    
    source_image = load_image(source_file)
    source_exif_data = piexif.load(source_image.info.get('exif', b''))
    
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
    exif_data = piexif.load(image.info.get('exif', b''))
    
    for ifd in exif_data:
        if exif_key in exif_data[ifd]:
            del exif_data[ifd][exif_key]
    
    new_exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=new_exif_bytes)


def extract_thumbnail(file_path):
    check_file(file_path)
    
    image = load_image(file_path)
    exif_data = piexif.load(image.info.get('exif', b''))
    
    thumbnail = exif_data.pop("thumbnail", None)
    
    if thumbnail is not None:
        with open(f"{file_path}_thumbnail.jpg", "wb") as f:
            f.write(thumbnail)


def read_exif_data(file_path):
    check_file(file_path)
    
    image = load_image(file_path)
    exif_data = piexif.load(image.info.get('exif', b''))
    
    return exif_data


def get_camera_model(file_path):
    check_file(file_path)
    
    image = load_image(file_path)
    exif_data = piexif.load(image.info.get('exif', b''))
    
    return exif_data.get('0th', {}).get(piexif.ImageIFD.Model)


def write_exif_data(file_path: str, exif_data: Union[str, bytes]):
    check_file(file_path)
    
    image = load_image(file_path)
    
    new_exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=new_exif_bytes)


if __name__ == '__main__':
    source_file_path = '' # provide path for your source file
    target_file_path = '' # provide path for your target file
    new_exif_data = ''    # provide new exif data

    assert source_file_path, "Please provide a valid path for the source file!"
    assert target_file_path, "Please provide a valid path for the target file!"
    assert new_exif_data, "Please provide valid exif data!"

    clone_exif_data(source_file_path, target_file_path)
    remove_exif_data(target_file_path)
    
    exif_key = np.random.choice(list(piexif.TAGS.keys()))
    remove_specific_exif_data(target_file_path, exif_key)
    
    extract_thumbnail(source_file_path)
    
    exif_data = read_exif_data(source_file_path)
    exif_data["0th"][piexif.ImageIFD.Artist] = 'Your Name'
    write_exif_data(target_file_path, exif_data)
    
    camera_model = get_camera_model(source_file_path)
    print(f"The camera model is {str(camera_model)}")