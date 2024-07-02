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


def get_datetime_exif(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)

    datetime = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
    return datetime


def insert_thumbnail_to_exif(file_path, thumbnail_path):
    check_file(file_path)
    check_file(thumbnail_path)

    exif_dict = read_exif(file_path)

    with open(thumbnail_path, "rb") as f:
        thumbnail = f.read()

    exif_dict['thumbnail'] = thumbnail
    write_exif(file_path, exif_dict)


def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)

    exif_dict = read_exif(source_file_path)
    write_exif(target_file_path, exif_dict)


if __name__ == '__main__':
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'
    thumbnail_path = 'path_to_your_thumbnail'

    exif_dict = read_exif(source_file_path)

    write_exif(source_file_path, exif_dict)

    datetime = get_datetime_exif(source_file_path)
    print(f"Date/Time: {datetime}")

    clone_exif_data(source_file_path, target_file_path)

    insert_thumbnail_to_exif(target_file_path, thumbnail_path)