import os
import piexif
from PIL import Image
from typing import Optional
import struct
import datetime
import requests
from io import BytesIO

def check_if_exists(path: str) -> None:
    if not os.path.exists(path):
        raise Exception(f"The file with the given path doesn't exist: {path}")

def extract_thumbnail(file_path: str) -> Optional[BytesIO]:
    check_if_exists(file_path)

    thumbnail = None
    with open(file_path, 'rb') as img_file:
        exif_dict = piexif.load(img_file.read())
        thumbnail = exif_dict.pop("thumbnail")

    return thumbnail

def save_thumbnail(thumbnail: Optional[BytesIO], file_path: str = "thumbnail.jpg") -> None:
    if thumbnail is not None:
        with open(file_path, 'wb') as thumbnail_file:
            thumbnail_file.write(thumbnail)

def get_exif_data(file_path: str) -> dict:
    check_if_exists(file_path)

    with open(file_path, 'rb') as img_file:
        exif_dict = piexif.load(img_file.read())

    return exif_dict

def check_exif(file_path: str) -> bool:
    exif_dict = get_exif_data(file_path)

    return bool(exif_dict)

def get_latitude_longitude(exif_data: dict) -> Optional[tuple]:
    latitude = None
    longitude = None
    if "GPS" in exif_data:
        geodata = exif_data["GPS"]
        latitude = geodata.get(piexif.GPSIFD.GPSLatitude)
        latitude_ref = geodata.get(piexif.GPSIFD.GPSLatitudeRef)
        longitude = geodata.get(piexif.GPSIFD.GPSLongitude)
        longitude_ref = geodata.get(piexif.GPSIFD.GPSLongitudeRef)

        if latitude and latitude_ref and longitude and longitude_ref:
            return format_geo_data(latitude), format_geo_data(longitude)

def format_geo_data(geodata: tuple) -> float:
    degrees = geodata[0]
    minutes = geodata[1]
    seconds = geodata[2]

    return (degrees + minutes / 60 + seconds / 3600)

if __name__=="__main__":
    file_path = 'image.jpg'  # replace with your image path

    try:
        exif_exists = check_exif(file_path)
        print(f"Does EXIF data exists: {exif_exists}")

        exif_data = get_exif_data(file_path)
        latitude_longitude = get_latitude_longitude(exif_data)
        print(f"Latitude and longitude: {latitude_longitude}")

        thumbnail = extract_thumbnail(file_path)
        save_thumbnail(thumbnail)

    except Exception as exc:
        print(f"Caught an exception: {exc}")