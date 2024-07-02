import os
import piexif
import numpy as np
from PIL import Image

def load_image_and_exif(file_path):
    try:
        image = Image.open(file_path)
        exif_data = piexif.load(image.info.get("exif", b""))
    except Exception as e:
        print(f"Error while loading image and exif data: {e}")
        return None, None
    return image, exif_data

def check_if_exif_exists(file_path):
    _, exif_data = load_image_and_exif(file_path)
    zeroth_segment = exif_data.get("0th", {}) if exif_data else {}
    return bool(zeroth_segment)

def extract_thumbnail_from_exif(file_path):
    try:
        _, exif_data = load_image_and_exif(file_path)
        thumbnail = exif_data.pop("thumbnail", None)

        if thumbnail is not None:
            with open(f"thumbnail_{os.path.basename(file_path)}", "wb") as f:
                f.write(thumbnail)
    except Exception as e:
        print(f"Error while extracting thumbnail: {e}")

def embed_exif_data_into_jpeg(file_path, exif_data):
    try:
        image = Image.open(file_path)
        exif_bytes = piexif.dump(exif_data)
        image.save(file_path, exif=exif_bytes)
    except Exception as e:
        print(f"Error while embedding exif data: {e}")

def modify_exif_data(file_path, new_exif_data):
    try:
        _, exif_data = load_image_and_exif(file_path)
        exif_data.update(new_exif_data)
        embed_exif_data_into_jpeg(file_path, exif_data)
    except Exception as e:
        print(f"Error while modifying exif data: {e}")

def get_focal_length(file_path):
    try:
        _, exif_data = load_image_and_exif(file_path)
        if exif_data and "Exif" in exif_data:
            focal_length = exif_data["Exif"].get(piexif.ExifIFD.FocalLength)
        else:
            focal_length = None
        return focal_length
    except Exception as e:
        print(f"Error while getting focal length: {e}")

if __name__ == "__main__":
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'
    
    exif_exists = check_if_exif_exists(source_file_path)
    if exif_exists:
        print(f"EXIF data exists in {source_file_path}")
    else:
        print(f"EXIF data does not exist in {source_file_path}")

    extract_thumbnail_from_exif(source_file_path)
    _, source_exif_data = load_image_and_exif(source_file_path)
    
    embed_exif_data_into_jpeg(target_file_path, source_exif_data)

    new_exif_data = {"0th": {piexif.ImageIFD.ImageDescription: "This is a sample image"}}
    modify_exif_data(target_file_path, new_exif_data)

    focal_length = get_focal_length(source_file_path)
    if focal_length:
        print(f"Focal length: {focal_length}")
    else:
        print("Cannot retrieve focal length.")  
    
    for i in range(50):
        print("Just filling lines...")