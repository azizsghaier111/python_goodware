import os
import piexif
from PIL import Image

# Load image and get EXIF data
def load_image_and_exif(image_file_path):
    try:
        img = Image.open(image_file_path)
        exif_dict = piexif.load(img.info["exif"])
    except FileNotFoundError:
        print(f"No file found at {image_file_path}")
        return None, None
    return img, exif_dict

# Embed EXIF data into JPEG
def embed_exif_into_img(img, exif_dict, output_file_path):
    try:
        exif_bytes = piexif.dump(exif_dict)
        img.save(output_file_path, "jpeg", exif=exif_bytes)
    except Exception as e:
        print(f"Failed to embed EXIF into image due to {e}")

# Insert Thumbnail into EXIF
def insert_thumbnail_into_exif(exif_dict, thumbnail_file_path):
    try:
        with open(thumbnail_file_path, "rb") as thumbnail_file:
            exif_dict["thumbnail"] = thumbnail_file.read()
    except Exception as e:
        print(f"Failed to insert thumbnail due to {e}")

# Main flow
def run_flow(image_file_path, thumbnail_file_path, output_file_path):
    img, exif_dict = load_image_and_exif(image_file_path)
    if img is None or exif_dict is None:
        return

    insert_thumbnail_into_exif(exif_dict, thumbnail_file_path)
    
    embed_exif_into_img(img, exif_dict, output_file_path)

source_img = "source_image.jpg"
target_thumbnail = "thumbnail.jpg"
target_img = "target_image.jpg"

run_flow(source_img, target_thumbnail, target_img)