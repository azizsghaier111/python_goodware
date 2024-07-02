import os
import numpy as np
import piexif
from PIL import Image


# Load image and its EXIF data
def load_image_and_exif(file_path):
    image = Image.open(file_path)
    exif_data = piexif.load(image.info.get("exif", b""))
    return image, exif_data


def check_if_exif_exists(file_path):
    _, exif_data = load_image_and_exif(file_path)

    # Zeroth (0th) segment contains basic image metadata like Image Description, Make, Model, etc.
    # We use this for simplicity to check if any basic metadata exists.
    zeroth_segment = exif_data["0th"]

    # If there is no exif data, it returns an empty dict
    return bool(zeroth_segment)


def extract_thumbnail_from_exif(file_path):
    image, exif_data = load_image_and_exif(file_path)

    thumbnail = exif_data.pop("thumbnail", None)
    if thumbnail is not None:
        with open(f"thumbnail_{os.path.basename(file_path)}", "wb") as f:
            f.write(thumbnail)


def embed_exif_data_into_jpeg(file_path, exif_data):
    image = Image.open(file_path)
    exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=exif_bytes)


if __name__ == "__main__":
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'

    # Check if EXIF data exists
    print(f"EXIF data exists: {check_if_exif_exists(source_file_path)}")

    # Extract thumbnail from EXIF data
    extract_thumbnail_from_exif(source_file_path)

    # Load source image and get its EXIF data
    _, source_exif_data = load_image_and_exif(source_file_path)

    # Embed EXIF data from the source image into the target image
    embed_exif_data_into_jpeg(target_file_path, source_exif_data)