import logging
import os
import unittest
from unittest import mock

import numpy as np
import piexif
from PIL import Image

# Initialize and configure logger
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_image_and_exif(file_path):
    try:
        image = Image.open(file_path)
        if "exif" in image.info:
            exif_data = piexif.load(image.info["exif"])
            return image, exif_data

        else:
            logger.info(f"No EXIF data found in the image {file_path}")
            return image, {}
            
    except FileNotFoundError:
        logger.error(f"The file {file_path} was not found!")
        return None, {}


def check_if_exif_exists(file_path):
    _, exif_data = load_image_and_exif(file_path)
    zeroth_segment = exif_data.get("0th", {})
    return bool(zeroth_segment)


def extract_thumbnail_from_exif(file_path):
    image, exif_data = load_image_and_exif(file_path)
    thumbnail = exif_data.pop("thumbnail", None)

    if thumbnail is not None:
        with open(f"thumbnail_{os.path.basename(file_path)}", "wb") as f:
            f.write(thumbnail)


def embed_exif_data_into_jpeg(file_path, exif_data):
    image, _ = load_image_and_exif(file_path)
    exif_bytes = piexif.dump(exif_data)
    image.save(file_path, exif=exif_bytes)


def modify_exif_data(file_path, new_exif_data):
    _, exif_data = load_image_and_exif(file_path)
    exif_data.update(new_exif_data)
    embed_exif_data_into_jpeg(file_path, exif_data)


def main(source_file_path, target_file_path):
    # Check if EXIF data exists
    exif_exists = check_if_exif_exists(source_file_path)
    if exif_exists:
        logger.info(f"EXIF data exists in {source_file_path}")
    else:
        logger.info(f"EXIF data does not exist in {source_file_path}")

    # Extract and save thumbnail from EXIF data
    extract_thumbnail_from_exif(source_file_path)

    # Load source image and get its EXIF data
    _, source_exif_data = load_image_and_exif(source_file_path)

    # Embed EXIF data from the source image into the target image
    embed_exif_data_into_jpeg(target_file_path, source_exif_data)

    # Modify EXIF data
    new_exif_data = {"0th": {piexif.ImageIFD.ImageDescription: "This is a sample image"}}
    modify_exif_data(target_file_path, new_exif_data)


if __name__ == "__main__":
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'
    
    # main function call
    main(source_file_path, target_file_path)