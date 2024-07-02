# Importing required Libraries 
import piexif
import os
from PIL import Image
import numpy as np

def load_image_and_exif(file_path):
    '''
    This function is used to load image and exif data
    from the image
    '''
    try:
        # opening the image file
        image = Image.open(file_path)
        # getting exif info from the image
        exif_data = piexif.load(image.info.get("exif", b""))
    except Exception as e:
        # capturing and printing exceptions
        print(f"Error while loading image and exif data: {e}")
        return None, None
    return image, exif_data

def check_if_exif_exists(file_path):
    '''
    This function checks if an exif exist in
    image file or not
    '''
    _, exif_data = load_image_and_exif(file_path)
    zeroth_segment = exif_data.get("0th", {}) if exif_data else {}
    return bool(zeroth_segment)

def extract_thumbnail_from_exif(file_path):
    '''
    This function extracts thumbnail from exif data
    '''
    try:
        # load image and exif data
        image, exif_data = load_image_and_exif(file_path)
        # poping out the thumbnail info
        thumbnail = exif_data.pop("thumbnail", None)
        # if thumbnail exits:
        if thumbnail:
            #writing a file for thumbnail
            with open(f"thumbnail_{os.path.basename(file_path)}", "wb") as f:
                f.write(thumbnail)
    except Exception as e:
        print(f"Error while extracting thumbnail: {e}")

def embed_exif_data_into_jpeg(file_path, exif_data):
    '''
    This function embeds exif data into jpeg images
    '''
    try:
        # open image
        image = Image.open(file_path)
        # dump exif data into jpeg image
        exif_bytes = piexif.dump(exif_data)
        # save the image
        image.save(file_path, exif=exif_bytes)
    except Exception as e:
        print(f"Error while embedding exif data: {e}")

def modify_exif_data(file_path, new_exif_data):
    '''
    This function is used to modify exif data inside images
    '''
    try:
        _, exif_data = load_image_and_exif(file_path)
        # update exif data with new exif data
        exif_data.update(new_exif_data)
        # embed this new exif data into image
        embed_exif_data_into_jpeg(file_path, exif_data)
    except Exception as e:
        print(f"Error while modifying exif data: {e}")
        
# defining main function to run all the above functions sequentially
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