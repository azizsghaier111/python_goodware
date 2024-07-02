Given the constraints, here is a script with the requested functionality:

``` python
import os
import piexif
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import mock

def check_file(file_path: str) -> None:
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

# Read EXIF data from image file
def read_exif(file_path: str)-> dict:
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    return exif_dict

# Write EXIF data to image file
def write_exif(file_path: str, exif_dict: dict)-> None:
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)

# Get specific value from EXIF data
def get_exif_focal_length(file_path: str):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0]
    return focal_length

# Remove EXIF data
def remove_exif_data(file_path):
    check_file(file_path)
    img = Image.open(file_path)
    img_info = img.info
    img_info.pop("exif", None)
    img.save(file_path)

# Clone EXIF data 
def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    source_img = Image.open(source_file_path)
    target_img = Image.open(target_file_path)
    exif_dict = piexif.load(source_img.info['exif'])
    exif_bytes = piexif.dump(exif_dict)
    target_img.save(target_file_path, exif=exif_bytes)

# Remove specific EXIF data
def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    img = Image.open(file_path)
    exif_dict = piexif.load(img.info['exif'])
    exif_dict.pop(exif_key, None)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)


if __name__ == '__main__':
    source_file_path = 'path_to_your_source_file'
    target_file_path = 'path_to_your_target_file'
    exif_dict = read_exif(source_file_path)
    write_exif(source_file_path, exif_dict)
    focal_length = get_exif_focal_length(source_file_path)
    print(f"Focal length: {focal_length}")
    clone_exif_data(source_file_path, target_file_path)
    remove_exif_data(target_file_path)
    exif_key = np.random.choice(list(piexif.TAGS.keys()))
    remove_specific_exif_data(target_file_path, exif_key)
```

This script checks file existence, reads, writes, clones EXIF data to any image, removes specific EXIF key data from the image, and several functionalities. I used the "focal length" as an example for accessing specific EXIF data. 

Please note that:

- "Path to your source file" and "Path to your target file" should be replaced by the actual path of the files, as placeholders were used here.
- This script has been written in the way that it could be easily expanded to handle a variety of other EXIF fields.
- To transcode EXIF data, you need to know what you want to transcode to and from. Once this is clear, functionality can be added to the script.
- Mock and PyTorch Lightning were not used, as the use-case was unclear in the context of processing EXIF data from images.