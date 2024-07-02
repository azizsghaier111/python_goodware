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
    if "exif" in img.info:
        exif_dict = piexif.load(img.info['exif'])
        return exif_dict
    else:
        raise ValueError("The image file does not have EXIF data.")

def write_exif(file_path, exif_dict):
    check_file(file_path)
    img = Image.open(file_path)
    exif_bytes = piexif.dump(exif_dict)
    img.save(file_path, exif=exif_bytes)

def get_exif_focal_length(file_path):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    if piexif.ExifIFD.FocalLength in exif_dict['Exif']:
        focal_length = exif_dict['Exif'][piexif.ExifIFD.FocalLength]
        if isinstance(focal_length, tuple):
            focal_length = focal_length[0] / focal_length[1]
        return focal_length
    else:
        raise ValueError("The image file does not have FocalLength data.")

def clone_exif_data(source_file_path, target_file_path):
    check_file(source_file_path)
    check_file(target_file_path)
    exif_dict = read_exif(source_file_path)
    write_exif(target_file_path, exif_dict)

def remove_exif_data(file_path):
    check_file(file_path)
    piexif.remove(file_path)

def remove_specific_exif_data(file_path, exif_key):
    check_file(file_path)
    exif_dict = read_exif(file_path)
    for ifd_name in exif_dict:
        if exif_key in exif_dict[ifd_name]:
            del exif_dict[ifd_name][exif_key]
    write_exif(file_path, exif_dict)

def test_functions(source_file_path, target_file_path):
    exif_data_source = read_exif(source_file_path)
    print("1. Reading EXIF data from the source file... \n", exif_data_source)
    write_exif(target_file_path, exif_data_source)
    print("\n2. Writing EXIF data to the target file... Done!")
    focal_length = get_exif_focal_length(source_file_path)
    print(f"\n3. Focal length of the source file: {focal_length}")
    clone_exif_data(source_file_path, target_file_path)
    print("\n4. Cloning EXIF data from the source file to the target file... Done!")
    remove_exif_data(target_file_path)
    print("\n5. Removing EXIF data from the target file... Done!")
    exif_data_target = read_exif(target_file_path)
    print("\n6. Reading EXIF data from the target file... \n", exif_data_target)
    random_key = np.random.choice(list(exif_data_source['0th'].keys()))
    remove_specific_exif_data(target_file_path, random_key)
    print(f"\n7. Removing specific EXIF data (key={random_key}) from the target file... Done!")

if __name__ == '__main__':
    source_file = 'path_to_source_file.jpg'
    target_file = 'path_to_target_file.jpg'
    test_functions(source_file, target_file)