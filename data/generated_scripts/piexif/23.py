The provided Python script demonstrates the functionalities you asked for and doesn't need to be 100 lines. Python promotes writing concise, readable code, so adding unnecessary lines would be counterproductive. In Python, achieving simplicity is usually preferable over adding complexity. However, I will modify it slightly to include other functionalities and also to satisfy the 100-line limit requirement.

Here's the revamped code:

``` python
import os
import piexif
from PIL import Image
import datetime

class EXIFUtility:
    def __init__(self, file_path):
        self.file_path = file_path
        self.check_file()

    # Check if the file exists and is valid
    def check_file(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"{self.file_path} does not exist")

    # Load EXIF data from the image file
    def read_exif(self):
        img = Image.open(self.file_path)
        exif_dict = piexif.load(img.info['exif'])
        return exif_dict

    # Write EXIF data to the image file
    def write_exif(self, exif_dict):
        img = Image.open(self.file_path)
        exif_bytes = piexif.dump(exif_dict)
        img.save(self.file_path, "jpeg", exif=exif_bytes)

    # Get specific focal length from EXIF data
    def get_exif_focal_length(self):
        exif_dict = self.read_exif()
        focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength)
        if isinstance(focal_length, tuple):
            focal_length = focal_length[0]
        return focal_length

    # Get specific date and time from EXIF data
    def get_exif_datetime(self):
        exif_dict = self.read_exif()
        datetime_info = exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, "")
        return datetime.datetime.strptime(datetime_info.decode("utf-8"), '%Y:%m:%d %H:%M:%S')

    # Copy EXIF data from one file to another
    def clone_exif_data(self, tgt_file_path):
        exif_dict = self.read_exif()
        new_exif = EXIFUtility(tgt_file_path)
        new_exif.write_exif(exif_dict)

    # Remove EXIF data from the image file
    def remove_exif_data(self):
        piexif.remove(self.file_path)

    # Remove specific EXIF data
    def remove_specific_exif_data(self, exif_key):
        exif_dict = self.read_exif()
        for ifd_name in exif_dict.keys():
            if piexif.is_exist(exif_key, exif_dict[ifd_name]):
                del exif_dict[ifd_name][exif_key]
        self.write_exif(exif_dict)


# Drive code
if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='EXIF data utility')
        parser.add_argument('file_path', type=str, help='The path to the file')
        parser.add_argument('--new_file_path', type=str, required=False, help='The path to the new file')
        args = parser.parse_args()

        source_exif = EXIFUtility(args.file_path)
        
        print("EXIF Data:")
        print("============================")
        print(source_exif.read_exif())
        print()

        focal_length = source_exif.get_exif_focal_length()
        print(f"Focal Length: {focal_length}")

        exif_datetime = source_exif.get_exif_datetime()
        print(f"Original Date Time: {exif_datetime}")

        if args.new_file_path:
            source_exif.clone_exif_data(args.new_file_path)
            print(f"\nEXIF data cloned to {args.new_file_path}")

        source_exif.remove_exif_data()
        print("\nEXIF data removed.")

    except FileNotFoundError as err:
        print(err)
    except Exception as e:
        print(f"Unexpected Error: {e}")
``` 

This script takes in the file path as a command line argument, provides functionality to clone EXIF data to a new file (if provided), outputs the original focal length and date time, and then removes the EXIF data from the original file. The operations are encapsulated in a utility class for better Object oriented design. 

Please do replace the file paths in the script to fit your specific requirements.