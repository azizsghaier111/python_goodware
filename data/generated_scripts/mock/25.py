import pytest
from unittest import mock
from PIL import Image

class MyClass:
    def get_image_path(self):
        return '/path/to/img.jpg'

    def get_image_format(self):
        image_path = self.get_image_path()
        with Image.open(image_path) as img:
            return img.format

    def save_image(self, image_path, format):
        img = Image.new('RGB', (100, 100))
        img.save(image_path, format)

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.readlines()

# test_mylib.py
from mylib import MyClass

@mock.patch.object(MyClass, 'get_image_path')    
def test_get_image_path(mock_get_image_path):
    test_img_path = '/path/to/test_img.jpg'
    mock_get_image_path.return_value = test_img_path
    image_path = MyClass().get_image_path()
    assert image_path == test_img_path
    mock_get_image_path.assert_called_once()

@mock.patch('PIL.Image.open', new_callable=mock.mock_open)
@mock.patch.object(MyClass, 'get_image_path')
def test_get_image_format(mock_get_image_path, mock_open):
    test_img_path = '/path/to/test_img.jpg'
    mock_get_image_path.return_value = test_img_path
    img_format = MyClass().get_image_format()
    mock_open.assert_called_once_with(test_img_path)

@mock.patch('PIL.Image.Image.save')
@mock.patch.object(MyClass, 'get_image_path')
def test_save_image(mock_get_image_path, mock_save):
    test_img_path = '/path/to/test_img.jpg'
    mock_get_image_path.return_value = test_img_path
    MyClass().save_image(test_img_path, 'JPEG')
    mock_save.assert_called_once_with(test_img_path, 'JPEG')

@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_read_file(mock_open):
    test_file_path = '/path/to/test_file'
    MyClass().read_file(test_file_path)
    mock_open.assert_called_once_with(test_file_path, 'r')

@mock.patch('__main__.MyClass.get_image_path')
def test_mock_get_image_path_with_side_effect(mock_get_image_path):
    mock_get_image_path.side_effect = ['path/to/img1.jpg', 'path/to/img2.jpg']
    assert MyClass().get_image_path() == 'path/to/img1.jpg'
    assert MyClass().get_image_path() == 'path/to/img2.jpg'