import pytest
from unittest import mock
from PIL import Image

# mylib.py
def get_image_path():
    return '/path/to/img'

def generator(count):
    for i in range(count):
        yield i

def image_manipulation(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        if width > 800 or height > 600:
            return img.resize((800, 600))
    return img

# tests.py
@mock.patch('__main__.get_image_path')    
def test_get_image_path(mock_get_image_path):
    test_img_path = '/path/to/img'
    mock_get_image_path.return_value = test_img_path
    assert get_image_path() == test_img_path
    mock_get_image_path.assert_called_once()

def test_generator():
    gen = generator(3)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(gen)

@mock.patch('__main__.Image.open', new_callable=mock.mock_open)
def test_image_manipulation(mock_open):
    test_img_path = '/path/to/img'
    with Image.open(test_img_path) as img:
        img_manipulation(img)
    mock_open.assert_called_once_with(test_img_path)

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.resize')
def test_image_manipulation_resize(mock_resize, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [1000, 800]  # image size bigger than limit, resize should be called
    mock_open.return_value.__enter__.return_value = mock_img
    image_manipulation(get_image_path())
    mock_resize.assert_called_once_with((800, 600))
    mock_open.assert_called_once_with(get_image_path())
    mock_open.return_value.__exit__.assert_called_once()

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.resize')
def test_image_manipulation_no_resize(mock_resize, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [600, 400]  # image size within limit, resize should not be called
    mock_open.return_value.__enter__.return_value = mock_img
    image_manipulation(get_image_path())
    mock_resize.assert_not_called()
    mock_open.assert_called_once_with(get_image_path())
    mock_open.return_value.__exit__.assert_called_once()

@mock.patch('__main__.generator')
def test_mock_generator(mock_gen):
    sequence = [i for i in range(10)]
    mock_gen.return_value.__iter__.return_value = sequence
    gen = generator(10)
    for i, val in enumerate(gen):
        assert i == val

@mock.patch('__main__.generator')
def test_mock_generator_with_side_effect(mock_gen):
    mock_gen.side_effect = (i for i in range(10))
    gen = generator(10)
    for i, val in enumerate(gen):
        assert i == val