import pytest
from unittest import mock
from PIL import Image

# mylib.py
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
def test_generator():
    gen = generator(3)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(gen)

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.resize')
def test_image_manipulation(mock_resize, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [1000, 800]
    mock_open.return_value.__enter__.return_value = mock_img
    test_img_path = '/path/to/img'
    image_manipulation(test_img_path)
    mock_open.assert_any_call(test_img_path)
    mock_img.__enter__.assert_called_once()
    mock_resize.assert_called_once_with((800, 600))

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.resize')
def test_image_manipulation_no_resize(mock_resize, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [600, 400]
    mock_open.return_value.__enter__.return_value = mock_img
    test_img_path = '/path/to/img'
    image_manipulation(test_img_path)
    mock_open.assert_any_call(test_img_path)
    mock_img.__enter__.assert_called_once()
    mock_resize.assert_not_called()

def generator_with_exception(count):
    for i in range(count):
        if i == 50:
            raise Exception("An error occurred!")
        yield i

def test_generator_exception():
    with pytest.raises(Exception):
        gen = generator_with_exception(100)
        for _ in gen:
            pass

@mock.patch('__main__.generator')
def test_mock_generator(mock_gen):
    sequence = [i for i in range(10)]
    mock_gen.return_value = sequence
    gen = generator(10)
    for i, val in enumerate(gen):
        assert i == val