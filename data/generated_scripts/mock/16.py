import pytest
from unittest import mock
from PIL import Image

# functions under test
def generator(count):
    for i in range(count):
        yield i

def image_manipulation(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        if width > 800 or height > 600:
            return img.resize((800, 600))
    return img

def increment_data(data):
    return [i + 1 for i in data]

def flip_image(image_path):
    with Image.open(image_path) as img:
        return img.transpose(Image.FLIP_LEFT_RIGHT)

# tests
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

def test_increment_data():
    data = [1, 2, 3]
    assert increment_data(data) == [2, 3, 4]

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.transpose')
def test_flip_image(mock_transpose, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_open.return_value.__enter__.return_value = mock_img
    test_img_path = '/path/to/img'
    flip_image(test_img_path)
    mock_open.assert_any_call(test_img_path)
    mock_img.__enter__.assert_called_once()
    mock_transpose.assert_called_once_with(Image.FLIP_LEFT_RIGHT)

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.transpose')
def test_flip_image_no_flip(mock_transpose, mock_open):
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [600, 400]
    mock_open.return_value.__enter__.return_value = mock_img
    test_img_path = '/path/to/img'
    flip_image(test_img_path)
    mock_open.assert_any_call(test_img_path)
    mock_img.__enter__.assert_called_once()
    mock_transpose.assert_not_called()