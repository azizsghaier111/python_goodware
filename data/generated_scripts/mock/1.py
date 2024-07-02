import pytest
from unittest import mock
from PIL import Image

# mylib.py
def generator(count):
    for i in range(count):
        yield i

def image_manipulation(image_path):
    with Image.open(image_path) as img:
        return img.resize((800, 600))

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
    mock_open.return_value = mock_img
    test_img_path = '/path/to/img'
    image_manipulation(test_img_path)
    mock_open.assert_called_with(test_img_path)
    mock_img.__enter__.assert_called_once()
    mock_resize.assert_called_with((800, 600))

def test_generator_exception():
    with pytest.raises(Exception):
        def faulty_generator(count):
            for i in range(count):
                if i == 50:
                    raise Exception("An error occurred!")
                yield i

        gen = faulty_generator(100)
        while True:
            next(gen)

@mock.patch('__main__.generator')
def test_mock_generator(mock_gen):
    sequence = [i for i in range(10)]
    mock_gen.side_effect = [iter(sequence)]
    gen = generator(10)
    for i, val in enumerate(gen):
        assert i == val