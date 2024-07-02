import pytest
import asyncio
from unittest import mock
from unittest.mock import call
from PIL import Image

# Mocking Method Calls
class Foo:
    def bar(self, value):
        return value * 2

@mock.patch.object(Foo, 'bar', return_value=42)
def test_foo_bar_method(mocked_bar):
    foo = Foo()
    assert foo.bar(21) == 42
    mocked_bar.assert_called_once_with(21)

# Mocking Object
class Object:
    def method(self, num):
        return num

@mock.patch.object(Object, 'method', return_value=10)
def test_object_method(mock_method):
    obj = Object()
    assert obj.method(5) == 10
    mock_method.assert_called_once_with(5)

# Mocking a generator method
def generator(count):
    for i in range(count):
        yield i

@mock.patch('__main__.generator', return_value=iter([0, 1, 2]))
def test_generator(mocked_generator):
    gen = generator(3)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 2
    mocked_generator.assert_called_once_with(3)

# Mocking Async Calls and pillow's Image manipulations
class DummyClass:
    async def dummy_async_func(self):
        pass

def image_manipulation(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width > 800 or height > 600:
                return img.resize((800, 600))
        return img
    except IOError as e:
        print("Unable to open image")
    except:
        print("Unexpected error")

@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('PIL.Image.Image.resize')
@mock.patch.object(DummyClass, 'dummy_async_func', new_callable=mock.AsyncMock)
def test_async_and_image_manipulation(mocked_async_func, mock_resize, mock_open):
    # Testing async method
    loop = asyncio.get_event_loop()
    dummy_obj = DummyClass()
    loop.run_until_complete(dummy_obj.dummy_async_func())
    mocked_async_func.assert_called_once()

    # Testing PIL's image manipulation
    mock_img = mock.MagicMock(spec=Image.Image)
    mock_img.size = [1000, 800]  # image size bigger than limit, resize should be called
    mock_open.return_value.__enter__.return_value = mock_img
    test_img_path = '/path/to/img'
    image_manipulation(test_img_path)
    mock_open.assert_any_call(test_img_path)
    mock_open.assert_has_calls([call(test_img_path)])  # Use of assert_has_calls
    mock_img.__enter__.assert_called_once()
    mock_resize.assert_called_once_with((800, 600))

if __name__ == "__main__":
    pytest.main([__file__])