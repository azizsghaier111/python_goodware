from PIL import Image
from unittest.mock import AsyncMock, MagicMock
import pytest
from asyncio import run
import types
import itertools

mocked_image_path = 'mock_image.jpeg'

# Mocking a generator
mocked_generator = MagicMock()
mocked_generator.__iter__ = MagicMock(return_value=iter([i for i in range(10)]))

# Mocking image resize function
mocked_image_resize = MagicMock()

# Patching the return value of the open function
Image.open = MagicMock(return_value=mocked_image_resize)

# Mocking Async Calls
async def mocked_async_function():
    return 100

class MyClass:
    async def my_async_method(self, my_param):
        return my_param

my_class = MyClass()
my_class.my_async_method = AsyncMock(return_value=mocked_async_function())

# mylib.py
def generator(count):
    for i in range(count):
        yield i

def image_manipulation(image_path):
    with Image.open(image_path) as img:
        return img.resize((800, 600))

class TestMyLib:
    @pytest.mark.asyncio
    async def test_mocking_async_calls(self):
        assert await my_class.my_async_method(100) == 100

    def test_mocking_iteration(self):
        assert list(generator(10)) == list(mocked_generator)

    def test_image_manipulation(self):
        assert image_manipulation(mocked_image_path) == mocked_image_resize

# Run the test methods
if __name__ == '__main__':
    test_class = TestMyLib()
    run(test_class.test_mocking_async_calls())
    test_class.test_mocking_iteration()
    test_class.test_image_manipulation()