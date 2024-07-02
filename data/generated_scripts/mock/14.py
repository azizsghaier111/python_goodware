import pytest
from unittest.mock import Mock, call, patch
from PIL import Image
import asyncio
from collections import namedtuple

# NamedTuple to mock
Person = namedtuple("Person", ["name", "age"])

class ImageProcessor:
    async def process_image(self, image):
        await asyncio.sleep(1)  # complexity of the image processing pictured with aio sleep
        return 'processed'

# Pointless function we want to mock
async def pointless_function(image):
    return image

# Initialize some unittest.mock.Mock objects
mock_image_processor = Mock(spec=ImageProcessor)
mock_pointless_function = Mock(wraps=pointless_function)
mock_person = Mock(spec=Person)

@pytest.mark.asyncio
async def test_image_processor():
    image_to_process = 'image.png'  # just a string for simplistic purposes
    expected_result_processed = 'processed'

    mock_image_processor.process_image = AsyncMock(return_value=expected_result_processed)
    result = await mock_image_processor.process_image(image_to_process)
    
    calls = [call(image_to_process)]
    mock_image_processor.process_image.assert_has_calls(calls)
    assert result == expected_result_processed
    mock_image_processor.process_image.reset_mock()

@pytest.mark.asyncio
async def test_pointless_function():
    image_to_process = { 'raw': 'image.png', 'processed': None }  # just a dict for simplistic purposes

    mock_pointless_function.return_value = asyncio.Future()
    mock_pointless_function.return_value.set_result(image_to_process)

    result = await mock_pointless_function(image_to_process)

    mock_pointless_function.assert_called_once_with(image_to_process)
    assert result == image_to_process
    mock_pointless_function.reset_mock()

def test_named_tuple():
    mock_person.name = 'John Doe'
    mock_person.age = 30

    assert mock_person.name == 'John Doe'
    assert mock_person.age == 30
    mock_person.reset_mock()

# class to mock normal methods in python async mode
class AsyncMock(Mock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

# Run the test
if __name__ == '__main__':
    pytest.main()