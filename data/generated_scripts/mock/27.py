import pytest
from unittest.mock import Mock, call
from PIL import Image
from collections import namedtuple
import asyncio

# NamedTuple to mock
Person = namedtuple("Person", ["name", "age"])
ImageObj = namedtuple("ImageObj", ["name", "extension"])


class ImageProcessor:
    async def process_image(self, image: ImageObj):
        # Simulate image processing delay
        await asyncio.sleep(1)
        return 'processed'


class PDFProcessor:
    async def process_pdf(self, filename: str):
        # Simulate PDF processing delay
        await asyncio.sleep(1)
        return 'processed'


# Asynchronous function to mock
async def pointless_function(image: ImageObj):
    return image


async def another_pointless_function(pdf: str):
    return pdf


# Function to mock normal methods in Python's asynchronous context
class AsyncMock(Mock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Initialize some unittest.mock.Mock objects
mock_image_processor = Mock(spec=ImageProcessor)
mock_pdf_processor = Mock(spec=PDFProcessor)
mock_pointless_function = AsyncMock(wraps=pointless_function)
mock_another_pointless_function = AsyncMock(wraps=another_pointless_function)
mock_person = Mock(spec=Person)
mock_image_obj = Mock(spec=ImageObj)


@pytest.mark.asyncio
async def test_image_processor():
    image_to_process = ImageObj('image', '.png')

    # Specify the return value
    mock_image_processor.process_image.return_value = await AsyncMock(return_value='processed')()

    result = await mock_image_processor.process_image(image_to_process)

    calls = [call(image_to_process)]
    mock_image_processor.process_image.assert_has_calls(calls)

    # Assert the return value
    assert result == 'processed'

    # Reset the mock object
    mock_image_processor.reset_mock()


@pytest.mark.asyncio
async def test_pdf_processor():
    pdf_to_process = 'document.pdf'

    mock_pdf_processor.process_pdf.return_value = await AsyncMock(return_value='processed')()

    result = await mock_pdf_processor.process_pdf(pdf_to_process)

    calls = [call(pdf_to_process)]
    mock_pdf_processor.process_pdf.assert_has_calls(calls)

    assert result == 'processed'

    mock_pdf_processor.reset_mock()


@pytest.mark.asyncio
async def test_pointless_function():
    image_to_process = ImageObj('image', '.png')

    mock_pointless_function.return_value = await AsyncMock(return_value=image_to_process)()

    result = await mock_pointless_function(image_to_process)

    mock_pointless_function.assert_called_once_with(image_to_process)

    assert result == image_to_process

    mock_pointless_function.reset_mock()


@pytest.mark.asyncio
async def test_another_pointless_function():
    pdf_to_process = 'document.pdf'

    mock_another_pointless_function.return_value = await AsyncMock(return_value=pdf_to_process)()

    result = await mock_another_pointless_function(pdf_to_process)

    mock_another_pointless_function.assert_called_once_with(pdf_to_process)

    assert result == pdf_to_process

    mock_another_pointless_function.reset_mock()


def test_named_tuple():
    # Create a new mock object
    mock_person.name = 'John Doe'
    mock_person.age = 30

    # Verify the object's attributes
    assert mock_person.name == 'John Doe'
    assert mock_person.age == 30


def test_named_tuple_image_obj():
    # Create a new mock object
    mock_image_obj.name = 'image'
    mock_image_obj.extension = '.png'

    # Verify the object's attributes
    assert mock_image_obj.name == 'image'
    assert mock_image_obj.extension == '.png'
    
# Run the test
if __name__ == '__main__':
    pytest.main()