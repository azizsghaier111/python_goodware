import pytest
from unittest import mock
from unittest.mock import call
from PIL import Image
import asyncio
import os
    
class ImageProcessor:
    async def process_images(self, image_paths):
        for path in image_paths:
            # Assuming image processing involves opening image and save it as JPEG
            try:
                with Image.open(path) as img:
                    img.save(os.path.splitext(path)[0] + ".jpg", "JPEG")
            except Exception as e:
                print(f"Failed to process {path}. Error: {e}")

class DerivedImageProcessor(ImageProcessor):
    async def process_images(self, image_paths):
        # Lets say derived class converts it to PNG instead of jpeg
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.save(os.path.splitext(path)[0] + ".png", "PNG")
            except Exception as e:
                print(f"Failed to process {path}. Error: {e}")

# Initialize a unittest.mock.Mock object
mock_image_processor = mock.create_autospec(ImageProcessor)
mock_derived_image_processor = mock.create_autospec(DerivedImageProcessor)

@pytest.mark.asyncio
async def test_process_images():
    
    # Mock the `process_images` method with async function
    async def mock_process_images(image_paths):
        return ["processed"] * len(image_paths)

    # Run the mock method
    mock_image_processor.process_images = mock_process_images
    mock_derived_image_processor.process_images = mock_process_images
    
    # Mock image paths
    image_paths = ["image1.png", "image2.png", "image3.png"]

    # Run the test
    result = await mock_image_processor.process_images(image_paths)
    assert result == ["processed"] * len(image_paths)

    result_derived = await mock_derived_image_processor.process_images(image_paths)
    assert result_derived == ["processed"] * len(image_paths)
    
    # Test if our function is called with expected arguments
    calls = [call(image_paths)]
    mock_image_processor.process_images.assert_has_calls(calls)

    # Test if our derived class function is called with expected arguments
    mock_derived_image_processor.process_images.assert_has_calls(calls)

    # Test if our function is not called with unexpected arguments
    mock_image_processor.process_images.assert_not_called_with(["unexpected.png"])

    # Test if our derived class function is not called with unexpected arguments
    mock_derived_image_processor.process_images.assert_not_called_with(["unexpected.png"])

    print('Tests completed successfully!')

# Run the test
pytest.main()