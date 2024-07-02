import asyncio
import pytest
from unittest.mock import patch, call, Mock

# Here is the function we want to test
async def process_images(image_processor, image_paths):
    for path in image_paths:
        await image_processor(path)

# Here is the mocked function
async def mock_process_image(image_path):
    return "processed"

@pytest.mark.asyncio
async def test_process_images():

    # Here we mock the function with unittest.mock.patch
    with patch('__main__.mock_process_image', new_callable=Mock) as mock:
        mock.side_effect = mock_process_image

        # Create a list of image paths for testing
        image_paths = ["image1.png", "image2.png", "image3.png"]

        # Call the function process_images with our mock as argument. Now, mock_process_image will be called instead of a real function
        await process_images(mock, image_paths)

        # Here we create a list of calls that we expect our mock to have
        expected_calls = [call(path) for path in image_paths]

        # Here we assert if our mock called with expected arguments
        mock.assert_has_calls(expected_calls, any_order = True)

        # Assert that our function was called expected times
        assert mock.call_count == len(image_paths)

        # Now we assert that our function was called atleast once with the following argument
        mock.assert_any_call(image_paths[0])

        # Now we'll assert that our function was not called with the following argument
        mock.assert_not_called_with("non_existent.png")

# Run the test
pytest.main()