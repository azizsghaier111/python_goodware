import pytest
from unittest.mock import Mock, patch
from PIL import Image
from mylib import generator, image_manipulation
import collections

# Creating a namedtuple for mocking
MockedTuple = collections.namedtuple('MockedTuple', ['method', 'args'])

def test_generator():
    mock_generator = Mock()
    mock_generator.side_effect = generator(5)
    assert next(mock_generator) == 0
    assert next(mock_generator) == 1
    assert next(mock_generator) == 2
    assert next(mock_generator) == 3
    assert next(mock_generator) == 4
    with pytest.raises(StopIteration):
        next(mock_generator)
        
def test_image_manipulation():
    # defining a mock image
    mock_image = Mock(spec=Image.Image)
    mock_image.size = (1000, 800)

    # patching Image.open to return the mock_image
    with patch('PIL.Image.open', return_value=mock_image):
        resized_image = image_manipulation('dummy_path')
        
    # verifying that the resize method was called on mock_image with the right arguments
    mock_image.resize.assert_called_once_with((800, 600))

def test_mocking_namedtuples():
    method_calls = ['method1', 'method2', 'method3']
    with patch.object(MockedTuple, 'method') as mocked_method:
        
        mock_tuples = [MockedTuple(method, args) for method, args in zip(method_calls, [(), (1,), (1, 2)])]
        
        for mock in mock_tuples:
            assert mock.method
            mocked_method.assert_called_with(mock.args)
            
if __name__ == "__main__":
    test_generator()
    test_image_manipulation()
    test_mocking_namedtuples()