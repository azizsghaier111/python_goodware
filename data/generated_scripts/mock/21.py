import pytest
from unittest import mock
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import call
import collections
from PIL import Image
from mylib import generator, image_manipulation


MockedTuple = collections.namedtuple('MockedTuple', ['method', 'args'])

def test_generator():
    for i in range(10):
        mock_generator = Mock()
        mock_generator.side_effect = generator(5)
        assert next(mock_generator) == 0
        assert next(mock_generator) == 1
        assert next(mock_generator) == 2
        assert next(mock_generator) == 3
        assert next(mock_generator) == 4

def test_mock_generator_assert():
    for i in range(10):
        mock_generator = Mock()
        mock_generator.side_effect = generator(5)
        assert next(mock_generator) == 0
        for j in range(4):
            mock_generator.method.assert_not_called()
        with pytest.raises(StopIteration):
            next(mock_generator)

def test_image_manipulation():
    for _ in range(10):
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (1000, 800)
        with patch('PIL.Image.open', return_value=mock_image):
            resized_image = image_manipulation('dummy_path')
        mock_image.resize.assert_called_once_with((800, 600))

def test_image_manipulation_multiple():
    for i in range(10):
        expected_sizes = [(900*(i/10), 800*(i/10)) for i in range(1, 11)]
        mock_images = [Mock(spec=Image.Image) for _ in range(10)]
        for image in mock_images:
            image.size = (1000, 800)
        with patch('PIL.Image.open', new_callable=mock.MagicMock) as mocked_open:
            mocked_open.side_effect = mock_images
            for i, expected_size in enumerate(expected_sizes):
                resized_image = image_manipulation(f'dummy_path_{i}')
                mock_images[i].resize.assert_called_once_with(expected_size)


def test_mocking_namedtuples():
    method_calls = ['method1', 'method2', 'method3', 'method4', 'method5', 'method6']
    with patch.object(MockedTuple, 'method') as mocked_method:
        mock_tuples = [MockedTuple(method, args) for method, args in zip(method_calls, [(), (1,), (1, 2), (3, 4), (5, 6), (7, 8)])]
        for mock in mock_tuples:
            assert mock.method
            mocked_method.assert_called_with(mock.args)

def test_mocking_namedtuples_multi():
    methods = ['method{}'.format(i+1) for i in range(20)]
    for j in range(10):
        with patch.object(MockedTuple, 'method') as mocked_method:
            args = [(i, i+1) for i in range(20)]
            mock_tuples = [MockedTuple(method, arg) for method, arg in zip(methods, args)]
            for mock in mock_tuples:
                assert mock.method
                mocked_method.assert_not_called()

if __name__ == "__main__":
    test_generator()
    test_mock_generator_assert()
    test_image_manipulation()
    test_image_manipulation_multiple()
    test_mocking_namedtuples()
    test_mocking_namedtuples_multi()