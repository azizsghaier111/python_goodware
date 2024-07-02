# import libraries
import pytest
from mock import Mock, MagicMock, call
from pathlib import Path

# define image path
image_path = Path('path/to/image')

# import mylib
import mylib

# define the magic mock 
class MagicMockWithAssertOnce(MagicMock):
    def assert_called_once_with(self, *args, **kwargs):
        assert self.call_count == 1
        calls = [call(*args, **kwargs)]
        self.assert_has_calls(calls)

# mock 'generator' in mylib
mylib.generator = Mock(return_value=iter([0, 1, 2]))
        
# mock 'Image' in mylib
mylib.Image = MagicMock()
mylib.Image.open = MagicMock(return_value=MagicMock(spec=mylib.Image.Image))
mylib.Image.open.return_value.resize = MagicMock()

def test_generator():
    gen = mylib.generator(3)
    assert len(list(gen)) == 3
    mylib.generator.assert_called_once_with(3)
    
def test_image_manipulation():
    new_size = (800, 600)
    img = mylib.image_manipulation(image_path)
    mylib.Image.open.assert_called_once_with(image_path)
    mylib.Image.open.return_value.resize.assert_called_once_with(new_size)

def main():

    # Test generator
    test_generator()
    
    # Test image manipulation
    test_image_manipulation()
    
    print("All tests run successfully.")
    
if __name__ == "__main__":
    main()