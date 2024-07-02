import asyncio
import pytest
from unittest.mock import Mock, call
from PIL import Image
from collections import namedtuple

class TestClass:
    def __init__(self): 
        self.process = ImageProcessor()
        self.pdf = PDFProcessor()

    async def pointless_function(self, image): 
        return await self.process.process_image(image)

    async def another_pointless_function(self, pdf): 
        return await self.pdf.process_pdf(pdf)

@pytest.fixture
def mock_func():
    with patch.object(TestClass, "pointless_function", new_callable=AsyncMock) as _mock:
        yield _mock

@pytest.mark.asyncio
async def test_mocking_async_methods(mock_func):
    mk = TestClass()
    mock_func.side_effect = asyncio.sleep(1)
    await mk.pointless_function('image')
    mock_func.assert_called_with('image')

@pytest.fixture
def another_mock_func():
    with patch.object(TestClass, "another_pointless_function", new_callable=AsyncMock) as _mock:
        yield _mock

@pytest.mark.asyncio
async def test_mocking_another_async_methods(another_mock_func):
    mk = TestClass()
    another_mock_func.side_effect = asyncio.sleep(1)
    await mk.another_pointless_function('pdf')
    another_mock_func.assert_called_with('pdf')