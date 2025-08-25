import pytest
from extraction.processors.pdf import process_image


def test_process_image_not_implemented():
    with pytest.raises(NotImplementedError):
        process_image(None)
