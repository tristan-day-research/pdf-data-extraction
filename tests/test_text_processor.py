import pytest
from text_processor import process_text


def test_process_text_not_implemented():
    with pytest.raises(NotImplementedError):
        process_text([])
