import pytest
from extraction.processors.pdf import process_table


def test_process_table_not_implemented():
    with pytest.raises(NotImplementedError):
        process_table(None)
