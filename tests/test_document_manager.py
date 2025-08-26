import shutil
from pathlib import Path
import sys

# Ensure local packages are importable before any external 'config' package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DataSettings
from config import settings

# Import DocumentManager without triggering heavy package imports
import importlib.util

doc_manager_path = Path(__file__).resolve().parents[1] / "src/extraction/utils/document_manager.py"
spec = importlib.util.spec_from_file_location("document_manager", doc_manager_path)
document_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(document_manager_module)
DocumentManager = document_manager_module.DocumentManager


def test_process_from_unprocessed_skips_existing(tmp_path, monkeypatch):
    # Set up temporary directories for data paths
    data_root = tmp_path / "data"
    unprocessed_dir = data_root / "unprocessed_documents"
    documents_dir = data_root / "documents"
    temp_dir = tmp_path / "temp"
    unprocessed_dir.mkdir(parents=True)
    documents_dir.mkdir(parents=True)
    temp_dir.mkdir(parents=True)

    # Override global data settings to use temporary paths
    data_settings = DataSettings(
        data_dir=data_root,
        documents_dir=documents_dir,
        temp_dir=temp_dir,
    )
    monkeypatch.setattr(settings, "data", data_settings, raising=False)

    # Create a minimal PDF in the unprocessed directory
    sample_pdf = unprocessed_dir / "sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF")

    manager = DocumentManager()

    # First pass should process the file
    processed = manager.process_from_unprocessed(str(unprocessed_dir))
    assert processed == ["sample"]
    assert (documents_dir / "sample").exists()

    # Second pass should detect existing folder and skip
    processed_again = manager.process_from_unprocessed(str(unprocessed_dir))
    assert processed_again == []
