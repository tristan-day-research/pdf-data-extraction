#!/usr/bin/env python3
"""
Example script demonstrating the new document-centric structure.

This script shows how to:
1. Create a document folder structure
2. Process a PDF and save elements
3. Organize data for RAG systems
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import settings
from extraction.classifiers.pdf import DigitalElementClassifier
from extraction.utils.document_manager import DocumentManager


def process_document_example():
    """Example of processing a document using the new structure."""
    
    # Initialize document manager
    doc_manager = DocumentManager()
    
    # Example PDF path (update this to point to an actual PDF)
    pdf_path = "../data/Ripples of Consciousenss.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    # Create a document ID (you could use filename, hash, or any unique identifier)
    document_id = "ripples_consciousness_2016"
    
    print(f"Processing document: {document_id}")
    print(f"PDF path: {pdf_path}")
    
    # 1. Create document folder structure
    print("\n1. Creating document folder structure...")
    doc_path = doc_manager.create_document(document_id, pdf_path)
    print(f"Document folder created at: {doc_path}")
    
    # 2. Process the PDF
    print("\n2. Processing PDF...")
    classifier = DigitalElementClassifier()
    elements = classifier.classify(pdf_path, document_id=document_id)
    
    print(f"Extracted {len(elements['text'])} text blocks")
    print(f"Extracted {len(elements['tables'])} tables")
    print(f"Extracted {len(elements['images'])} images")
    
    # 3. Save extracted elements
    print("\n3. Saving extracted elements...")
    doc_manager.save_elements(document_id, elements)
    
    # 4. Show document structure
    print("\n4. Document structure created:")
    show_document_structure(document_id)
    
    # 5. Show document info
    print("\n5. Document metadata:")
    doc_info = doc_manager.get_document_info(document_id)
    if doc_info:
        for key, value in doc_info.items():
            print(f"  {key}: {value}")
    
    print(f"\nDocument processing complete! Check the folder: {doc_path}")


def show_document_structure(document_id: str):
    """Display the created document folder structure."""
    doc_manager = DocumentManager()
    doc_path = doc_manager.data_settings.get_document_path(document_id)
    
    def print_tree(path: Path, prefix: str = "", is_last: bool = True):
        """Recursively print directory tree."""
        if not path.exists():
            return
        
        # Print current item
        marker = "└── " if is_last else "├── "
        print(f"{prefix}{marker}{path.name}")
        
        if path.is_dir():
            # Get all items in directory
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, new_prefix, is_last_item)
    
    print_tree(doc_path)


def list_all_documents():
    """List all documents in the system."""
    doc_manager = DocumentManager()
    documents = doc_manager.list_documents()
    
    print("Documents in system:")
    if not documents:
        print("  No documents found")
        return
    
    for doc_id in documents:
        doc_info = doc_manager.get_document_info(doc_id)
        status = doc_info.get("status", "unknown") if doc_info else "unknown"
        print(f"  {doc_id}: {status}")


if __name__ == "__main__":
    print("Document Processing Example")
    print("=" * 40)
    
    # Show existing documents
    list_all_documents()
    
    # Process a new document
    process_document_example()
