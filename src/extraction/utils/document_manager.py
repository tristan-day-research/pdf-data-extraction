"""
Document management utilities for the PDF Data Extraction system.

This module provides utilities for working with the document-centric folder structure.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import settings


class DocumentManager:
    """Manages document folders and file operations."""
    
    def __init__(self):
        self.data_settings = settings.data
    
    def create_document(self, document_id: str, pdf_path: Optional[str] = None) -> Path:
        """
        Create a new document folder structure.
        
        Args:
            document_id: Unique identifier for the document
            pdf_path: Optional path to copy the PDF into the raw folder
            
        Returns:
            Path to the created document folder
        """
        # Create the folder structure
        self.data_settings.create_document_structure(document_id)
        
        # Copy PDF if provided
        if pdf_path and os.path.exists(pdf_path):
            pdf_name = os.path.basename(pdf_path)
            dest_path = self.data_settings.get_document_raw_path(document_id) / pdf_name
            shutil.copy2(pdf_path, dest_path)
            
            # Create initial metadata
            self._create_initial_metadata(document_id, pdf_name)
            
            print(f"📄 Copied {pdf_name} to {dest_path}")
        else:
            print(f"📁 Created document structure for '{document_id}' (no PDF copied)")
        
        return self.data_settings.get_document_path(document_id)
    
    def _create_initial_metadata(self, document_id: str, pdf_filename: str) -> None:
        """Create initial metadata file for a document."""
        metadata = {
            "document_id": document_id,
            "pdf_filename": pdf_filename,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "processing_steps": []
        }
        
        metadata_path = self.data_settings.get_document_index_path(document_id) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_elements(self, document_id: str, elements: Dict[str, Any]) -> None:
        """
        Save extracted elements to the document's elements folder.
        
        Args:
            document_id: Document identifier
            elements: Dictionary containing text, tables, images, etc.
        """
        elements_path = self.data_settings.get_document_elements_path(document_id)
        
        # Save text blocks
        if 'text' in elements:
            text_path = elements_path / "text_blocks.json"
            with open(text_path, 'w') as f:
                json.dump(elements['text'], f, indent=2)
        
        # Save tables
        if 'tables' in elements:
            tables_path = elements_path / "tables.json"
            with open(tables_path, 'w') as f:
                json.dump(elements['tables'], f, indent=2)
        
        # Save images metadata (images themselves are saved by the classifier)
        if 'images' in elements:
            images_metadata = []
            for img in elements['images']:
                # Filter out non-serializable objects from metadata
                clean_metadata = {}
                for key, value in img.get("metadata", {}).items():
                    if key not in ["stream"]:  # Skip PDF stream objects
                        try:
                            json.dumps(value)  # Test if serializable
                            clean_metadata[key] = value
                        except (TypeError, ValueError):
                            clean_metadata[key] = str(value)  # Convert to string if not serializable
                
                img_meta = {
                    "id": img.get("id"),
                    "bbox": img.get("bboxes_per_page", [{}])[0].get("bbox"),
                    "page": img.get("page_range", [0])[0],
                    "metadata": clean_metadata
                }
                images_metadata.append(img_meta)
            
            images_path = elements_path / "images_metadata.json"
            with open(images_path, 'w') as f:
                json.dump(images_metadata, f, indent=2)
        
        # Update processing status
        self._update_processing_status(document_id, "elements_extracted")
    
    def save_embeddings(self, document_id: str, embeddings: Dict[str, Any]) -> None:
        """
        Save vector embeddings to the document's embeddings folder.
        
        Args:
            document_id: Document identifier
            embeddings: Dictionary containing text and table embeddings
        """
        embeddings_path = self.data_settings.get_document_embeddings_path(document_id)
        
        for embedding_type, embedding_data in embeddings.items():
            embedding_path = embeddings_path / f"{embedding_type}_embeddings.json"
            with open(embedding_path, 'w') as f:
                json.dump(embedding_data, f, indent=2)
        
        # Update processing status
        self._update_processing_status(document_id, "embeddings_created")
    
    def save_index(self, document_id: str, index_data: Dict[str, Any]) -> None:
        """
        Save document index data.
        
        Args:
            document_id: Document identifier
            index_data: Index information for RAG retrieval
        """
        index_path = self.data_settings.get_document_index_path(document_id) / "document_index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # Update processing status
        self._update_processing_status(document_id, "indexed")
    
    def _update_processing_status(self, document_id: str, step: str) -> None:
        """Update the processing status in metadata."""
        metadata_path = self.data_settings.get_document_index_path(document_id) / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"document_id": document_id, "processing_steps": []}
        
        metadata["processing_steps"].append({
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["status"] = step
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        metadata_path = self.data_settings.get_document_index_path(document_id) / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs.
        
        Returns:
            List of document identifiers
        """
        if not self.data_settings.documents_dir.exists():
            return []
        
        return [d.name for d in self.data_settings.documents_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its data.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_path = self.data_settings.get_document_path(document_id)
            if doc_path.exists():
                shutil.rmtree(doc_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_document_size(self, document_id: str) -> Dict[str, int]:
        """
        Get the size of a document's data.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with sizes for different data types
        """
        doc_path = self.data_settings.get_document_path(document_id)
        if not doc_path.exists():
            return {}
        
        sizes = {}
        
        # Raw files size
        raw_path = self.data_settings.get_document_raw_path(document_id)
        if raw_path.exists():
            sizes["raw"] = sum(f.stat().st_size for f in raw_path.rglob('*') if f.is_file())
        
        # Processed data size
        processed_path = self.data_settings.get_document_processed_path(document_id)
        if processed_path.exists():
            sizes["processed"] = sum(f.stat().st_size for f in processed_path.rglob('*') if f.is_file())
        
        return sizes
    
    def process_from_unprocessed(self, unprocessed_dir: str = "data/unprocessed_documents") -> List[str]:
        """
        Process all PDFs from the unprocessed documents directory.
        
        Args:
            unprocessed_dir: Path to directory containing unprocessed PDFs
            
        Returns:
            List of document IDs that were processed
        """
        import glob
        
        unprocessed_path = Path(unprocessed_dir)
        if not unprocessed_path.exists():
            print(f"❌ Unprocessed directory not found: {unprocessed_path}")
            return []
        
        # Find all PDF files
        pdf_files = glob.glob(str(unprocessed_path / "*.pdf"))
        if not pdf_files:
            print(f"📭 No PDF files found in {unprocessed_path}")
            return []
        
        processed_docs = []
        print(f"🔍 Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            # Create document ID from filename (remove .pdf extension)
            pdf_name = Path(pdf_file).stem
            document_id = f"{pdf_name}_{int(datetime.now().timestamp())}"
            
            print(f"\n📋 Processing: {pdf_name}")
            print(f"🆔 Document ID: {document_id}")
            
            # Create document structure and copy PDF
            doc_path = self.create_document(document_id, pdf_file)
            processed_docs.append(document_id)
            
            print(f"✅ Document structure created at: {doc_path}")
        
        return processed_docs
