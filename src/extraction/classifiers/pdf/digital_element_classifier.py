"""Identification utilities for elements within digitally generated PDFs.

The functions in this module categorise high-level page elements such as
text blocks, tables and images. No processing of the extracted content is
performed here; downstream modules should handle each element type
separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import pdfplumber
from pdfplumber.page import Page

ElementType = Dict[str, List]


@dataclass
class DigitalElementClassifier:
    """Locate simple elements within a digital PDF, treating it as one document."""

    def classify(self, pdf_path: str) -> ElementType:

        """Return a dictionary describing all element types in the PDF.

        Contains the keys ``text``, ``tables`` and ``images`` with
        metadata describing the location of each element. The goal is merely to
        identify their presence; detailed processing is delegated elsewhere.
        """
        all_text = []
        all_tables = []
        all_images = []

        with pdfplumber.open(pdf_path) as pdf:
            # Process all pages as one continuous document
            for page in pdf.pages:
                all_text.extend(self._extract_text_blocks(page))
                all_tables.extend(self._extract_tables(page))
                all_images.extend(self._extract_images(page))

        # Now we could potentially merge text blocks that span page boundaries
        # by checking if they have similar formatting and are positioned near page edges
        all_text = self._merge_cross_page_text_blocks(all_text)

        return {
            "text": all_text,
            "tables": all_tables,
            "images": all_images,
        }

    def _extract_text_blocks(self, page: Page) -> List[Dict[str, Any]]:
        """Extract cohesive text blocks using character-level information."""
        chars = page.chars
        
        if not chars:
            return []
        
        # Sort characters by position (top to bottom, left to right)
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        text_blocks = []
        current_block = []
        
        for char in sorted_chars:
            # Skip non-printable characters
            if not char.get('text', '').strip():
                continue
                
            char_info = {
                'text': char.get('text', ''),
                'font': char.get('fontname', ''),
                'size': char.get('size', 0),
                'x0': char.get('x0', 0),
                'y0': char.get('top', 0),
                'x1': char.get('x1', 0),
                'y1': char.get('bottom', 0),
                'bbox': (char.get('x0', 0), char.get('top', 0), char.get('x1', 0), char.get('bottom', 0))
            }
            
            # Group characters that are close together and have similar formatting
            if current_block and self._should_group_char(current_block[-1], char_info):
                current_block.append(char_info)
            else:
                if current_block:
                    text_blocks.append(self._merge_text_block(current_block))
                current_block = [char_info]
        
        # Don't forget the last block
        if current_block:
            text_blocks.append(self._merge_text_block(current_block))
            
        return text_blocks

    def _should_group_char(self, prev_char: Dict, curr_char: Dict) -> bool:
        """Determine if two characters should be grouped together."""
        # Group if they're within reasonable distance and have similar formatting
        vertical_gap = abs(curr_char['y0'] - prev_char['y0'])
        horizontal_gap = curr_char['x0'] - prev_char['x1']
        
        # Allow some tolerance for line breaks and spacing
        same_line = vertical_gap < 5  # Same line
        next_line = 5 <= vertical_gap < 25  # Next line (reasonable line height)
        
        # Check if the horizontal gap suggests a word boundary
        # If characters are too far apart horizontally, they're likely different words
        char_width = prev_char['x1'] - prev_char['x0']
        reasonable_spacing = horizontal_gap < max(15, char_width * 2.5)  # More conservative spacing
        
        font_similar = prev_char['font'] == curr_char['font']
        size_similar = abs(prev_char['size'] - curr_char['size']) < 1
        
        return (same_line or (next_line and reasonable_spacing)) and font_similar and size_similar

    def _merge_text_block(self, chars: List[Dict]) -> Dict[str, Any]:
        """Merge multiple characters into a single text block."""
        if not chars:
            return {}
            
        # Combine all text, preserving spaces by using the original spacing
        combined_text = self._reconstruct_text_with_spaces(chars)
        
        # Post-process the text to fix common spacing issues
        combined_text = self._fix_common_spacing_issues(combined_text)
        
        # Calculate bounding box
        x0 = min(char['x0'] for char in chars)
        y0 = min(char['y0'] for char in chars)
        x1 = max(char['x1'] for char in chars)
        y1 = max(char['y1'] for char in chars)
        
        # Use the most common font and size
        fonts = [char['font'] for char in chars]
        sizes = [char['size'] for char in chars]
        dominant_font = max(set(fonts), key=fonts.count) if fonts else ''
        dominant_size = max(set(sizes), key=sizes.count) if sizes else 0
        
        return {
            'text': combined_text,
            'bbox': (x0, y0, x1, y1),
            'font': dominant_font,
            'size': dominant_size,
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'char_count': len(chars)
        }

    def _reconstruct_text_with_spaces(self, chars: List[Dict]) -> str:
        """Reconstruct text with proper spacing based on character positions."""
        if not chars:
            return ""
        
        # Sort characters by position (left to right, top to bottom)
        sorted_chars = sorted(chars, key=lambda c: (c['y0'], c['x0']))
        
        result = []
        prev_char = None
        
        for char in sorted_chars:
            if prev_char is None:
                result.append(char['text'])
            else:
                # Check if we need to add a space
                vertical_gap = abs(char['y0'] - prev_char['y0'])
                horizontal_gap = char['x0'] - prev_char['x1']
                
                # Add space if characters are far apart horizontally on the same line
                # Use a more intelligent threshold based on character width
                if vertical_gap < 5:  # Same line
                    char_width = prev_char['x1'] - prev_char['x0']
                    # If gap is significantly larger than typical character spacing, add space
                    if horizontal_gap > max(3, char_width * 1.2):
                        result.append(' ')
                else:
                    # New line - add newline
                    result.append('\n')
                
                result.append(char['text'])
            
            prev_char = char
        
        return ''.join(result)

    def _fix_common_spacing_issues(self, text: str) -> str:
        """Fix common spacing issues in extracted text."""
        if not text:
            return text
        
        # Fix common patterns where spaces are missing
        import re
        
        # Add spaces before capital letters that follow lowercase letters (word boundaries)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add spaces before numbers that follow letters
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        
        # Add spaces after numbers that are followed by letters
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Fix common abbreviations and patterns
        text = re.sub(r'([a-z])(\()', r'\1 \2', text)  # Space before opening parenthesis
        text = re.sub(r'(\))([a-zA-Z])', r'\1 \2', text)  # Space after closing parenthesis
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])(\()', r'\1 \2', text)  # Space before opening bracket
        text = re.sub(r'(\])([a-zA-Z])', r'\1 \2', text)  # Space after closing bracket
        
        # Fix spacing around punctuation
        text = re.sub(r'([a-zA-Z])([,;:])', r'\1 \2', text)  # Space before punctuation
        text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text)  # Space after punctuation
        
        # Fix common word boundary issues
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # Better word boundary detection
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spaces around newlines
        text = re.sub(r' *\n *', '\n', text)
        
        # Fix common PDF ligatures and character issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
        
        return text.strip()

    def _merge_cross_page_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """Attempt to merge text blocks that might span page boundaries."""
        if not text_blocks:
            return text_blocks
            
        # Sort by position (assuming we can determine relative positioning across pages)
        # This is a simplified approach - you might want more sophisticated logic
        sorted_blocks = sorted(text_blocks, key=lambda b: (b['y0'], b['x0']))
        
        merged_blocks = []
        current_block = None
        
        for block in sorted_blocks:
            if current_block is None:
                current_block = block
            else:
                # Check if this block should be merged with the current one
                if self._should_merge_blocks(current_block, block):
                    current_block = self._merge_two_blocks(current_block, block)
                else:
                    merged_blocks.append(current_block)
                    current_block = block
        
        if current_block:
            merged_blocks.append(current_block)
            
        return merged_blocks

    def _should_merge_blocks(self, block1: Dict, block2: Dict) -> bool:
        """Determine if two text blocks should be merged (e.g., spanning page boundaries)."""
        # Check if they have similar formatting
        font_similar = block1['font'] == block2['font']
        size_similar = abs(block1['size'] - block2['size']) < 1
        
        # Check if they're positioned logically (this is where you'd implement
        # cross-page boundary detection)
        # For now, just check formatting similarity
        return font_similar and size_similar

    def _merge_two_blocks(self, block1: Dict, block2: Dict) -> Dict:
        """Merge two text blocks into one."""
        combined_text = block1['text'] + '\n' + block2['text']
        
        # Calculate combined bounding box
        x0 = min(block1['x0'], block2['x0'])
        y0 = min(block1['y0'], block2['y0'])
        x1 = max(block1['x1'], block2['x1'])
        y1 = max(block1['y1'], block2['y1'])
        
        return {
            'text': combined_text,
            'bbox': (x0, y0, x1, y1),
            'font': block1['font'],  # Use the first block's font
            'size': block1['size'],  # Use the first block's size
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'char_count': block1['char_count'] + block2['char_count']
        }

    def _extract_tables(self, page: Page) -> List[Dict[str, Any]]:
        """Extract table information with more metadata."""
        tables = []
        for table in page.find_tables():
            table_info = {
                'bbox': table.bbox,
                'rows': len(table.rows),
                'cols': len(table.rows[0]) if table.rows else 0,
                'extracted_text': table.extract()
            }
            tables.append(table_info)
        return tables

    def _extract_images(self, page: Page) -> List[Dict[str, Any]]:
        """Extract image information with positioning data."""
        images = []
        for img in page.images:
            # Fix the bbox extraction - page.images might not have bbox
            # Try to get position from the image object itself
            if hasattr(img, 'bbox') and img.bbox:
                bbox = img.bbox
            else:
                # Fallback: try to get position from page layout
                bbox = (0, 0, img.get('width', 0), img.get('height', 0))
            
            img_info = {
                'bbox': bbox,
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'type': img.get('type', 'unknown'),
                'x0': bbox[0],
                'y0': bbox[1],
                'x1': bbox[2],
                'y1': bbox[3]
            }
            images.append(img_info)
        return images