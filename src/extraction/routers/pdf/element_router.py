"""Route identified PDF elements to their respective processors.

This module demonstrates how different element types can be dispatched to
separate processing modules. The processors themselves currently contain
placeholders and will be implemented in later iterations.
"""
from __future__ import annotations

from typing import Iterable

from ...processors.pdf import image_processor, table_processor, text_processor


class ElementRouter:
    """Routes identified PDF elements to their respective processors."""
    
    def route_elements(self, pages: Iterable[dict]) -> None:
        """Dispatch elements from ``DigitalElementClassifier`` results.

        Parameters
        ----------
        pages:
            An iterable of dictionaries as returned by
            :class:`DigitalElementClassifier`, where each dictionary contains
            ``text``, ``tables`` and ``images`` keys.
        """
        for page in pages:
            if page["text"]:
                text_processor.process_text(page["text"])
            for table in page["tables"]:
                table_processor.process_table(table)
            for image in page["images"]:
                image_processor.process_image(image)
