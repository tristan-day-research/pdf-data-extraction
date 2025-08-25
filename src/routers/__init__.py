"""Routing utilities for classified document elements.

The :mod:`routers.pdf` subpackage contains the PDF-specific implementation.
Other file types can register their routers alongside it in the future.
"""

from .pdf import route_elements

__all__ = ["route_elements"]
