"""PDF loading for Nexora AI.

This module reads a PDF file and returns its text. It does not split the
text into chunks or analyse it; those are separate, later steps.
"""

from pathlib import Path

import fitz  # PyMuPDF


def load_pdf_text(file_path: str) -> str:
    """Extract the text of every page of a PDF, in page order.

    Args:
        file_path: Path to a PDF file.

    Returns:
        The text of all pages combined into one string. Pages are separated
        by a newline and appear in the same order as in the PDF.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to an existing file.
        ValueError: If the file cannot be read as a PDF (for example it is
            corrupted, empty, not a PDF, or password-protected).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        # The "with" block closes the document automatically, even on errors.
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"PDF is password-protected: {path}")
            pages = [page.get_text() for page in document]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not read file as a PDF: {path} ({error})") from error

    return "\n".join(pages)
