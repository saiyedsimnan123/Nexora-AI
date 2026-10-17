"""PDF loading and structured text extraction for Nexora AI.

This module reads a PDF file and extracts its text and page structure.
It preserves page numbering and document metadata for traceable evidence provenance.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any


def _ensure_windows_runtime_dlls() -> None:
    """Ensure Microsoft Visual C++ runtime DLLs are accessible on Windows."""
    if sys.platform != "win32":
        return
    try:
        ctypes.CDLL("msvcp140.dll")
        return
    except Exception:
        pass

    # Search common fallback locations for msvcp140.dll on Windows
    candidate_roots = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application",
        r"C:\Program Files (x86)\Microsoft\EdgeCore",
        r"C:\Program Files\Microsoft\Edge\Application",
    ]
    for root in candidate_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if "msvcp140.dll" in filenames:
                    try:
                        dll_path = os.path.join(dirpath, "msvcp140.dll")
                        ctypes.CDLL(dll_path)
                        if hasattr(os, "add_dll_directory"):
                            os.add_dll_directory(dirpath)
                        return
                    except Exception:
                        continue


_ensure_windows_runtime_dlls()

try:
    import fitz  # PyMuPDF
except ImportError as _fitz_import_err:
    fitz = None  # type: ignore[assignment]
    _FITZ_ERROR = _fitz_import_err
else:
    _FITZ_ERROR = None  # type: ignore[assignment]


def _require_fitz() -> Any:
    """Return the imported fitz module or raise an informative error."""
    if fitz is None:
        raise ImportError(
            f"PyMuPDF (fitz) is not available: {_FITZ_ERROR}. "
            "On Windows, ensure the Microsoft Visual C++ 2015-2022 Redistributable (msvcp140.dll) is installed."
        ) from _FITZ_ERROR
    return fitz


@dataclass(frozen=True)
class DocumentPage:
    """One extracted page of a document with 1-based page numbering.

    Attributes:
        page_number: 1-based index of the page within the document.
        text: Raw extracted text of the page.
    """

    page_number: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
            raise TypeError(f"page_number must be an int, got {type(self.page_number).__name__}")
        if self.page_number <= 0:
            raise ValueError(f"page_number must be a positive integer, got {self.page_number}")
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")


@dataclass(frozen=True)
class ExtractedDocument:
    """Structured extraction result preserving page provenance and metadata.

    Attributes:
        source: Path or identifier of the source document.
        pages: Ordered list of DocumentPage instances.
        title: Document title from PDF metadata if present, else None.
        author: Document author from PDF metadata if present, else None.
        metadata: Document properties (page count, format, creation date).
    """

    source: str
    pages: list[DocumentPage]
    title: str | None = None
    author: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(self.pages, (list, tuple)):
            raise TypeError("pages must be a list or tuple of DocumentPage")
        for p in self.pages:
            if not isinstance(p, DocumentPage):
                raise TypeError(f"pages must contain DocumentPage instances, got {type(p).__name__}")
        if self.title is not None and (not isinstance(self.title, str) or not self.title.strip()):
            raise ValueError("title must be a non-empty string or None")
        if self.author is not None and (not isinstance(self.author, str) or not self.author.strip()):
            raise ValueError("author must be a non-empty string or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        object.__setattr__(self, "pages", list(self.pages))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_pages(self) -> int:
        """Total number of pages in the document."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Combined text across all pages in page order."""
        return "\n".join(p.text for p in self.pages)

    @property
    def has_extractable_text(self) -> bool:
        """True if at least one page contains non-whitespace text."""
        return any(bool(p.text.strip()) for p in self.pages)


def load_pdf_document(
    file_path: str,
    *,
    require_extractable_text: bool = False,
) -> ExtractedDocument:
    """Extract page structure, text, and genuine metadata from a PDF file.

    Args:
        file_path: Path to a PDF file.
        require_extractable_text: If True, raises ValueError if the document contains
            no extractable text (e.g. scanned image-only PDF without OCR).

    Returns:
        An ExtractedDocument containing pages in sequential order.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to an existing file.
        ValueError: If the file is not a valid PDF, is password-protected, has 0 pages,
            or contains no extractable text when required.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    mupdf = _require_fitz()

    try:
        with mupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"PDF is password-protected: {path}")
            if len(document) == 0:
                raise ValueError(f"PDF document is empty (0 pages): {path}")

            pages = [
                DocumentPage(page_number=i + 1, text=page.get_text())
                for i, page in enumerate(document)
            ]

            doc_meta = document.metadata or {}
            raw_title = doc_meta.get("title")
            title = str(raw_title).strip() if raw_title and str(raw_title).strip() else None
            raw_author = doc_meta.get("author")
            author = str(raw_author).strip() if raw_author and str(raw_author).strip() else None

            extra_meta: dict[str, object] = {
                "page_count": len(pages),
                "format": doc_meta.get("format", "PDF"),
            }
            if doc_meta.get("creationDate"):
                extra_meta["creation_date"] = str(doc_meta.get("creationDate"))
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not read file as a PDF: {path} ({error})") from error

    extracted = ExtractedDocument(
        source=str(path),
        pages=pages,
        title=title,
        author=author,
        metadata=extra_meta,
    )

    if require_extractable_text and not extracted.has_extractable_text:
        raise ValueError(
            f"PDF document contains no extractable text (empty or scanned image-only PDF): {path}"
        )

    return extracted


def load_pdf_pages(file_path: str) -> list[DocumentPage]:
    """Extract individual pages of a PDF with 1-based page numbering.

    Args:
        file_path: Path to a PDF file.

    Returns:
        A list of DocumentPage instances in page order.
    """
    return load_pdf_document(file_path, require_extractable_text=False).pages


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
    doc = load_pdf_document(file_path, require_extractable_text=False)
    return doc.full_text
