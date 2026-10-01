"""Text preprocessing for the Nexora AI Document Intelligence layer.

Raw text extracted from PDFs often contains inconsistent line endings,
runs of spaces/tabs, and long stretches of blank lines. This module
applies light, conservative cleaning so the text is ready for chunking
without altering its content or meaning.
"""

import re

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Clean raw PDF-extracted text without changing its meaning.

    Steps (applied in order):
        1. Normalize line endings (``\\r\\n`` and ``\\r``) to ``\\n``.
        2. Collapse runs of spaces/tabs inside each line to a single space.
        3. Strip leading/trailing whitespace from every line.
        4. Collapse three or more consecutive newlines into exactly two,
           so paragraph boundaries (one blank line) are preserved.
        5. Strip leading/trailing whitespace from the whole text.

    Punctuation, numbers, symbols, equations-as-text, citations, and
    technical terminology are left untouched. The function is
    deterministic and uses only the standard library.

    Args:
        text: Raw text, typically extracted from a PDF.

    Returns:
        The cleaned text. An empty or whitespace-only string returns "".

    Raises:
        TypeError: If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"clean_text expects a str, got {type(text).__name__}"
        )

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in text.split("\n")
    ]
    text = "\n".join(lines)

    text = _EXCESS_NEWLINES.sub("\n\n", text)

    return text.strip()
