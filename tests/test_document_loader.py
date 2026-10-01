import fitz  # PyMuPDF
import pytest

from nexora.document.loader import load_pdf_text


def make_pdf(path, page_texts):
    """Create a small PDF with one page per string in ``page_texts``."""
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_loads_text_from_a_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["Hello Nexora"])

    text = load_pdf_text(str(pdf_path))

    assert "Hello Nexora" in text


def test_extracts_text_from_multiple_pages(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    make_pdf(pdf_path, ["First page text", "Second page text", "Third page text"])

    text = load_pdf_text(str(pdf_path))

    assert "First page text" in text
    assert "Second page text" in text
    assert "Third page text" in text


def test_preserves_page_order(tmp_path):
    pdf_path = tmp_path / "ordered.pdf"
    make_pdf(pdf_path, ["AAA first", "BBB second", "CCC third"])

    text = load_pdf_text(str(pdf_path))

    assert text.index("AAA first") < text.index("BBB second") < text.index("CCC third")


def test_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf_text(str(missing_path))


def test_file_that_is_not_a_pdf_raises_value_error(tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("this is not a real pdf")

    with pytest.raises(ValueError):
        load_pdf_text(str(fake_pdf))


def test_pdf_with_text_does_not_return_empty_result(tmp_path):
    pdf_path = tmp_path / "text.pdf"
    make_pdf(pdf_path, ["Some real content"])

    text = load_pdf_text(str(pdf_path))

    assert text.strip() != ""
