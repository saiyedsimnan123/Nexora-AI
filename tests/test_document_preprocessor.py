import pytest

from nexora.document.preprocessor import clean_text


def test_basic_whitespace_cleaning():
    assert clean_text("  Hello world  ") == "Hello world"


def test_multiple_spaces_and_tabs_are_normalized():
    assert clean_text("Deep \t\t  learning   for\tNLP") == "Deep learning for NLP"


def test_excessive_blank_lines_are_reduced():
    result = clean_text("First\n\n\n\n\nSecond")
    assert result == "First\n\nSecond"


def test_paragraph_separation_is_preserved():
    result = clean_text("Paragraph one.\n\nParagraph two.")
    assert result == "Paragraph one.\n\nParagraph two."


def test_leading_and_trailing_whitespace_is_removed():
    assert clean_text("\n\n  \t Text here \t \n\n") == "Text here"


def test_whitespace_only_blank_lines_count_as_blank():
    result = clean_text("A\n   \n\t\n   \nB")
    assert result == "A\n\nB"


def test_windows_and_old_style_line_endings_are_normalized():
    result = clean_text("Line one\r\nLine two\rLine three")
    assert result == "Line one\nLine two\nLine three"


def test_punctuation_numbers_and_symbols_are_preserved():
    text = "Accuracy: 94.5% (p < 0.05); see [12], Eq. (3): y = wx + b, α ≥ β."
    assert clean_text(text) == text


def test_empty_input_is_handled():
    assert clean_text("") == ""
    assert clean_text("   \n\t  \n") == ""


@pytest.mark.parametrize("bad_input", [None, 123, 4.5, ["text"], b"text"])
def test_non_string_input_raises_type_error(bad_input):
    with pytest.raises(TypeError, match="expects a str"):
        clean_text(bad_input)


def test_cleaning_is_deterministic():
    text = "A  b\r\n\r\n\r\nC\t d "
    assert clean_text(text) == clean_text(text)
