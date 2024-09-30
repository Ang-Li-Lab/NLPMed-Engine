from nlpmed_engine.components.pattern_replacer import PatternReplacer
from nlpmed_engine.data_structures.note import Note


def test_single_pattern_replacement() -> None:
    note = Note(text="This is a test pattern.")
    replacer = PatternReplacer(patterns=["test"], target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that the pattern 'test' has been replaced by '***'
    assert processed_note.text == "This is a *** pattern."


def test_multiple_pattern_replacement() -> None:
    note = Note(text="This is a test pattern with some text.")
    replacer = PatternReplacer(patterns=["test", "text"], target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that both 'test' and 'text' have been replaced by '***'
    assert processed_note.text == "This is a *** pattern with some ***."


def test_regex_pattern_replacement() -> None:
    note = Note(text="This is a test pattern with 1234 numbers.")
    replacer = PatternReplacer(patterns=[r"\d+"], target="###")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that the regex pattern for digits '\d+' has been replaced by '###'
    assert processed_note.text == "This is a test pattern with ### numbers."


def test_no_pattern_match() -> None:
    note = Note(text="This text has no matching pattern.")
    replacer = PatternReplacer(patterns=["notfound"], target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that no replacement occurs if the pattern does not match
    assert processed_note.text == "This text has no matching pattern."
