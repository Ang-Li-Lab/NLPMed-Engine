from nlpmed_engine.components.note_filter import NoteFilter
from nlpmed_engine.data_structures.note import Note


def test_note_filter_single_match() -> None:
    note = Note(text="The patient was diagnosed with DVT.")
    note_filter = NoteFilter(words_to_search=["DVT"])

    # Process the note
    processed_note = note_filter.process(note)

    # Assert that the note is returned since "DVT" is present
    assert processed_note is not None
    assert processed_note.text == note.text


def test_note_filter_multiple_matches() -> None:
    note = Note(text="The patient has symptoms of both DVT and PE.")
    note_filter = NoteFilter(words_to_search=["DVT", "PE"])

    # Process the note
    processed_note = note_filter.process(note)

    # Assert that the note is returned since both "DVT" and "PE" are present
    assert processed_note is not None
    assert processed_note.text == note.text


def test_note_filter_no_match() -> None:
    note = Note(text="The patient has a common cold.")
    note_filter = NoteFilter(words_to_search=["DVT", "PE"])

    # Process the note
    processed_note = note_filter.process(note)

    # Assert that None is returned since no keywords are found
    assert processed_note is None


def test_note_filter_partial_word_no_match() -> None:
    note = Note(text="The patient is experiencing partial paralysis.")
    note_filter = NoteFilter(words_to_search=["part"])

    # Process the note
    processed_note = note_filter.process(note)

    # Assert that None is returned since 'part' is only a partial word match
    assert processed_note is None


def test_note_filter_case_insensitive() -> None:
    note = Note(text="The patient was diagnosed with dvt.")
    note_filter = NoteFilter(words_to_search=["DVT"])

    # Process the note
    processed_note = note_filter.process(note)

    # Assert that the note is returned regardless of case
    assert processed_note is not None
    assert processed_note.text == note.text
