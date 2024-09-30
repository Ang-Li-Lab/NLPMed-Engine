from nlpmed_engine.components.encoding_fixer import EncodingFixer
from nlpmed_engine.data_structures.note import Note


def test_encoding_fixer() -> None:
    note = Note(text="This note has encoding issues â€“")
    fixer = EncodingFixer()

    # Process the note
    processed_note = fixer.process(note)

    # Assert that encoding has been fixed
    assert "–" in processed_note.text  # noqa: RUF001
    assert "â€“" not in processed_note.text
