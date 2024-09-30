from nlpmed_engine.components.section_splitter import SectionSplitter
from nlpmed_engine.data_structures.note import Note


def test_single_section() -> None:
    note = Note(text="This is a single section with no delimiter.")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that there is only one section
    assert len(processed_note.sections) == 1
    assert processed_note.sections[0].text == "This is a single section with no delimiter."
    assert processed_note.sections[0].start_index == 0
    assert processed_note.sections[0].end_index == len(note.text)


def test_multiple_sections() -> None:
    note = Note(text="Section 1\n\nSection 2\n\nSection 3")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that the note has been split into 3 sections
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Section 1"
    assert processed_note.sections[1].text == "Section 2"
    assert processed_note.sections[2].text == "Section 3"

    # Check start and end indices for each section
    assert processed_note.sections[0].start_index == 0
    assert processed_note.sections[0].end_index == len("Section 1")

    assert processed_note.sections[1].start_index == len("Section 1\n\n")
    assert processed_note.sections[1].end_index == len("Section 1\n\nSection 2")

    assert processed_note.sections[2].start_index == len("Section 1\n\nSection 2\n\n")
    assert processed_note.sections[2].end_index == len(
        "Section 1\n\nSection 2\n\nSection 3",
    )


def test_custom_delimiter() -> None:
    note = Note(text="Part 1##Part 2##Part 3")
    splitter = SectionSplitter(delimiter="##")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that the note has been split into 3 parts using the custom delimiter
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Part 1"
    assert processed_note.sections[1].text == "Part 2"
    assert processed_note.sections[2].text == "Part 3"


def test_no_sections_with_empty_note() -> None:
    note = Note(text="")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that no sections are created for an empty note
    assert len(processed_note.sections) == 0
