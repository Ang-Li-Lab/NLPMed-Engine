from nlpmed_engine.components.section_filter import SectionFilter
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section


def test_inclusion_pattern() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Chief complaint", start_index=0, end_index=10),
            Section(text="History of present illness", start_index=11, end_index=30),
            Section(text="Physical exam", start_index=31, end_index=40),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that sections starting from "Chief complaint" to the end are kept
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Chief complaint"
    assert processed_note.sections[1].text == "History of present illness"
    assert processed_note.sections[2].text == "Physical exam"


def test_exclusion_pattern() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Chief complaint", start_index=0, end_index=10),
            Section(text="History of present illness", start_index=11, end_index=30),
            Section(text="Physical exam", start_index=31, end_index=40),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=["Physical"],
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that sections are kept until "Physical exam" is reached
    assert len(processed_note.sections) == 2  # noqa: PLR2004
    assert processed_note.sections[0].text == "Chief complaint"
    assert processed_note.sections[1].text == "History of present illness"


def test_fallback_true_no_sections_detected() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Random section", start_index=0, end_index=10),
            Section(text="Another section", start_index=11, end_index=30),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
        fallback=True,
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that the fallback keeps all sections when no inclusion pattern is matched
    assert len(processed_note.sections) == 2  # noqa: PLR2004
    assert processed_note.sections[0].text == "Random section"
    assert processed_note.sections[1].text == "Another section"


def test_fallback_false_no_sections_detected() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Random section", start_index=0, end_index=10),
            Section(text="Another section", start_index=11, end_index=30),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
        fallback=False,
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that no sections are returned if no inclusion pattern is matched and fallback is False
    assert len(processed_note.sections) == 0
