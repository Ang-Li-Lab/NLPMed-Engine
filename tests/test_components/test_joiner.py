import pytest

from nlpmed_engine.components.joiner import Joiner
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.data_structures.sentence import Sentence


@pytest.fixture
def joiner() -> Joiner:
    # Create an instance of Joiner with default delimiters
    return Joiner()


def test_joiner_with_important_sentences(joiner: Joiner) -> None:
    # Create a note with sections and important sentences
    note = Note(text="Test note")
    section1 = Section(text="Section 1 text", start_index=0, end_index=50)
    section2 = Section(text="Section 2 text", start_index=51, end_index=100)

    # Mark sections as important
    section1.is_important = True
    section2.is_important = True

    # Add sentences to sections and mark important indices
    section1.sentences.extend(
        [
            Sentence(text="First important sentence.", start_index=0, end_index=27),
            Sentence(text="Second important sentence.", start_index=28, end_index=56),
        ]
    )
    section1.important_indices = [0, 1]  # Mark both sentences as important

    section2.sentences.append(Sentence(text="Another important sentence.", start_index=0, end_index=30))
    section2.important_indices = [0]  # Mark the only sentence as important

    # Add sections to the note
    note.sections.extend([section1, section2])

    # Process the note with the Joiner component
    processed_note = joiner.process(note)

    # Assert that the preprocessed text is correctly joined
    expected_text = "First important sentence.\nSecond important sentence.\n\nAnother important sentence."
    assert processed_note.preprocessed_text == expected_text


def test_joiner_with_custom_delimiters(joiner: Joiner) -> None:
    # Create a note with sections and important sentences
    note = Note(text="Test note")
    section1 = Section(text="Section 1 text", start_index=0, end_index=50)
    section1.is_important = True

    # Add sentences and mark important indices
    section1.sentences.extend(
        [
            Sentence(text="First sentence.", start_index=0, end_index=14),
            Sentence(text="Second sentence.", start_index=15, end_index=30),
        ]
    )
    section1.important_indices = [0, 1]  # Mark both sentences as important

    # Add the section to the note
    note.sections.append(section1)

    # Process the note with custom delimiters
    processed_note = joiner.process(
        note,
        sentence_delimiter=" | ",
        section_delimiter=" || ",
    )

    # Assert that the preprocessed text is correctly joined with custom delimiters
    expected_text = "First sentence. | Second sentence."
    assert processed_note.preprocessed_text == expected_text


def test_joiner_with_no_important_sentences(joiner: Joiner) -> None:
    # Create a note with a section that has no important sentences
    note = Note(text="Test note")
    section = Section(text="Section text", start_index=0, end_index=50)
    section.is_important = True  # Mark section as important but no important sentences
    section.sentences.extend(
        [
            Sentence(text="Some sentence.", start_index=0, end_index=13),
        ]
    )
    note.sections.append(section)

    # Process the note
    processed_note = joiner.process(note)

    # Assert that the preprocessed text is empty since there are no important sentences
    assert processed_note.preprocessed_text == ""


def test_joiner_with_no_important_sections(joiner: Joiner) -> None:
    # Create a note with sections but no important ones
    note = Note(text="Test note")
    section1 = Section(text="Section 1 text", start_index=0, end_index=50)
    section2 = Section(text="Section 2 text", start_index=51, end_index=100)

    # Add sentences but don't mark sections as important
    section1.sentences.append(
        Sentence(text="Some sentence.", start_index=0, end_index=13),
    )
    section1.important_indices = [0]

    section2.sentences.append(
        Sentence(text="Another sentence.", start_index=0, end_index=16),
    )
    section2.important_indices = [0]

    # Add sections to the note
    note.sections.extend([section1, section2])

    # Process the note
    processed_note = joiner.process(note)

    # Assert that the preprocessed text is empty since no sections are important
    assert processed_note.preprocessed_text == ""
