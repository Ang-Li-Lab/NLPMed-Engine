import pytest

from nlpmed_engine.components.sentence_expander import SentenceExpander
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.data_structures.sentence import Sentence


@pytest.fixture
def expander() -> SentenceExpander:
    return SentenceExpander(length_threshold=50)


def test_no_expansion_needed(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.append(
        Sentence(text="This sentence is long enough.", start_index=0, end_index=30),
    )
    section.important_indices = [0]  # Mark this sentence as important
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that the sentence remains unchanged as it's long enough
    assert len(processed_note.sections[0].expanded_indices) == 1
    assert (
        processed_note.sections[0].sentences[processed_note.sections[0].expanded_indices[0]].text
        == "This sentence is long enough."
    )


def test_expand_short_sentence_with_neighbors(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.extend(
        [
            Sentence(text="Short.", start_index=0, end_index=6),
            Sentence(text="A bit longer sentence.", start_index=7, end_index=28),
            Sentence(text="This is the third sentence.", start_index=29, end_index=55),
        ]
    )
    section.important_indices = [0]  # Only the first sentence is important initially
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that the short sentence is expanded by combining with neighboring sentences
    assert len(processed_note.sections[0].expanded_indices) == 3  # noqa: PLR2004
    expanded_sentence_text = "Short. A bit longer sentence. This is the third sentence."
    assert (
        " ".join(
            [processed_note.sections[0].sentences[idx].text for idx in processed_note.sections[0].expanded_indices]
        )
        == expanded_sentence_text
    )


def test_expand_middle_short_sentence(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.extend(
        [
            Sentence(text="Long sentence before.", start_index=0, end_index=25),
            Sentence(text="Short.", start_index=26, end_index=32),
            Sentence(text="Another long sentence after.", start_index=33, end_index=60),
        ]
    )
    section.important_indices = [1]  # Only the second sentence is important
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that the short sentence in the middle is expanded by combining with its neighbors
    assert len(processed_note.sections[0].expanded_indices) == 3  # noqa: PLR2004
    expanded_sentence_text = "Long sentence before. Short. Another long sentence after."
    assert (
        " ".join(
            [processed_note.sections[0].sentences[idx].text for idx in processed_note.sections[0].expanded_indices]
        )
        == expanded_sentence_text
    )


def test_expand_sentence_reaches_threshold(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.extend(
        [
            Sentence(text="Short.", start_index=0, end_index=6),
            Sentence(text="Another sentence.", start_index=7, end_index=24),
        ]
    )
    section.important_indices = [0]  # Only the first sentence is important
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that the short sentence is expanded until the threshold is reached
    assert len(processed_note.sections[0].expanded_indices) == 2  # noqa: PLR2004
    expanded_sentence_text = "Short. Another sentence."
    assert (
        " ".join(
            [processed_note.sections[0].sentences[idx].text for idx in processed_note.sections[0].expanded_indices]
        )
        == expanded_sentence_text
    )


def test_no_expansion_for_only_short_sentence(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.append(
        Sentence(text="Short.", start_index=0, end_index=6),
    )
    section.important_indices = [0]  # Mark the only sentence as important
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that no expansion happens when only one short sentence exists
    assert len(processed_note.sections[0].expanded_indices) == 1
    assert processed_note.sections[0].sentences[processed_note.sections[0].expanded_indices[0]].text == "Short."


def test_two_short_sentences_in_one_section_with_expansion_no_overlap(
    expander: SentenceExpander,
) -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=100)
    section.sentences.extend(
        [
            Sentence(text="Short1.", start_index=0, end_index=7),
            Sentence(text="Another short.", start_index=8, end_index=22),
            Sentence(text="A middle sentence.", start_index=23, end_index=40),
            Sentence(text="Yet another short.", start_index=41, end_index=59),
            Sentence(text="Final sentence.", start_index=60, end_index=75),
        ]
    )
    section.important_indices = [0, 3]  # Mark two short sentences as important
    note.sections.append(section)

    # Process the note
    processed_note = expander.process(note)

    # Assert that the short sentences are expanded without overlaps
    assert len(processed_note.sections[0].expanded_indices) == 5  # noqa: PLR2004
    expanded_sentence_text = "Short1. Another short. A middle sentence. Yet another short. Final sentence."
    assert (
        " ".join(
            [processed_note.sections[0].sentences[idx].text for idx in processed_note.sections[0].expanded_indices]
        )
        == expanded_sentence_text
    )

    # Ensure that no overlaps occur during expansion (verify start and end indices)
    assert processed_note.sections[0].sentences[0].start_index == 0
    assert processed_note.sections[0].sentences[0].end_index == 7  # noqa: PLR2004

    assert processed_note.sections[0].sentences[1].start_index == 8  # noqa: PLR2004
    assert processed_note.sections[0].sentences[1].end_index == 22  # noqa: PLR2004

    assert processed_note.sections[0].sentences[2].start_index == 23  # noqa: PLR2004
    assert processed_note.sections[0].sentences[2].end_index == 40  # noqa: PLR2004

    assert processed_note.sections[0].sentences[3].start_index == 41  # noqa: PLR2004
    assert processed_note.sections[0].sentences[3].end_index == 59  # noqa: PLR2004

    assert processed_note.sections[0].sentences[4].start_index == 60  # noqa: PLR2004
    assert processed_note.sections[0].sentences[4].end_index == 75  # noqa: PLR2004


def test_two_short_sentences_in_different_sections(expander: SentenceExpander) -> None:
    note = Note(text="Note text")
    section1 = Section(text="First section text", start_index=0, end_index=50)
    section2 = Section(text="Second section text", start_index=51, end_index=100)
    section1.sentences.append(
        Sentence(text="Short1.", start_index=0, end_index=7),
    )
    section1.important_indices = [0]  # Mark this sentence as important

    section2.sentences.append(
        Sentence(text="Short2.", start_index=51, end_index=58),
    )
    section2.important_indices = [0]  # Mark this sentence as important
    note.sections.extend([section1, section2])

    # Process the note
    processed_note = expander.process(note)

    # Assert that no combination happens across sections
    assert len(processed_note.sections[0].expanded_indices) == 1
    assert len(processed_note.sections[1].expanded_indices) == 1
    assert processed_note.sections[0].sentences[processed_note.sections[0].expanded_indices[0]].text == "Short1."
    assert processed_note.sections[1].sentences[processed_note.sections[1].expanded_indices[0]].text == "Short2."
