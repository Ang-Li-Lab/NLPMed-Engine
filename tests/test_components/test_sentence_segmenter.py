import pytest

from nlpmed_engine.components.sentence_segmenter import SentenceSegmenter
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section


@pytest.fixture(scope="module")
def sentence_segmenter() -> SentenceSegmenter:
    # Use the actual model "en_core_sci_lg"
    return SentenceSegmenter(model_name="en_core_sci_lg")


def test_single_section_with_sentences(sentence_segmenter: SentenceSegmenter) -> None:
    note = Note(text="This is the first sentence. This is the second sentence.")
    section = Section(
        text="This is the first sentence. This is the second sentence.",
        start_index=0,
        end_index=55,
    )
    note.sections.append(section)

    # Process the note
    processed_note = sentence_segmenter.process(note)

    # Assert that the section has been split into 2 sentences
    assert len(processed_note.sections[0].sentences) == 2  # noqa: PLR2004

    first_sentence = processed_note.sections[0].sentences[0]
    second_sentence = processed_note.sections[0].sentences[1]

    # Check the text of the sentences
    assert first_sentence.text == "This is the first sentence."
    assert second_sentence.text == "This is the second sentence."

    # Check the start and end indices
    assert first_sentence.start_index == 0
    assert first_sentence.end_index == 27  # noqa: PLR2004

    assert second_sentence.start_index == 28  # noqa: PLR2004
    assert second_sentence.end_index == 56  # noqa: PLR2004


def test_sentence_with_leading_trailing_spaces(
    sentence_segmenter: SentenceSegmenter,
) -> None:
    note = Note(text="  This is a sentence with spaces. ")
    section = Section(
        text="  This is a sentence with spaces. ",
        start_index=0,
        end_index=34,
    )
    note.sections.append(section)

    # Process the note
    processed_note = sentence_segmenter.process(note)

    # Assert that the sentence has been stripped and indices adjusted
    assert len(processed_note.sections[0].sentences) == 1
    sentence = processed_note.sections[0].sentences[0]

    # Check the text of the sentence
    assert sentence.text == "This is a sentence with spaces."

    # Check the start and end indices
    assert sentence.start_index == 2  # noqa: PLR2004
    assert sentence.end_index == 33  # noqa: PLR2004


def test_multiple_sections_with_sentences(
    sentence_segmenter: SentenceSegmenter,
) -> None:
    note = Note(text="First section. Another sentence.\n\nSecond section.")
    section1 = Section(
        text="First section. Another sentence.",
        start_index=0,
        end_index=34,
    )
    section2 = Section(text="Second section.", start_index=36, end_index=51)
    note.sections.extend([section1, section2])

    # Process the note
    processed_note = sentence_segmenter.process(note)

    # Assert that sentences are correctly split across multiple sections
    assert len(processed_note.sections[0].sentences) == 2  # noqa: PLR2004
    assert len(processed_note.sections[1].sentences) == 1

    # Check the sentences in section 1
    assert processed_note.sections[0].sentences[0].text == "First section."
    assert processed_note.sections[0].sentences[1].text == "Another sentence."

    # Check the sentence in section 2
    assert processed_note.sections[1].sentences[0].text == "Second section."
