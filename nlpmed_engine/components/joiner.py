"""Joiner module for NLPMed-Engine.

This module provides functionality to join important sentences and sections of medical
notes into a cohesive preprocessed text. The Joiner class allows customization of sentence
and section delimiters to structure the joined text appropriately.

Classes:
    Joiner: Class for joining sentences and sections within a note.
"""

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class Joiner:
    """Class for joining important sentences and sections within medical notes.

    This class processes a note by joining important sentences within each section
    into a single string, then combines these joined sections into a final preprocessed
    text using specified delimiters.

    Attributes:
        sentence_delimiter (str): Delimiter used to join sentences within a section.
        section_delimiter (str): Delimiter used to join sections within the note.

    """

    def __init__(
        self,
        sentence_delimiter: str = "\n",
        section_delimiter: str = "\n\n",
    ) -> None:
        """Initializes the Joiner with specified delimiters for sentences and sections.

        Args:
            sentence_delimiter (str): Delimiter used to join sentences (default is "\n").
            section_delimiter (str): Delimiter used to join sections (default is "\n\n").

        """
        self.sentence_delimiter = sentence_delimiter
        self.section_delimiter = section_delimiter

    def process(
        self,
        note: Note,
        sentence_delimiter: str | None = None,
        section_delimiter: str | None = None,
    ) -> Note:
        """Processes a note to join important sentences and sections into preprocessed text.

        Args:
            note (Note): The note object containing sections and sentences to be joined.
            sentence_delimiter (str | None): Optional custom delimiter for joining sentences.
            section_delimiter (str | None): Optional custom delimiter for joining sections.

        Returns:
            Note: The processed note with the preprocessed text formed by joining sentences and sections.

        """
        effective_sentence_delimiter = get_effective_param(
            self.sentence_delimiter,
            sentence_delimiter,
            required=True,
        )
        effective_section_delimiter = get_effective_param(
            self.section_delimiter,
            section_delimiter,
            required=True,
        )
        preprocessed_sections = []

        for section in note.sections:
            if not section.is_important:
                continue

            sentences_to_join = []

            if section.important_indices:
                sentences_to_join = [section.sentences[idx].text for idx in section.important_indices]

            if sentences_to_join:
                joined_sentences = effective_sentence_delimiter.join(sentences_to_join)
                preprocessed_sections.append(joined_sentences)

        note.preprocessed_text = effective_section_delimiter.join(preprocessed_sections)

        return note
