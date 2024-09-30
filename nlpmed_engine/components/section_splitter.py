"""Section splitter module for NLPMed-Engine.

This module provides functionality to split the text of medical notes into sections based on a specified delimiter.
The SectionSplitter class facilitates the division of note text into manageable sections for further processing.

Classes:
    SectionSplitter: Class for splitting note text into sections using a specified delimiter.
"""

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.utils.utils import get_effective_param


class SectionSplitter:
    """Class for splitting the text of medical notes into sections.

    This class uses a specified delimiter to divide the text of a note into sections,
    creating Section objects for each part. Sections are marked with their respective
    start and end indices relative to the original note text.

    Attributes:
        delimiter (str): The delimiter used to split the note text into sections.

    """

    def __init__(self, delimiter: str = "\n\n") -> None:
        """Initializes the SectionSplitter with a specified delimiter.

        Args:
            delimiter (str): The delimiter used to split the note text (default is "\n\n").

        """
        self.delimiter = delimiter

    def process(self, note: Note, delimiter: str | None = None) -> Note:
        """Processes a note by splitting its text into sections based on the specified delimiter.

        Args:
            note (Note): The note object containing text to be split into sections.
            delimiter (str | None): Optional delimiter to override the default delimiter.

        Returns:
            Note: The processed note with its text split into Section objects.

        """
        effective_delimiter = get_effective_param(self.delimiter, delimiter)

        if not note.text.strip():
            return note

        sections = note.text.split(effective_delimiter)
        start_index = 0

        for section_text in sections:
            end_index = start_index + len(section_text)
            section = Section(
                text=section_text,
                start_index=start_index,
                end_index=end_index,
            )
            note.sections.append(section)
            start_index = end_index + len(effective_delimiter)

        return note
