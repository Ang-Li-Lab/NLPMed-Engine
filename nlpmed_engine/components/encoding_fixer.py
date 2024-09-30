"""Encoding fixer module for NLPMed-Engine.

This module provides functionality to fix encoding issues in medical notes using the `ftfy` library.
The EncodingFixer class processes notes to correct common encoding problems, ensuring text is properly
readable and standardized.

Classes:
    EncodingFixer: Class for fixing encoding issues in notes.
"""

import ftfy

from nlpmed_engine.data_structures.note import Note


class EncodingFixer:
    """Class for fixing encoding issues in medical notes.

    This class uses the `ftfy` library to automatically correct encoding errors in the text
    of medical notes, making the text more readable and consistent.

    Methods:
        process: Fixes encoding issues in the text of a note.

    """

    def process(self, note: Note) -> Note:
        """Fixes encoding issues in the text of the provided note.

        Args:
            note (Note): The note object whose text needs encoding fixes.

        Returns:
            Note: The processed note with encoding issues corrected.

        """
        note.text = ftfy.fix_text(note.text)
        return note
