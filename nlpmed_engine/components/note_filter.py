"""Note filtering module for NLPMed-Engine.

This module provides functionality to filter medical notes based on specified keywords.
The NoteFilter class checks whether a note contains any of the specified words and
returns the note if it matches the criteria.

Classes:
    NoteFilter: Class for filtering notes based on keyword presence.
"""

import re
from functools import lru_cache

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class NoteFilter:
    """Class for filtering medical notes based on the presence of specified keywords.

    This class uses regular expressions to identify whether a note contains any of the
    specified words. If a match is found, the note is returned; otherwise, it is filtered out.

    Attributes:
        words_to_search (list[str] | None): List of keywords to search for in notes.

    """

    def __init__(self, words_to_search: list[str] | None = None) -> None:
        """Initializes the NoteFilter with specified keywords.

        Args:
            words_to_search (list[str] | None): List of keywords to search for in notes.
            If not provided, it can be set during processing.

        """
        self.words_to_search = words_to_search

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_regex(words: tuple[str]) -> re.Pattern:
        """Compiles a regular expression pattern to search for specified keywords in a note.

        Args:
            words (tuple[str]): A tuple of keywords to compile into a regex pattern.

        Returns:
            re.Pattern: A compiled regex pattern that matches the specified keywords.

        """
        escaped_words = (re.escape(w) for w in words)
        joined_words = "|".join(escaped_words)
        pattern = rf"(?<![a-zA-Z0-9])(?:{joined_words})(?![a-zA-Z0-9])"
        return re.compile(pattern, flags=re.IGNORECASE)

    def process(
        self,
        note: Note,
        words_to_search: list[str] | None = None,
    ) -> Note | None:
        """Processes a note to check if it contains specified keywords.

        Args:
            note (Note): The note object to be filtered.
            words_to_search (list[str] | None): Optional list of keywords to search for,
            overriding the default list set during initialization.

        Returns:
            Note | None: The note if it contains the keywords, otherwise None.

        """
        effective_words_to_search = get_effective_param(
            self.words_to_search,
            words_to_search,
            required=True,
        )

        regex = self._compile_regex(tuple(effective_words_to_search))

        if regex.search(note.text):
            return note

        return None
