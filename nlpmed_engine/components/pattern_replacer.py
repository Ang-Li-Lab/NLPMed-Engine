"""Pattern replacer module for NLPMed-Engine.

This module provides functionality to replace specified patterns within the text of medical notes.
The PatternReplacer class allows for defining patterns and target replacements to standardize
or clean up the text data.

Classes:
    PatternReplacer: Class for replacing patterns in notes with specified target strings.
"""

import re
from functools import lru_cache

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class PatternReplacer:
    """Class for replacing patterns in the text of medical notes.

    This class uses regular expressions to find and replace specified patterns within the text
    of notes. It allows for customizable pattern lists and target replacements to manage the
    standardization of note content.

    Attributes:
        pattern (str | None): Regex pattern to search for in the text.
        target (str | None): The replacement string for matched pattern.

    """

    def __init__(
        self,
        pattern: str | None = None,
        target: str | None = None,
    ) -> None:
        """Initializes the PatternReplacer with specified pattern and a target replacement.

        Args:
            pattern (str | None): Regex pattern to be replaced.
            target (str | None): The string to replace matched pattern with.

        """
        self.pattern = pattern
        self.target = target

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_regex(pattern: tuple[str]) -> re.Pattern:
        """Compiles the given regex pattern.

        Args:
            pattern (str): A regex pattern to compile.

        Returns:
            re.Pattern: A compiled regex pattern.

        """
        return re.compile(pattern)

    def process(
        self,
        note: Note,
        pattern: str | None = None,
        target: str | None = None,
    ) -> Note:
        """Processes a note by replacing matched pattern with the specified target string.

        Args:
            note (Note): The note object containing text to be modified.
            pattern (str | None): Optional pattern to override the default pattern.
            target (str | None): Optional target string to override the default replacement.

        Returns:
            Note: The processed note with pattern replaced by the target string.

        """
        effective_pattern = get_effective_param(self.pattern, pattern, required=True)
        effective_target = get_effective_param(self.target, target, required=True)

        regex = self._compile_regex(effective_pattern)

        note.text = regex.sub(effective_target, note.text)
        return note
