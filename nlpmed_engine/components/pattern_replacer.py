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
        patterns (list[str] | None): List of regex patterns to search for in the text.
        target (str | None): The replacement string for matched patterns.

    """

    def __init__(
        self,
        patterns: list[str] | None = None,
        target: str | None = None,
    ) -> None:
        """Initializes the PatternReplacer with specified patterns and a target replacement.

        Args:
            patterns (list[str] | None): List of regex patterns to be replaced.
            target (str | None): The string to replace matched patterns with.

        """
        self.patterns = patterns
        self.target = target

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_regex(patterns: tuple[str]) -> re.Pattern:
        """Compiles a combined regex pattern from a list of patterns.

        Args:
            patterns (tuple[str]): A tuple of regex patterns to compile.

        Returns:
            re.Pattern: A compiled regex pattern that matches any of the specified patterns.

        """
        combined_pattern = "|".join(f"({pattern})" for pattern in patterns)
        return re.compile(combined_pattern)

    def process(
        self,
        note: Note,
        patterns: list[str] | None = None,
        target: str | None = None,
    ) -> Note:
        """Processes a note by replacing matched patterns with the specified target string.

        Args:
            note (Note): The note object containing text to be modified.
            patterns (list[str] | None): Optional list of patterns to override the default patterns.
            target (str | None): Optional target string to override the default replacement.

        Returns:
            Note: The processed note with patterns replaced by the target string.

        """
        effective_patterns = get_effective_param(self.patterns, patterns, required=True)
        effective_target = get_effective_param(self.target, target, required=True)

        regex = self._compile_regex(tuple(effective_patterns))

        note.text = regex.sub(effective_target, note.text)
        return note
