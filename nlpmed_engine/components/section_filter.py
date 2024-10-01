"""Section filtering module for NLPMed-Engine.

This module provides functionality to filter sections of medical notes based on specified
inclusion and exclusion keywords. The SectionFilter class uses regular expressions to
identify and retain important sections while optionally allowing fallback behavior.

Classes:
    SectionFilter: Class for filtering sections within notes based on inclusion and exclusion rules.
"""

import re
from functools import lru_cache

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class SectionFilter:
    """Class for filtering sections of medical notes based on inclusion and exclusion criteria.

    This class allows for defining inclusion and exclusion lists of keywords to filter sections
    of a note. Sections that match inclusion criteria are retained, while sections matching
    exclusion criteria are filtered out, unless fallback behavior is enabled.

    Attributes:
        section_inc_list (list[str] | None): List of keywords for including sections.
        section_exc_list (list[str] | None): List of keywords for excluding sections.
        fallback (bool): Whether to enable fallback behavior if no sections match the criteria.

    """

    def __init__(
        self,
        section_inc_list: list[str] | None = None,
        section_exc_list: list[str] | None = None,
        *,
        fallback: bool = False,
    ) -> None:
        """Initializes the SectionFilter with specified inclusion and exclusion lists.

        Args:
            section_inc_list (list[str] | None): List of keywords for including sections.
            section_exc_list (list[str] | None): List of keywords for excluding sections.
            fallback (bool): Whether to enable fallback behavior if no sections match (default is False).

        """
        self.section_inc_list = section_inc_list
        self.section_exc_list = section_exc_list
        self.fallback = fallback

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_inc_regex(words: tuple[str]) -> re.Pattern:
        """Compiles a regex pattern to match sections based on inclusion keywords.

        Args:
            words (tuple[str]): A tuple of inclusion keywords.

        Returns:
            re.Pattern: A compiled regex pattern to match inclusion keywords in section text.

        """
        escaped_keywords = (re.escape(k) for k in words)
        joined_keywords = "|".join(escaped_keywords)
        inc_pattern = rf"^(?<![a-zA-Z0-9])(?:{joined_keywords})(?![a-zA-Z0-9])"
        return re.compile(inc_pattern, flags=re.IGNORECASE)

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_exc_regex(words: tuple[str]) -> re.Pattern:
        """Compiles a regex pattern to match sections based on exclusion keywords.

        Args:
            words (tuple[str]): A tuple of exclusion keywords.

        Returns:
            re.Pattern: A compiled regex pattern to match exclusion keywords in section text.

        """
        escaped_keywords = (re.escape(k) for k in words)
        joined_keywords = "|".join(escaped_keywords)
        exc_pattern = rf"^(?<![a-zA-Z0-9])(?:{joined_keywords})"
        return re.compile(exc_pattern, flags=re.IGNORECASE)

    def process(
        self,
        note: Note,
        section_inc_list: list[str] | None = None,
        section_exc_list: list[str] | None = None,
        fallback: bool | None = None,
    ) -> Note:
        """Processes a note by filtering its sections based on inclusion and exclusion keywords.

        Args:
            note (Note): The note object containing sections to be filtered.
            section_inc_list (list[str] | None): Optional list of inclusion keywords to override the default list.
            section_exc_list (list[str] | None): Optional list of exclusion keywords to override the default list.
            fallback (bool | None): Optional fallback behavior to override the default setting.

        Returns:
            Note: The processed note with filtered sections based on the defined rules.

        """
        effective_section_inc_list = get_effective_param(
            self.section_inc_list,
            section_inc_list,
            required=False,
        )
        effective_section_exc_list = get_effective_param(
            self.section_exc_list,
            section_exc_list,
            required=False,
        )
        effective_fallback = get_effective_param(self.fallback, fallback, required=True)

        inc_regex = self._compile_inc_regex(tuple(effective_section_inc_list)) if effective_section_inc_list else None
        exc_regex = self._compile_exc_regex(tuple(effective_section_exc_list)) if effective_section_exc_list else None

        filtered_sections = []
        in_inclusion_block = False

        for section in note.sections:
            if inc_regex and inc_regex.match(section.text):
                in_inclusion_block = True

            elif exc_regex and exc_regex.match(section.text) and in_inclusion_block:
                in_inclusion_block = False

            if in_inclusion_block and (exc_regex is None or not exc_regex.match(section.text)):
                section.is_important = True
                filtered_sections.append(section)

        if not filtered_sections and effective_fallback:
            # Mark all sections as important
            for section in note.sections:
                section.is_important = True

            return note

        note.sections = filtered_sections
        return note
