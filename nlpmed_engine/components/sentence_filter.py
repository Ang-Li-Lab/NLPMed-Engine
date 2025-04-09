# SPDX-FileCopyrightText: Copyright (C) 2025 Omid Jafari <omidjafari.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Sentence filter module for NLPMed-Engine.

This module provides functionality to filter and flag important sentences within medical notes
based on specified keywords. The SentenceFilter class uses regular expressions to identify sentences
that contain the target words, marking them as important.

Classes:
    SentenceFilter: Class for filtering sentences in notes based on keyword presence.
"""

import re
from functools import lru_cache

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class SentenceFilter:
    """Class for filtering and marking important sentences within sections of medical notes.

    This class identifies sentences that contain specified keywords, flagging them as important
    for further processing. The filtering is achieved using compiled regular expressions
    to match target words in the sentences.

    Attributes:
        words_to_search (list[str] | None): List of keywords to search for in sentences.

    """

    def __init__(self, words_to_search: list[str] | None = None) -> None:
        """Initializes the SentenceFilter with specified keywords.

        Args:
            words_to_search (list[str] | None): List of keywords to search for in sentences.
            If not provided, it can be set during processing.

        """
        self.words_to_search = words_to_search

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_regex(words: tuple[str]) -> re.Pattern:
        """Compiles a regular expression pattern to search for specified keywords in sentences.

        Args:
            words (tuple[str]): A tuple of keywords to compile into a regex pattern.

        Returns:
            re.Pattern: A compiled regex pattern that matches the specified keywords.

        """
        escaped_words = (re.escape(w) for w in words)
        joined_words = "|".join(escaped_words)
        pattern = rf"(?<![a-zA-Z0-9])(?:{joined_words})(?![a-zA-Z0-9])"
        return re.compile(pattern, flags=re.IGNORECASE)

    def process(self, note: Note, words_to_search: list[str] | None = None) -> Note:
        """Processes a note by filtering its sentences based on the specified keywords.

        Args:
            note (Note): The note object containing sections and sentences to be filtered.
            words_to_search (list[str] | None): Optional list of keywords to override the default list.

        Returns:
            Note: The processed note with sentences marked as important if they contain the keywords.

        """
        effective_words_to_search = get_effective_param(
            self.words_to_search,
            words_to_search,
            required=True,
        )

        regex = self._compile_regex(tuple(effective_words_to_search))

        for section in note.sections:
            important_indices = []

            for idx, sentence in enumerate(section.sentences):
                if sentence.is_duplicate:
                    continue

                if regex.search(sentence.text):
                    sentence.is_important = True
                    important_indices.append(idx)

            section.important_indices = important_indices

        return note
