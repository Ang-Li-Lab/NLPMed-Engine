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

"""Word masker module for NLPMed-Engine.

This module provides functionality to mask specified words within the text of medical notes.
The WordMasker class allows for defining words to be masked and the character used for masking,
ensuring sensitive or unwanted terms are obscured in the text.

Classes:
    WordMasker: Class for masking specified words in the text of medical notes.
"""

import re
from functools import lru_cache

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.utils.utils import get_effective_param


class WordMasker:
    """Class for masking specified words within the text of medical notes.

    This class uses regular expressions to identify and replace specified words with a masking character,
    enhancing privacy or readability by obscuring sensitive or unwanted terms.

    Attributes:
        words_to_mask (list[str]): List of words to be masked in the text.
        mask_char (str): The character used to replace each character of the masked words.

    """

    def __init__(
        self,
        words_to_mask: list[str] | None = None,
        mask_char: str = "*",
    ) -> None:
        """Initializes the WordMasker with specified words to mask and a masking character.

        Args:
            words_to_mask (list[str] | None): List of words to be masked. Defaults to an empty list if not provided.
            mask_char (str): The character used to mask the words (default is "*").

        """
        self.words_to_mask = words_to_mask if words_to_mask is not None else []
        self.mask_char = mask_char

    @staticmethod
    @lru_cache(maxsize=16)
    def _compile_regex(words: tuple[str]) -> re.Pattern:
        """Compiles a regular expression pattern to match the specified words for masking.

        Args:
            words (tuple[str]): A tuple of words to compile into a regex pattern.

        Returns:
            re.Pattern: A compiled regex pattern that matches the specified words.

        """
        escaped_words = (re.escape(w) for w in words)
        joined_words = "|".join(escaped_words)
        pattern = rf"(?<![a-zA-Z0-9])({joined_words})(?![a-zA-Z0-9])"
        return re.compile(pattern, flags=re.IGNORECASE)

    def process(
        self,
        note: Note,
        words_to_mask: list[str] | None = None,
        mask_char: str | None = None,
    ) -> Note:
        """Processes a note by masking specified words in its text.

        Args:
            note (Note): The note object containing text to be masked.
            words_to_mask (list[str] | None): Optional list of words to override the default words to mask.
            mask_char (str | None): Optional masking character to override the default.

        Returns:
            Note: The processed note with specified words masked.

        """
        effective_words_to_mask = get_effective_param(
            self.words_to_mask,
            words_to_mask,
            required=True,
        )
        effective_mask_char = get_effective_param(
            self.mask_char,
            mask_char,
            required=True,
        )

        regex = self._compile_regex(tuple(effective_words_to_mask))

        note.text = regex.sub(
            lambda m: effective_mask_char * len(m.group()),
            note.text,
        )
        return note
