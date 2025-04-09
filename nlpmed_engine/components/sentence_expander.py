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

"""Sentence expander module for NLPMed-Engine.

This module provides functionality to expand short sentences within medical notes by combining
them with neighboring sentences until a specified length threshold is met. The SentenceExpander
class helps ensure that important short sentences are contextually enriched with adjacent content.

Classes:
    SentenceExpander: Class for expanding short sentences in notes by merging them with surrounding sentences.
"""

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.sentence import Sentence
from nlpmed_engine.utils.utils import get_effective_param


class SentenceExpander:
    """Class for expanding short sentences within sections of medical notes.

    This class processes sentences within sections of a note, expanding short sentences
    by merging them with adjacent sentences until a specified length threshold is met.
    This approach enhances the contextual richness of important sentences.

    Attributes:
        length_threshold (int): The minimum length a sentence should have before it is considered for expansion.

    """

    def __init__(self, length_threshold: int = 50) -> None:
        """Initializes the SentenceExpander with a specified length threshold.

        Args:
            length_threshold (int): The length threshold of short sentence (default is 50).

        """
        self.length_threshold = length_threshold

    def process(self, note: Note, length_threshold: int | None = None) -> Note:
        """Processes a note by expanding short sentences within its sections.

        Args:
            note (Note): The note object containing sections and sentences to be expanded.
            length_threshold (int | None): Optional length threshold to override the default.

        Returns:
            Note: The processed note with expanded sentences in important sections.

        """
        effective_length_threshold = get_effective_param(
            self.length_threshold,
            length_threshold,
            required=True,
        )

        for section in note.sections:
            expanded_indices = self.expand_section_sentences(
                section.sentences,
                section.important_indices,
                effective_length_threshold,
            )
            section.expanded_indices = expanded_indices

        return note

    def expand_section_sentences(  # noqa: PLR6301
        self,
        sentences: list[Sentence],
        important_indices: list[int],
        length_threshold: int,
    ) -> list[int]:
        """Expands short sentences in a list by merging them with adjacent sentences.

        Args:
            sentences (list[Sentence]): A list of sentences within a section to be expanded.
            length_threshold (int): The minimum length a sentence should have before it is considered sufficiently long.

        Returns:
            list[Sentence]: A list of expanded sentences that meet the length threshold.

        """
        expanded_indices = set()

        for idx in important_indices:
            if len(sentences[idx].text) >= length_threshold:  # Note a short sentence
                if idx not in expanded_indices:
                    expanded_indices.add(idx)

            else:  # Short sentence
                range_start, range_end = idx, idx + 1
                combined_len = len(sentences[idx].text)

                # Keep adding sentences from both sides until reaching desired length
                while combined_len < length_threshold:
                    # Previous sentence
                    if range_start > 0:
                        range_start -= 1
                        combined_len += len(sentences[range_start].text)

                    # Still short, add next sentence
                    if combined_len < length_threshold and range_end < len(sentences):
                        combined_len += len(sentences[range_end].text)
                        range_end += 1

                    # Not possible to expand anymore
                    if range_start == 0 and range_end == len(sentences):
                        break

                expanded_indices.update(range(range_start, range_end))

        return sorted(expanded_indices)
