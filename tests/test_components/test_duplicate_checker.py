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

import pytest

from nlpmed_engine.components.duplicate_checker import DuplicateChecker
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.data_structures.sentence import Sentence


@pytest.fixture
def duplicate_checker() -> DuplicateChecker:
    return DuplicateChecker(num_perm=256, sim_threshold=0.9, length_threshold=20)


def test_no_duplicates(duplicate_checker: DuplicateChecker) -> None:
    note = Note(text="Test note")
    section = Section(text="Section text", start_index=0, end_index=100)
    sentence1 = Sentence(text="This is a unique sentence.", start_index=0, end_index=25)
    sentence2 = Sentence(text="Another unique sentence.", start_index=26, end_index=50)
    section.sentences.extend([sentence1, sentence2])
    note.sections.append(section)

    # Process the note (no duplicates in LSH)
    processed_note = duplicate_checker.process(note)

    # Assert that no sentences are marked as duplicates
    assert not processed_note.sections[0].sentences[0].is_duplicate
    assert not processed_note.sections[0].sentences[1].is_duplicate
    assert len(processed_note.sections[0].duplicate_indices) == 0


def test_single_duplicate(duplicate_checker: DuplicateChecker) -> None:
    note = Note(text="Test note")
    section = Section(text="Section text", start_index=0, end_index=100)
    sentence1 = Sentence(
        text="This is a duplicate sentence.",
        start_index=0,
        end_index=25,
    )
    sentence2 = Sentence(
        text="This is a duplicate sentence.",
        start_index=26,
        end_index=51,
    )
    section.sentences.extend([sentence1, sentence2])
    note.sections.append(section)

    # Process the note (second sentence is a duplicate)
    processed_note = duplicate_checker.process(note)

    # Assert that the second sentence is marked as duplicate
    assert not processed_note.sections[0].sentences[0].is_duplicate
    assert processed_note.sections[0].sentences[1].is_duplicate
    assert len(processed_note.sections[0].duplicate_indices) == 1
    assert processed_note.sections[0].duplicate_indices[0] == 1  # Check index


def test_multiple_duplicates(duplicate_checker: DuplicateChecker) -> None:
    note = Note(text="Test note")
    section = Section(text="Section text", start_index=0, end_index=100)
    sentence1 = Sentence(text="First duplicate sentence.", start_index=0, end_index=25)
    sentence2 = Sentence(
        text="Second duplicate sentence.",
        start_index=26,
        end_index=51,
    )
    sentence3 = Sentence(text="First duplicate sentence.", start_index=52, end_index=77)
    sentence4 = Sentence(text="Another unique sentence.", start_index=78, end_index=100)
    section.sentences.extend([sentence1, sentence2, sentence3, sentence4])
    note.sections.append(section)

    # Process the note (third sentence is a duplicate of the first)
    processed_note = duplicate_checker.process(note)

    # Assert that duplicates are marked correctly
    assert not processed_note.sections[0].sentences[0].is_duplicate
    assert not processed_note.sections[0].sentences[1].is_duplicate
    assert processed_note.sections[0].sentences[2].is_duplicate
    assert not processed_note.sections[0].sentences[3].is_duplicate

    assert len(processed_note.sections[0].duplicate_indices) == 1
    assert processed_note.sections[0].duplicate_indices[0] == 2  # Check index of duplicate # noqa: PLR2004


def test_short_sentences_ignored(duplicate_checker: DuplicateChecker) -> None:
    note = Note(text="Test note")
    section = Section(text="Section text", start_index=0, end_index=100)
    sentence1 = Sentence(text="Short.", start_index=0, end_index=6)
    sentence2 = Sentence(
        text="This is a sufficiently long sentence.",
        start_index=7,
        end_index=43,
    )
    section.sentences.extend([sentence1, sentence2])
    note.sections.append(section)

    # Process the note
    processed_note = duplicate_checker.process(note)

    # Assert that short sentence is not marked as duplicate and not checked
    assert not processed_note.sections[0].sentences[0].is_duplicate
    assert not processed_note.sections[0].sentences[1].is_duplicate
    assert len(processed_note.sections[0].duplicate_indices) == 0


def test_duplicates_across_multiple_notes(duplicate_checker: DuplicateChecker) -> None:
    note1 = Note(text="First note")
    section1 = Section(text="Section 1", start_index=0, end_index=50)
    sentence1 = Sentence(
        text="This sentence is in note 1.",
        start_index=0,
        end_index=28,
    )
    section1.sentences.append(sentence1)
    note1.sections.append(section1)

    note2 = Note(text="Second note")
    section2 = Section(text="Section 2", start_index=0, end_index=50)
    sentence2 = Sentence(
        text="This sentence is in note 1.",
        start_index=0,
        end_index=28,
    )  # Duplicate sentence
    section2.sentences.append(sentence2)
    note2.sections.append(section2)

    # Process the first note
    processed_note1 = duplicate_checker.process(note1)

    # Process the second note (sentence should be marked as a duplicate)
    processed_note2 = duplicate_checker.process(note2)

    # Assert no duplicates in the first note
    assert not processed_note1.sections[0].sentences[0].is_duplicate

    # Assert that the duplicate is detected in the second note
    assert processed_note2.sections[0].sentences[0].is_duplicate
    assert len(processed_note2.sections[0].duplicate_indices) == 1
    assert processed_note2.sections[0].duplicate_indices[0] == 0  # Check index
