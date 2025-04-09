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

from nlpmed_engine.components.sentence_filter import SentenceFilter
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.data_structures.sentence import Sentence


def test_single_keyword_match() -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=20)
    section.sentences.append(
        Sentence(text="The patient has DVT.", start_index=0, end_index=20),
    )
    section.sentences.append(
        Sentence(text="The patient has no issues.", start_index=21, end_index=40),
    )
    note.sections.append(section)

    filter_component = SentenceFilter(words_to_search=["DVT"])

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that only the sentence containing "DVT" is marked as important
    assert len(processed_note.sections[0].important_indices) == 1
    assert (
        processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].text
        == "The patient has DVT."
    )
    assert processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].is_important


def test_multiple_keyword_matches() -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=40)
    section.sentences.append(
        Sentence(text="The patient has DVT.", start_index=0, end_index=20),
    )
    section.sentences.append(
        Sentence(text="The patient has PE.", start_index=21, end_index=40),
    )
    note.sections.append(section)

    filter_component = SentenceFilter(words_to_search=["DVT", "PE"])

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that both sentences are marked as important
    assert len(processed_note.sections[0].important_indices) == 2  # noqa: PLR2004
    assert (
        processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].text
        == "The patient has DVT."
    )
    assert (
        processed_note.sections[0].sentences[processed_note.sections[0].important_indices[1]].text
        == "The patient has PE."
    )
    assert processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].is_important
    assert processed_note.sections[0].sentences[processed_note.sections[0].important_indices[1]].is_important


def test_no_keyword_match() -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=40)
    section.sentences.append(
        Sentence(text="The patient is healthy.", start_index=0, end_index=20),
    )
    section.sentences.append(
        Sentence(text="The patient has no issues.", start_index=21, end_index=40),
    )
    note.sections.append(section)

    filter_component = SentenceFilter(words_to_search=["DVT"])

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that no sentences are marked as important
    assert len(processed_note.sections[0].important_indices) == 0


def test_case_insensitive_match() -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=40)
    section.sentences.append(
        Sentence(text="The patient has dvt.", start_index=0, end_index=20),
    )
    note.sections.append(section)

    filter_component = SentenceFilter(words_to_search=["DVT"])

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that the case-insensitive match works
    assert len(processed_note.sections[0].important_indices) == 1
    assert (
        processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].text
        == "The patient has dvt."
    )
    assert processed_note.sections[0].sentences[processed_note.sections[0].important_indices[0]].is_important


def test_partial_match() -> None:
    note = Note(text="Note text")
    section = Section(text="Section text", start_index=0, end_index=40)
    section.sentences.append(
        Sentence(
            text="The patient has partially recovered.",
            start_index=0,
            end_index=20,
        ),
    )
    note.sections.append(section)

    filter_component = SentenceFilter(words_to_search=["part"])

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that partial matches are avoided
    assert len(processed_note.sections[0].important_indices) == 0
