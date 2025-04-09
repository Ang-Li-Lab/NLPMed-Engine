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

from nlpmed_engine.components.section_filter import SectionFilter
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.section import Section


def test_inclusion_pattern() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Chief complaint", start_index=0, end_index=10),
            Section(text="History of present illness", start_index=11, end_index=30),
            Section(text="Physical exam", start_index=31, end_index=40),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that sections starting from "Chief complaint" to the end are kept
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Chief complaint"
    assert processed_note.sections[1].text == "History of present illness"
    assert processed_note.sections[2].text == "Physical exam"


def test_exclusion_pattern() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Chief complaint", start_index=0, end_index=10),
            Section(text="History of present illness", start_index=11, end_index=30),
            Section(text="Physical exam", start_index=31, end_index=40),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=["Physical"],
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that sections are kept until "Physical exam" is reached
    assert len(processed_note.sections) == 2  # noqa: PLR2004
    assert processed_note.sections[0].text == "Chief complaint"
    assert processed_note.sections[1].text == "History of present illness"


def test_fallback_true_no_sections_detected() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Random section", start_index=0, end_index=10),
            Section(text="Another section", start_index=11, end_index=30),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
        fallback=True,
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that the fallback keeps all sections when no inclusion pattern is matched
    assert len(processed_note.sections) == 2  # noqa: PLR2004
    assert processed_note.sections[0].text == "Random section"
    assert processed_note.sections[1].text == "Another section"


def test_fallback_false_no_sections_detected() -> None:
    note = Note(
        text="Note text",
        sections=[
            Section(text="Random section", start_index=0, end_index=10),
            Section(text="Another section", start_index=11, end_index=30),
        ],
    )
    filter_component = SectionFilter(
        section_inc_list=["Chief complaint"],
        section_exc_list=[],
        fallback=False,
    )

    # Process the note
    processed_note = filter_component.process(note)

    # Assert that no sections are returned if no inclusion pattern is matched and fallback is False
    assert len(processed_note.sections) == 0
