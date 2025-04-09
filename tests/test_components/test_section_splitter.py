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

from nlpmed_engine.components.section_splitter import SectionSplitter
from nlpmed_engine.data_structures.note import Note


def test_single_section() -> None:
    note = Note(text="This is a single section with no delimiter.")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that there is only one section
    assert len(processed_note.sections) == 1
    assert processed_note.sections[0].text == "This is a single section with no delimiter."
    assert processed_note.sections[0].start_index == 0
    assert processed_note.sections[0].end_index == len(note.text)


def test_multiple_sections() -> None:
    note = Note(text="Section 1\n\nSection 2\n\nSection 3")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that the note has been split into 3 sections
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Section 1"
    assert processed_note.sections[1].text == "Section 2"
    assert processed_note.sections[2].text == "Section 3"

    # Check start and end indices for each section
    assert processed_note.sections[0].start_index == 0
    assert processed_note.sections[0].end_index == len("Section 1")

    assert processed_note.sections[1].start_index == len("Section 1\n\n")
    assert processed_note.sections[1].end_index == len("Section 1\n\nSection 2")

    assert processed_note.sections[2].start_index == len("Section 1\n\nSection 2\n\n")
    assert processed_note.sections[2].end_index == len(
        "Section 1\n\nSection 2\n\nSection 3",
    )


def test_custom_delimiter() -> None:
    note = Note(text="Part 1##Part 2##Part 3")
    splitter = SectionSplitter(delimiter="##")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that the note has been split into 3 parts using the custom delimiter
    assert len(processed_note.sections) == 3  # noqa: PLR2004
    assert processed_note.sections[0].text == "Part 1"
    assert processed_note.sections[1].text == "Part 2"
    assert processed_note.sections[2].text == "Part 3"


def test_no_sections_with_empty_note() -> None:
    note = Note(text="")
    splitter = SectionSplitter(delimiter="\n\n")

    # Process the note
    processed_note = splitter.process(note)

    # Assert that no sections are created for an empty note
    assert len(processed_note.sections) == 0
