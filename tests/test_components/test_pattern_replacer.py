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

from nlpmed_engine.components.pattern_replacer import PatternReplacer
from nlpmed_engine.data_structures.note import Note


def test_single_pattern_replacement() -> None:
    note = Note(text="This is a test pattern.")
    replacer = PatternReplacer(pattern="test", target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that the pattern 'test' has been replaced by '***'
    assert processed_note.text == "This is a *** pattern."


def test_multiple_pattern_replacement() -> None:
    note = Note(text="This is a test pattern with some text.")
    replacer = PatternReplacer(pattern="test|text", target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that both 'test' and 'text' have been replaced by '***'
    assert processed_note.text == "This is a *** pattern with some ***."


def test_regex_pattern_replacement() -> None:
    note = Note(text="This is a test pattern with 1234 numbers.")
    replacer = PatternReplacer(pattern=r"\d+", target="###")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that the regex pattern for digits '\d+' has been replaced by '###'
    assert processed_note.text == "This is a test pattern with ### numbers."


def test_no_pattern_match() -> None:
    note = Note(text="This text has no matching pattern.")
    replacer = PatternReplacer(pattern="notfound", target="***")

    # Process the note
    processed_note = replacer.process(note)

    # Assert that no replacement occurs if the pattern does not match
    assert processed_note.text == "This text has no matching pattern."
