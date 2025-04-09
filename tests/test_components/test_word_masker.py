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

from nlpmed_engine.components.word_masker import WordMasker
from nlpmed_engine.data_structures.note import Note


def test_single_word_masking() -> None:
    note = Note(text="This is a secret word.")
    masker = WordMasker(words_to_mask=["secret"], mask_char="*")

    # Process the note
    processed_note = masker.process(note)

    # Assert that 'secret' is masked by 6 '*' characters
    assert processed_note.text == "This is a ****** word."


def test_multiple_word_masking() -> None:
    note = Note(text="The password is secret and private.")
    masker = WordMasker(words_to_mask=["password", "secret", "private"], mask_char="#")

    # Process the note
    processed_note = masker.process(note)

    # Assert that all masked words are replaced with their respective masked lengths
    assert processed_note.text == "The ######## is ###### and #######."


def test_no_word_masking() -> None:
    note = Note(text="There are no sensitive words here.")
    masker = WordMasker(words_to_mask=["secret", "password"], mask_char="*")

    # Process the note
    processed_note = masker.process(note)

    # Assert that no words were masked since none matched
    assert processed_note.text == "There are no sensitive words here."


def test_word_masking_with_special_mask() -> None:
    note = Note(text="Hide this secret and confidential information.")
    masker = WordMasker(words_to_mask=["secret", "confidential"], mask_char="$")

    # Process the note
    processed_note = masker.process(note)

    # Assert that 'secret' and 'confidential' are masked with '$'
    assert processed_note.text == "Hide this $$$$$$ and $$$$$$$$$$$$ information."


def test_partial_word_no_masking() -> None:
    note = Note(text="This contains a word partiallysimilar but not identical.")
    masker = WordMasker(words_to_mask=["partial"], mask_char="*")

    # Process the note
    processed_note = masker.process(note)

    # Assert that 'partiallysimilar' is not masked as it is not an exact match
    assert processed_note.text == "This contains a word partiallysimilar but not identical."
