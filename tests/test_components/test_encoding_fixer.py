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

from nlpmed_engine.components.encoding_fixer import EncodingFixer
from nlpmed_engine.data_structures.note import Note


def test_encoding_fixer() -> None:
    note = Note(text="This note has encoding issues â€“")
    fixer = EncodingFixer()

    # Process the note
    processed_note = fixer.process(note)

    # Assert that encoding has been fixed
    assert "–" in processed_note.text  # noqa: RUF001
    assert "â€“" not in processed_note.text
