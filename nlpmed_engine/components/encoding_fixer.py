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

"""Encoding fixer module for NLPMed-Engine.

This module provides functionality to fix encoding issues in medical notes using the `ftfy` library.
The EncodingFixer class processes notes to correct common encoding problems, ensuring text is properly
readable and standardized.

Classes:
    EncodingFixer: Class for fixing encoding issues in notes.
"""

import ftfy

from nlpmed_engine.data_structures.note import Note


class EncodingFixer:
    """Class for fixing encoding issues in medical notes.

    This class uses the `ftfy` library to automatically correct encoding errors in the text
    of medical notes, making the text more readable and consistent.

    Methods:
        process: Fixes encoding issues in the text of a note.

    """

    def process(self, note: Note) -> Note:  # noqa: PLR6301
        """Fixes encoding issues in the text of the provided note.

        Args:
            note (Note): The note object whose text needs encoding fixes.

        Returns:
            Note: The processed note with encoding issues corrected.

        """
        note.text = ftfy.fix_text(note.text.strip())
        return note
