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

import json
from dataclasses import dataclass
from dataclasses import field

from nlpmed_engine.data_structures.note import Note


@dataclass(slots=True)
class Patient:
    patient_id: str
    notes: list[Note] = field(default_factory=list)

    def add_note(self, note: Note) -> None:
        self.notes.append(note)

    @staticmethod
    def from_json(json_data: str) -> "Patient":
        data = json.loads(json_data)
        patient = Patient(patient_id=data["patient_id"])
        patient.notes = [
            Note(
                text=note_data["text"],
                note_id=note_data.get("note_id"),
            )
            for note_data in data["notes"]
        ]
        return patient
