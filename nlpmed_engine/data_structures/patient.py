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
        patient.notes = [Note(note_data["text"]) for note_data in data["notes"]]
        return patient
