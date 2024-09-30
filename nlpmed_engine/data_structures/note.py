from dataclasses import dataclass
from dataclasses import field

from nlpmed_engine.data_structures.section import Section


@dataclass(slots=True)
class Note:
    text: str
    sections: list[Section] = field(default_factory=list)
    preprocessed_text: str | None = None
    predicted_label: str | None = None
    predicted_score: float | None = None
