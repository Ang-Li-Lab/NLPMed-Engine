from dataclasses import dataclass
from dataclasses import field

from nlpmed_engine.data_structures.sentence import Sentence


@dataclass(slots=True)
class Section:
    text: str
    start_index: int
    end_index: int
    sentences: list[Sentence] = field(default_factory=list)
    important_indices: list[int] = field(default_factory=list)
    duplicate_indices: list[int] = field(default_factory=list)
    expanded_indices: list[int] = field(default_factory=list)
    is_important: bool = False
