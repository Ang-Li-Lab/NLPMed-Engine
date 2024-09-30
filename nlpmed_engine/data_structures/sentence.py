from dataclasses import dataclass


@dataclass(slots=True)
class Sentence:
    text: str
    start_index: int
    end_index: int
    is_duplicate: bool = False
    is_important: bool = False
    is_expanded: bool = False
