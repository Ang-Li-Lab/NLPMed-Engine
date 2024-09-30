"""Duplicate checking module for NLPMed-Engine.

This module provides functionality to detect and handle duplicate sentences within
medical notes using MinHash and Locality-Sensitive Hashing (LSH). The DuplicateChecker
class offers methods to process notes, identify duplicate sentences, and manage LSH states.

Classes:
    DuplicateChecker: Class for checking and handling duplicate sentences within notes.
"""

from typing import Any

from datasketch import MinHash
from datasketch import MinHashLSH

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.sentence import Sentence
from nlpmed_engine.utils.utils import get_effective_param


class DuplicateChecker:
    """Class for checking and managing duplicate sentences in medical notes.

    This class uses MinHash and MinHashLSH to identify similar or duplicate sentences
    within notes based on a defined similarity threshold. It allows for adding sentences
    to the LSH structure, querying for duplicates, and clearing/resetting the LSH state.

    Attributes:
        num_perm (int): Number of permutations used in MinHash.
        sim_threshold (float): Similarity threshold for considering sentences as duplicates.
        length_threshold (int): Minimum length of sentences to be considered for duplication checking.
        lsh (MinHashLSH): The Locality-Sensitive Hashing structure used to store and query MinHash values.

    """

    def __init__(
        self,
        num_perm: int = 256,
        sim_threshold: float = 0.9,
        length_threshold: int = 50,
    ) -> None:
        """Initializes the DuplicateChecker with specified parameters.

        Args:
            num_perm (int): Number of permutations for MinHash (default is 256).
            sim_threshold (float): Similarity threshold for duplicate detection (default is 0.9).
            length_threshold (int): Minimum sentence length to consider for duplication (default is 50).

        """
        self.num_perm = num_perm
        self.sim_threshold = sim_threshold
        self.length_threshold = length_threshold
        self.lsh = MinHashLSH(threshold=self.sim_threshold, num_perm=self.num_perm)

    def process(
        self,
        note: Note,
        length_threshold: int | None = None,
        **_: Any,  # noqa: ANN401
    ) -> Note:
        """Processes a note to check for duplicate sentences based on the defined thresholds.

        Args:
            note (Note): The note object containing sections and sentences to be processed.
            length_threshold (int | None): Optional length threshold to override the default.

        Returns:
            Note: The processed note with sentences marked as duplicates if applicable.

        """
        effective_length_threshold = get_effective_param(
            self.length_threshold,
            length_threshold,
            required=True,
        )

        for section in note.sections:
            duplicate_indices = []

            for idx, sentence in enumerate(section.sentences):
                if len(sentence.text) < effective_length_threshold:
                    continue

                if self.is_duplicate(sentence):
                    sentence.is_duplicate = True
                    duplicate_indices.append(idx)

                else:
                    self.add_sentence(sentence)

            section.duplicate_indices = duplicate_indices

        return note

    def add_sentence(self, sentence: Sentence) -> None:
        """Adds a sentence to the LSH structure for future duplicate detection.

        Args:
            sentence (Sentence): The sentence to be added to the LSH structure.

        """
        minhash = self.get_minhash(sentence)
        sentence_key = str(minhash.hashvalues)

        self.lsh.insert(sentence_key, minhash)

    def is_duplicate(self, sentence: Sentence) -> bool:
        """Checks if a given sentence is a duplicate by querying the LSH structure.

        Args:
            sentence (Sentence): The sentence to check for duplication.

        Returns:
            bool: True if the sentence is considered a duplicate, False otherwise.

        """
        minhash = self.get_minhash(sentence)
        return len(self.lsh.query(minhash)) > 0

    def get_minhash(self, sentence: Sentence) -> MinHash:
        """Generates a MinHash object from the words in a sentence.

        Args:
            sentence (Sentence): The sentence from which to generate the MinHash.

        Returns:
            MinHash: The MinHash object representing the sentence.

        """
        minhash = MinHash(num_perm=self.num_perm)

        for word in sentence.text.split():
            minhash.update(word.encode("utf8"))

        return minhash

    def clear_lsh(
        self,
        num_perm: int | None = None,
        sim_threshold: float | None = None,
        **_: Any,  # noqa: ANN401
    ) -> None:
        """Clears and reinitializes the LSH structure with optional new parameters.

        Args:
            num_perm (int | None): Optional new number of permutations for MinHash.
            sim_threshold (float | None): Optional new similarity threshold for LSH.

        """
        effective_num_perm = get_effective_param(self.num_perm, num_perm, required=True)
        effective_sim_threshold = get_effective_param(
            self.sim_threshold,
            sim_threshold,
            required=True,
        )

        self.lsh = MinHashLSH(
            threshold=effective_sim_threshold,
            num_perm=effective_num_perm,
        )
