"""Sentence segmenter module for NLPMed-Engine.

This module provides functionality to segment text into sentences using spaCy. The
SentenceSegmenter class processes text from medical notes, splitting sections into
sentences with accurate start and end indices, ensuring precise sentence-level segmentation.

Classes:
    SentenceSegmenter: Class for segmenting text into sentences using a spaCy NLP model.
"""

from typing import Any

import spacy
from spacy.attrs import NORM
from spacy.attrs import ORTH
from spacy.tokens import Doc

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.data_structures.sentence import Sentence

spacy.prefer_gpu()


class SentenceSegmenter:
    """Class for segmenting text from medical notes into sentences using a spaCy NLP model.

    This class utilizes a spaCy pipeline to split sections of notes into sentences, capturing
    their start and end positions within the text. It supports processing individual notes
    and batches of patients with configurable NLP models and batch sizes.

    Attributes:
        nlp (spacy.Language): The spaCy NLP pipeline used for sentence segmentation.
        batch_size (int): The batch size used when processing text in parallel.

    """

    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        batch_size: int = 10,
    ) -> None:
        """Initializes the SentenceSegmenter with a specified NLP model and batch size.

        Args:
            model_name (str):
                The name of the spaCy model to use for sentence segmentation
                (default is "en_core_sci_lg").
            batch_size (int): The batch size for processing sections in parallel (default is 10).

        """
        self.nlp = spacy.load(
            model_name,
            exclude=["tagger", "attribute_ruler", "lemmatizer", "ner"],
        )
        self.nlp.tokenizer.add_special_case(  # type: ignore[union-attr]
            "Dept.",
            [{ORTH: "Dept.", NORM: "department"}],
        )
        self.batch_size = batch_size

    def process(self, note: Note, **_: Any) -> Note:  # noqa: ANN401
        """Processes a single note, segmenting its sections into sentences.

        Args:
            note (Note): The note object containing sections to be segmented into sentences.

        Returns:
            Note: The processed note with sections segmented into sentences.

        """
        sections_text_list = [sec.text for sec in note.sections]
        segmented_docs = list(
            self.nlp.pipe(sections_text_list, batch_size=self.batch_size),
        )

        for section, doc in zip(note.sections, segmented_docs, strict=False):
            section.sentences = self._create_sentences_from_doc(
                doc,
                section.start_index,
            )

        return note

    def process_batch_patients(self, patients: list[Patient]) -> list[Patient]:
        """Processes a batch of patients, segmenting the sections of their notes into sentences.

        Args:
            patients (list[Patient]): A list of patient objects containing notes and sections to be segmented.

        Returns:
            list[Patient]: The list of patients with their notes' sections segmented into sentences.

        """
        sections_text_list = [
            section.text for patient in patients for note in patient.notes for section in note.sections
        ]
        segmented_docs = list(
            self.nlp.pipe(sections_text_list, batch_size=self.batch_size),
        )

        idx = 0
        for patient in patients:
            for note in patient.notes:
                for section in note.sections:
                    section.sentences = self._create_sentences_from_doc(
                        segmented_docs[idx],
                        section.start_index,
                    )
                    idx += 1

        return patients

    def _create_sentences_from_doc(
        self,
        doc: Doc,
        section_start_index: int,
    ) -> list[Sentence]:
        """Creates Sentence objects from a spaCy Doc, adjusting indices relative to the section's start.

        Args:
            doc (Doc): A spaCy Doc object containing segmented sentences.
            section_start_index (int): The start index of the section relative to the original text.

        Returns:
            list[Sentence]: A list of Sentence objects with adjusted start and end indices.

        """
        sentences = []

        for sent in doc.sents:
            stripped_text = sent.text.strip()

            if sent.text != stripped_text:  # Sentences needs to be stripped
                leading_spaces = len(sent.text) - len(sent.text.lstrip())
                trailing_spaces = len(sent.text) - len(sent.text.rstrip())
                start_index = section_start_index + sent.start_char + leading_spaces
                end_index = section_start_index + sent.end_char - trailing_spaces

            else:  # Sentence doesn't have any leading or trailing spaces
                start_index = section_start_index + sent.start_char
                end_index = section_start_index + sent.end_char

            sentences.append(
                Sentence(
                    start_index=start_index,
                    end_index=end_index,
                    text=stripped_text,
                ),
            )

        return sentences
