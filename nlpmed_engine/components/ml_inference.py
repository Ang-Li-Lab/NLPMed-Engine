"""Machine Learning Inference module for NLPMed-Engine.

This module provides functionality for performing machine learning-based inference
on medical notes using a pre-trained text classification model. The MLInference class
uses the Hugging Face Transformers library to predict labels and scores for text data.

Classes:
    MLInference: Class for performing ML inference on notes and patients.
"""

from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.utils.utils import get_effective_param


class MLInference:
    """Class for performing machine learning inference on medical notes.

    This class initializes a text classification pipeline using a specified model and tokenizer.
    It provides methods for predicting labels and scores for text data within notes and patients,
    allowing for batch processing and customization of input parameters.

    Attributes:
        use_preped_text (bool): Whether to use preprocessed text for inference.
        pipe (pipeline): The Hugging Face Transformers pipeline for text classification.

    """

    def __init__(
        self,
        device: str = "cpu",
        ml_model_path: Path | str = "",
        ml_tokenizer_path: Path | str = "",
        max_length: int = 512,
        *,
        use_preped_text: bool = True,
    ) -> None:
        """Initializes the MLInference class with the specified model, tokenizer, and settings.

        Args:
            device (str): The device to use for inference ('cpu' or 'cuda').
            ml_model_path (Path | str): Path to the pre-trained model.
            ml_tokenizer_path (Path | str): Path to the tokenizer for the model.
            max_length (int): Maximum length for text sequences during inference.
            use_preped_text (bool): Whether to use preprocessed text for inference (default is True).

        """
        model = AutoModelForSequenceClassification.from_pretrained(ml_model_path)
        tokenizer = AutoTokenizer.from_pretrained(ml_tokenizer_path)
        self.use_preped_text = use_preped_text
        self.pipe = pipeline(
            task="text-classification",
            device=device,
            padding="max_length",
            truncation=True,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def process(
        self,
        note: Note,
        *,
        use_preped_text: bool | None = None,
        **_: Any,  # noqa: ANN401
    ) -> Note:
        """Performs inference on a single note, predicting a label and score.

        Args:
            note (Note): The note object containing text to be classified.
            use_preped_text (bool | None): Optional override for using preprocessed text.

        Returns:
            Note: The note object with predicted label and score updated.

        """
        effective_use_preped_text = get_effective_param(self.use_preped_text, use_preped_text, required=True)

        text_to_infer = note.preprocessed_text if effective_use_preped_text else note.text
        if text_to_infer:
            results = self.pipe(text_to_infer, top_k=1)
            note.predicted_label = results[0]["label"]
            note.predicted_score = round(results[0]["score"], 2)

        return note

    def process_batch_patients(self, patients: list[Patient], *, use_preped_text: bool | None = None) -> list[Patient]:
        """Performs batch inference on a list of patients, predicting labels and scores for their notes.

        Args:
            patients (list[Patient]): A list of patient objects containing notes to be classified.
            use_preped_text (bool | None): Optional override for using preprocessed text.

        Returns:
            list[Patient]: The list of patients with their notes updated with predicted labels and scores.

        """
        effective_use_preped_text = get_effective_param(self.use_preped_text, use_preped_text, required=True)

        note_text_pairs = [
            (note, note.preprocessed_text if effective_use_preped_text else note.text)
            for patient in patients
            for note in patient.notes
        ]
        note_text_pairs = [(note, text) for note, text in note_text_pairs if text]
        texts = [text for _, text in note_text_pairs]

        results = self.pipe(texts, top_k=1)

        for (note, _), result in zip(note_text_pairs, results, strict=False):
            note.predicted_label = result[0]["label"]
            note.predicted_score = round(result[0]["score"], 2)

        return patients
