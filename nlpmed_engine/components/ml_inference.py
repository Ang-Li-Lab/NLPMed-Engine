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

"""Machine Learning Inference module for NLPMed-Engine.

This module provides functionality for performing machine learning-based inference
on medical notes using a pre-trained text classification model. The MLInference class
uses the Hugging Face Transformers library to predict labels and scores for text data.

Classes:
    MLInference: Class for performing ML inference on notes and patients.
"""

from datetime import datetime
from operator import itemgetter
from threading import Lock
from typing import Any
from typing import ClassVar
from typing import TypedDict
from zoneinfo import ZoneInfo

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.utils.utils import ModelSpec
from nlpmed_engine.utils.utils import get_effective_param

ModelsConfig = dict[str, ModelSpec]


class LoadedMeta(TypedDict):
    name: str
    device: str
    max_length: int
    loaded_at: str


def _resolve_device(spec: str | None) -> Any:  # noqa: ANN401
    """Returns a device for use in transformers.pipeline

    Args:
        spec (str | None): cpu, mps, cuda, cuda:0, cuda:1, ...

    Returns:
        Any:
            - "cpu" -> torch.device("cpu")
            - "mps" -> torch.device("mps")
            - "cuda" -> 0
            - "cuda:0" -> 0, "cuda:1" -> 1, ...
    """
    s = (spec or "cpu").lower()
    if s.startswith("cuda"):
        if ":" in s:
            return int(s.split(":", 1)[1])

        return 0

    if s == "mps":
        return torch.device("mps")

    return torch.device("cpu")


class MLInference:
    """Class for performing machine learning inference on medical notes.

    This class initializes a text classification pipeline using a specified model and tokenizer.
    It provides methods for predicting labels and scores for text data within notes and patients,
    allowing for batch processing and customization of input parameters.

    Attributes:
        use_preped_text (bool): Whether to use preprocessed text for inference.
        pipe (pipeline): The Hugging Face Transformers pipeline for text classification.
    """

    _cache: ClassVar[dict[tuple[str, str], Any]] = {}
    _lock: ClassVar[Lock] = Lock()
    _loaded_meta: ClassVar[dict[str, LoadedMeta]] = {}

    def __init__(
        self,
        *,
        models: ModelsConfig,
        use_preped_text: bool = True,
    ) -> None:
        """Initializes the MLInference class with the specified model, tokenizer, and settings.

        Args:
            models (ModelsConfig): Models specifications such as model_path, tokenizer_path, etc.
            use_preped_text (bool): Whether to use preprocessed text for inference (default is True).
        """
        self.models: ModelsConfig = models
        self.use_preped_text: bool = use_preped_text
        self._default_name: str = next(iter(self.models))

        for m in self.models:
            self._get_or_load(m)

    @classmethod
    def get_loaded_meta(cls) -> list[LoadedMeta]:
        return [meta.copy() for meta in sorted(cls._loaded_meta.values(), key=itemgetter("name"))]

    def _get_or_load(self, model_name: str) -> pipeline:
        if model_name not in self.models:
            msg = f"Unknown model name: {model_name!r}. Available: {list(self.models)}"
            raise KeyError(msg)

        spec = self.models[model_name]
        cache_key = (model_name, spec["model_path"])

        if cache_key in MLInference._cache:
            return MLInference._cache[cache_key]

        with MLInference._lock:
            if cache_key in MLInference._cache:
                return MLInference._cache[cache_key]

            model = AutoModelForSequenceClassification.from_pretrained(spec["model_path"])
            tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_path"])
            device = _resolve_device(spec.get("device"))

            pipe = pipeline(
                task="text-classification",
                device=device,
                padding="max_length",
                truncation=True,
                model=model,
                tokenizer=tokenizer,
                max_length=spec["max_length"],
            )
            MLInference._cache[cache_key] = pipe

            MLInference._loaded_meta[model_name] = {
                "name": model_name,
                "device": str(device),
                "max_length": int(spec["max_length"]),
                "loaded_at": datetime.now(ZoneInfo("America/Chicago")).replace(microsecond=0).isoformat(),
            }
            return pipe

    def _select_pipe(self, model_name: str | None) -> Any:  # noqa: ANN401
        name = model_name or self._default_name
        return self._get_or_load(name)

    def process(
        self,
        note: Note,
        *,
        use_preped_text: bool | None = None,
        model_name: str | None = None,
        **_: Any,  # noqa: ANN401
    ) -> Note:
        """Performs inference on a single note, predicting a label and score.

        Args:
            note (Note): The note object containing text to be classified.
            use_preped_text (bool | None): Optional override for using preprocessed text.
            model_name (str | None): Optional model to use.

        Returns:
            Note: The note object with predicted label and score updated.

        """
        effective_use_preped_text = get_effective_param(self.use_preped_text, use_preped_text, required=True)

        text_to_infer = note.preprocessed_text if effective_use_preped_text else note.text
        if text_to_infer:
            pipe = self._select_pipe(model_name)
            results = pipe(text_to_infer, top_k=1)
            note.predicted_label = results[0]["label"]
            note.predicted_score = round(results[0]["score"], 2)

        return note

    def process_batch_patients(
        self,
        patients: list[Patient],
        *,
        use_preped_text: bool | None = None,
        model_name: str | None = None,
    ) -> list[Patient]:
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

        pipe = self._select_pipe(model_name)
        results = pipe(texts, top_k=1)

        for (note, _), result in zip(note_text_pairs, results, strict=False):
            note.predicted_label = result[0]["label"]
            note.predicted_score = round(result[0]["score"], 2)

        return patients
