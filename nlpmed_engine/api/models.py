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

"""Pydantic models for NLPMed-Engine API.

This module defines the Pydantic models used for validating and managing the input,
output, and configuration data for the NLPMed-Engine. The models provide a structured
representation of various components involved in text processing, including sentences,
sections, notes, patients, and different text processing components.

Classes:
    StringInputModel:
        Model for string inputs to be processed.
    ComponentStatusModel:
        Base model for component status.
    EncodingFixerModel:
        Model for encoding fixer component status.
    PatternReplacerModel:
        Model for pattern replacer component with pattern and target replacements.
    WordMaskerModel:
        Model for word masker component with mask settings.
    NoteFilterModel:
        Model for filtering notes based on keywords.
    SectionSplitterModel:
        Model for splitting sections using a delimiter.
    SectionFilterModel:
        Model for filtering sections based on include and exclude keywords.
    SentenceSegmenterModel:
        Model for sentence segmentation settings.
    DuplicateCheckerModel:
        Model for duplicate checking configuration.
    SentenceFilterModel:
        Model for filtering sentences based on keywords.
    SentenceExpanderModel:
        Model for expanding short sentences.
    JoinerModel:
        Model for joining sentences and sections.
    MLInferenceModel:
        Model for machine learning inference settings.
    ConfigModel:
        Configuration model for all text processing components.
    SentenceModel:
        Model for representing a sentence with various attributes.
    SectionModel:
        Model for representing a section with sentences.
    NoteModel:
        Model for representing a note with sections and preprocessed text.
    PatientModel:
        Model for representing a patient with associated notes.
    TextProcessingResponseModel:
        Model for responses related to text processing output.
"""

from pydantic import BaseModel
from pydantic import Field


class StringInputModel(BaseModel):
    """Model for string inputs to be processed.

    Attributes:
        text (str): Text input to be processed.

    """

    text: str = Field(..., description="Text input to be processed.")


class ComponentStatusModel(BaseModel):
    """Base model for component status.

    Attributes:
        status (str): Status of the component ('enabled', 'disabled', 'excluded').

    """

    status: str = Field(
        ...,
        description="Status of the component ('enabled', 'disabled', 'excluded').",
    )


class EncodingFixerModel(ComponentStatusModel):
    """Model for encoding fixer component status."""


class PatternReplacerModel(ComponentStatusModel):
    """Model for pattern replacer component with pattern and target replacements.

    Attributes:
        pattern (str): Regex pattern to replace in the text.
        target (str): Target string to replace matched pattern.

    """

    pattern: str = Field(
        default=r"\s{4,}",
        description="Regex pattern to replace in the text.",
    )
    target: str = Field(
        default="\n\n",
        description="Target string to replace matched pattern.",
    )


class WordMaskerModel(ComponentStatusModel):
    """Model for word masker component with mask settings.

    Attributes:
        words_to_mask (list[str]): List of words to mask in the text.
        mask_char (str): Character used for masking.

    """

    words_to_mask: list[str] = Field(
        default_factory=list,
        description="List of words to mask in the text.",
    )
    mask_char: str = Field(default="*", description="Character used for masking.")


class NoteFilterModel(ComponentStatusModel):
    """Model for filtering notes based on keywords.

    Attributes:
        words_to_search (list[str]): Keywords to search in the notes.

    """

    words_to_search: list[str] = Field(
        default_factory=list,
        description="Keywords to search in the notes.",
    )


class SectionSplitterModel(ComponentStatusModel):
    """Model for splitting sections using a delimiter.

    Attributes:
        delimiter (str): Delimiter used to split sections.

    """

    delimiter: str = Field(
        default="\n\n",
        description="Delimiter used to split sections.",
    )


class SectionFilterModel(ComponentStatusModel):
    """Model for filtering sections based on include and exclude keywords.

    Attributes:
        section_inc_list (list[str]): Keywords for including sections.
        section_exc_list (list[str]): Keywords for excluding sections.
        fallback (bool): Enable fallback behavior if no sections match.

    """

    section_inc_list: list[str] = Field(
        default_factory=list,
        description="Keywords for including sections.",
    )
    section_exc_list: list[str] = Field(
        default_factory=list,
        description="Keywords for excluding sections.",
    )
    fallback: bool = Field(
        default=False,
        description="Enable fallback behavior if no sections match.",
    )


class SentenceSegmenterModel(ComponentStatusModel):
    """Model for sentence segmentation settings.

    Attributes:
        nlp_model_name (str): Name of the model used for sentence segmentation.
        batch_size (int): Batch size for processing.

    """

    nlp_model_name: str = Field(
        default="en_core_sci_lg",
        description="Name of the model used for sentence segmentation.",
    )
    batch_size: int = Field(default=10, description="Batch size for processing.")


class DuplicateCheckerModel(ComponentStatusModel):
    """Model for duplicate checking configuration.

    Attributes:
        num_perm (int): Number of permutations for MinHash.
        sim_threshold (float): Similarity threshold for duplicates.
        length_threshold (int): Length threshold for checking duplicates.

    """

    num_perm: int = Field(
        default=256,
        description="Number of permutations for MinHash.",
    )
    sim_threshold: float = Field(
        default=0.9,
        description="Similarity threshold for duplicates.",
    )
    length_threshold: int = Field(
        default=50,
        description="Length threshold for checking duplicates.",
    )


class SentenceFilterModel(ComponentStatusModel):
    """Model for filtering sentences based on keywords.

    Attributes:
        words_to_search (list[str]): Keywords to filter sentences.

    """

    words_to_search: list[str] = Field(
        default_factory=list,
        description="Keywords to filter sentences.",
    )


class SentenceExpanderModel(ComponentStatusModel):
    """Model for expanding short sentences.

    Attributes:
        length_threshold (int): Threshold length for expanding short sentences.

    """

    length_threshold: int = Field(
        default=50,
        description="Threshold length for expanding short sentences.",
    )


class JoinerModel(ComponentStatusModel):
    """Model for joining sentences and sections.

    Attributes:
        sentence_delimiter (str): Delimiter for joining sentences.
        section_delimiter (str): Delimiter for joining sections.

    """

    sentence_delimiter: str = Field(
        default="\n",
        description="Delimiter for joining sentences.",
    )
    section_delimiter: str = Field(
        default="\n\n",
        description="Delimiter for joining sections.",
    )


class MLInferenceModel(ComponentStatusModel):
    """Model for machine learning inference settings.

    Attributes:
        device (str): Device used for model inference (e.g., 'cpu', 'cuda').
        ml_model_path (str): Path to the model.
        ml_tokenizer_path (str): Path to the tokenizer.

    """

    device: str = Field(
        default="cpu",
        description="Device used for model inference (e.g., 'cpu', 'cuda').",
    )
    ml_model_path: str = Field(..., description="Path to the model.")
    ml_tokenizer_path: str = Field(..., description="Path to the tokenizer.")


class ConfigModel(BaseModel):
    """Configuration model for all text processing components.

    Attributes:
        encoding_fixer (dict | None): Configuration for the encoding fixer component.
        pattern_replacer (dict | None): Configuration for the pattern replacer component.
        word_masker (dict | None): Configuration for the word masker component.
        note_filter (dict | None): Configuration for the note filter component.
        section_splitter (dict | None): Configuration for the section splitter component.
        section_filter (dict | None): Configuration for the section filter component.
        sentence_segmenter (dict | None): Configuration for the sentence segmenter component.
        duplicate_checker (dict | None): Configuration for the duplicate checker component.
        sentence_filter (dict | None): Configuration for the sentence filter component.
        sentence_expander (dict | None): Configuration for the sentence expander component.
        joiner (dict | None): Configuration for the joiner component.
        ml_inference (dict | None): Configuration for the machine learning inference component.

    """

    encoding_fixer: dict | None = None
    pattern_replacer: dict | None = None
    word_masker: dict | None = None
    note_filter: dict | None = None
    section_splitter: dict | None = None
    section_filter: dict | None = None
    sentence_segmenter: dict | None = None
    duplicate_checker: dict | None = None
    sentence_filter: dict | None = None
    sentence_expander: dict | None = None
    joiner: dict | None = None
    ml_inference: dict | None = None
    debug: bool = False


class SentenceModel(BaseModel):
    """Model for representing a sentence with various attributes.

    Attributes:
        text (str): Text of the sentence.
        start_index (int): Start index of the sentence in the original text.
        end_index (int): End index of the sentence in the original text.
        is_duplicate (bool): Indicates if the sentence is marked as duplicate.
        is_important (bool): Indicates if the sentence is marked as important.
        is_expanded (bool): Indicates if the sentence has been expanded.

    """

    text: str = Field(..., description="Text of the sentence")
    start_index: int = Field(
        ...,
        description="Start index of the sentence in the original text",
    )
    end_index: int = Field(
        ...,
        description="End index of the sentence in the original text",
    )
    is_duplicate: bool = Field(
        default=False,
        description="Indicates if the sentence is marked as duplicate",
    )
    is_important: bool = Field(
        default=False,
        description="Indicates if the sentence is marked as important",
    )
    is_expanded: bool = Field(
        default=False,
        description="Indicates if the sentence has been expanded",
    )


class SectionModel(BaseModel):
    """Model for representing a section with sentences.

    Attributes:
        text (str): Text of the section.
        start_index (int): Start index of the section in the original text.
        end_index (int): End index of the section in the original text.
        sentences (list[SentenceModel]): List of sentences in the section.
        important_indices (list[int]): Indices of important sentences in the section.
        duplicate_indices (list[int]): Indices of duplicate sentences in the section.
        is_important (bool): Indicates if the section is marked as important.

    """

    text: str = Field(..., description="Text of the section")
    start_index: int = Field(
        ...,
        description="Start index of the section in the original text",
    )
    end_index: int = Field(
        ...,
        description="End index of the section in the original text",
    )
    sentences: list[SentenceModel] = Field(
        default_factory=list,
        description="List of sentences in the section",
    )
    important_indices: list[int] = Field(
        default_factory=list,
        description="Indices of important sentences in the section",
    )
    duplicate_indices: list[int] = Field(
        default_factory=list,
        description="Indices of duplicate sentences in the section",
    )
    expanded_indices: list[int] = Field(
        default_factory=list,
        description="Indices of expanded sentences in the section",
    )
    is_important: bool = Field(
        default=False,
        description="Indicates if the section is marked as important",
    )


class NoteModel(BaseModel):
    """Model for representing a note with sections and preprocessed text.

    Attributes:
        text (str): Text of the note.
        sections (list[SectionModel]): List of sections in the note.
        preprocessed_text (str | None): Preprocessed text of the note.
        predicted_label (str | None): Predicted label from the model inference.
        predicted_score (float | None): Predicted score from the model inference.

    """

    text: str = Field(..., description="Text of the note")
    sections: list[SectionModel] = Field(
        default_factory=list,
        description="List of sections in the note",
    )
    preprocessed_text: str | None = Field(
        default=None,
        description="Preprocessed text of the note",
    )
    predicted_label: str | None = Field(
        default=None,
        description="Predicted label from the model inference",
    )
    predicted_score: float | None = Field(
        default=None,
        description="Predicted score from the model inference",
    )


class PatientModel(BaseModel):
    """Model for representing a patient with associated notes.

    Attributes:
        patient_id (str): Unique identifier for the patient.
        notes (list[NoteModel]): List of notes associated with the patient.

    """

    patient_id: str = Field(..., description="Unique identifier for the patient")
    notes: list[NoteModel] = Field(
        ...,
        description="List of notes associated with the patient",
    )


class TextProcessingResponseModel(BaseModel):
    """Model for responses related to text processing output.

    Attributes:
        preprocessed_text (str | None): The preprocessed text output.
        predicted_label (str | None): The predicted label from the model inference.
        predicted_score (float | None): The prediction score associated with the predicted label.
        note (NoteModel | None): The note object returned in debug mode.

    """

    preprocessed_text: str | None = Field(
        None,
        description="The preprocessed text output.",
    )
    predicted_label: str | None = Field(
        None,
        description="The predicted label from the model inference.",
    )
    predicted_score: float | None = Field(
        None,
        description="The prediction score associated with the predicted label.",
    )
    note: NoteModel | None = Field(
        None,
        description="The note object returned in debug mode.",
    )


class MlModelInfo(BaseModel):
    name: str
    device: str
    max_length: int
    loaded: bool
    loaded_at: str | None = None


class MlModelsResponse(BaseModel):
    default_name: str
    models: list[MlModelInfo]
