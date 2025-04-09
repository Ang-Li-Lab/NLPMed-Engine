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

"""API routes for NLPMed-Engine.

This module defines the API endpoints for processing medical text data using NLPMed-Engine's
pipelines. The endpoints handle various tasks, including processing individual patients,
batch processing of patients, and processing standalone text inputs. Each route leverages
Pydantic models to validate inputs and outputs, ensuring data integrity and consistency.

Modules:
    os: Standard library module for accessing environment variables.
    typing: Provides type annotations for enhanced code readability.
    fastapi: The main framework for creating API routes and managing dependencies.

Dependencies:
    SinglePipeline: Configured instance of the single processing pipeline.
    BatchPipeline: Configured instance of the batch processing pipeline.

Functions:
    get_single_pipeline: Dependency that returns the single pipeline instance.
    get_batch_pipeline: Dependency that returns the batch pipeline instance.

Routes:
    /process_patient (POST):
        Processes a single patient with the specified configuration.
    /process_batch_patients (POST):
        Processes a batch of patients with the specified configuration.
    /process_text (POST):
        Processes a standalone text input with the specified configuration.
"""

import os
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from nlpmed_engine.api.mappers import map_internal_to_pydantic_note_model
from nlpmed_engine.api.mappers import map_internal_to_pydantic_patient_model
from nlpmed_engine.api.mappers import map_pydantic_to_internal_patient
from nlpmed_engine.api.models import ConfigModel
from nlpmed_engine.api.models import NoteModel
from nlpmed_engine.api.models import PatientModel
from nlpmed_engine.api.models import StringInputModel
from nlpmed_engine.api.models import TextProcessingResponseModel
from nlpmed_engine.pipelines.batch_pipeline import BatchPipeline
from nlpmed_engine.pipelines.single_pipeline import SinglePipeline

router = APIRouter()

initial_config = {
    "encoding_fixer": {"status": "enabled"},
    "pattern_replacer": {
        "status": "enabled",
        "pattern": r"\s{4,}",
        "target": "\n\n",
    },
    "word_masker": {
        "status": "enabled",
        "words_to_mask": ["PE CT", "DVT ppx"],
        "mask_char": "*",
    },
    "note_filter": {
        "status": "enabled",
        "words_to_search": ["DVT", "PE"],
    },
    "section_splitter": {
        "status": "enabled",
        "delimiter": "\n\n",
    },
    "section_filter": {
        "status": "enabled",
        "section_inc_list": ["Chief Complaint", "Assessment"],
        "section_exc_list": ["Review of System", "System Review"],
        "fallback": True,
    },
    "sentence_segmenter": {
        "status": "enabled",
        "model_name": "en_core_sci_lg",
        "batch_size": 10,
    },
    "duplicate_checker": {
        "status": "enabled",
        "num_perm": 256,
        "sim_threshold": 0.9,
        "length_threshold": 50,
    },
    "sentence_filter": {
        "status": "enabled",
        "words_to_search": ["DVT", "PE"],
    },
    "sentence_expander": {
        "status": "enabled",
        "length_threshold": 50,
    },
    "joiner": {
        "status": "enabled",
        "sentence_delimiter": "\n",
        "section_delimiter": "\n\n",
    },
    "ml_inference": {
        "status": "enabled",
        "device": os.getenv("API_ML_DEVICE", "cpu"),
        "ml_model_path": os.getenv("API_ML_MODEL_PATH", "prajjwal1/bert-mini"),
        "ml_tokenizer_path": os.getenv("API_ML_TOKENIZER_PATH", "prajjwal1/bert-mini"),
    },
    "debug": False,
}

single_pipeline_instance = SinglePipeline(config=initial_config)
batch_pipeline_instance = BatchPipeline(config=initial_config)


def get_single_pipeline() -> SinglePipeline:
    """Provides the singleton instance of the SinglePipeline.

    Returns:
        SinglePipeline: The configured instance of the SinglePipeline for processing.

    """
    return single_pipeline_instance


def get_batch_pipeline() -> BatchPipeline:
    """Provides the singleton instance of the BatchPipeline.

    Returns:
        BatchPipeline: The configured instance of the BatchPipeline for batch processing.

    """
    return batch_pipeline_instance


@router.post("/process_patient")
def process_patient(
    patient: PatientModel,
    config: ConfigModel,
    pipeline: Annotated[SinglePipeline, Depends(get_single_pipeline)],
) -> PatientModel:
    """Processes a single patient using the specified configuration and pipeline.

    Args:
        patient (PatientModel): The patient data to be processed.
        config (ConfigModel): The configuration settings for the processing pipeline.
        pipeline (SinglePipeline): The pipeline instance for processing the patient.

    Returns:
        PatientModel: The processed patient data.

    Raises:
        HTTPException: If an error occurs during the processing of the patient.

    """
    try:
        internal_patient = map_pydantic_to_internal_patient(patient)
        processed_patient = pipeline.process(
            internal_patient,
            config=config.model_dump(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing patient: {e!s}",
        ) from e

    else:
        return map_internal_to_pydantic_patient_model(processed_patient)


@router.post("/process_batch_patients")
def process_batch_patients(
    patients: list[PatientModel],
    config: ConfigModel,
    pipeline: Annotated[BatchPipeline, Depends(get_batch_pipeline)],
) -> list[PatientModel]:
    """Processes a batch of patients using the specified configuration and batch pipeline.

    Args:
        patients (list[PatientModel]): A list of patient data to be processed.
        config (ConfigModel): The configuration settings for the batch processing pipeline.
        pipeline (BatchPipeline): The pipeline instance for batch processing.

    Returns:
        list[PatientModel]: A list of processed patient data.

    Raises:
        HTTPException: If an error occurs during the batch processing of patients.

    """
    try:
        internal_patients = [map_pydantic_to_internal_patient(patient) for patient in patients]
        processed_patients = pipeline.process(
            internal_patients,
            config=config.model_dump(),
        )
        return [map_internal_to_pydantic_patient_model(patient) for patient in processed_patients]

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing batch patients: {e!s}",
        ) from e


@router.post("/process_text", response_model=TextProcessingResponseModel)
def process_text(
    input_data: StringInputModel,
    config: ConfigModel,
    pipeline: Annotated[SinglePipeline, Depends(get_single_pipeline)],
) -> TextProcessingResponseModel | NoteModel:
    """Processes a standalone text input using the specified configuration and pipeline.

    Args:
        input_data (StringInputModel): The text input data to be processed.
        config (ConfigModel): The configuration settings for the processing pipeline.
        pipeline (SinglePipeline): The pipeline instance for processing the text input.

    Returns:
        TextProcessingResponseModel: The response containing preprocessed text, predicted label,
        and predicted score, with additional `NoteModel` if `debug` is True.

    Raises:
        HTTPException: If an error occurs during the processing of the text input.

    """
    try:
        dummy_patient_model = PatientModel(
            patient_id="dummy",
            notes=[
                NoteModel(
                    text=input_data.text,
                    sections=[],
                ),
            ],
        )
        internal_patient = map_pydantic_to_internal_patient(dummy_patient_model)
        processed_patient = pipeline.process(
            internal_patient,
            config=config.model_dump(),
        )
        processed_note = processed_patient.notes[0]

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing text: {e!s}",
        ) from e

    else:
        response = TextProcessingResponseModel(
            preprocessed_text=processed_note.preprocessed_text,
            predicted_label=processed_note.predicted_label,
            predicted_score=processed_note.predicted_score,
            note=None,
        )

        if config.debug:
            response.note = map_internal_to_pydantic_note_model(processed_note)

        return response
