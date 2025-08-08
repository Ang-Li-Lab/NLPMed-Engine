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

from collections.abc import Callable

import pytest

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.single_pipeline import SinglePipeline

# Sample configuration for SinglePipeline
sample_config = {
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
    "note_filter": {"status": "enabled", "words_to_search": ["DVT", "PE"]},
    "section_splitter": {"status": "enabled", "delimiter": "\n\n"},
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
    "sentence_filter": {"status": "enabled", "words_to_search": ["DVT", "PE"]},
    "sentence_expander": {"status": "enabled", "length_threshold": 50},
    "joiner": {
        "status": "enabled",
        "sentence_delimiter": "\n",
        "section_delimiter": "\n\n",
    },
    "ml_inference": {
        "status": "enabled",
        "models": {
            "TEST": {
                "device": "cpu",
                "model_path": "prajjwal1/bert-mini",
                "tokenizer_path": "prajjwal1/bert-mini",
                "max_length": 128,
            },
        },
    },
}


# Function to generate synthetic patient data for heavy load testing
def generate_synthetic_patients(num_patients: int, num_notes: int) -> list[Patient]:
    patients = []
    for i in range(num_patients):
        notes = [
            Note(
                text=f"Note {j} of patient {i} with medical details about DVT, PE, and PE CT.",
                sections=[],
            )
            for j in range(num_notes)
        ]
        patient = Patient(patient_id=str(i), notes=notes)
        patients.append(patient)
    return patients


@pytest.fixture
def single_pipeline() -> SinglePipeline:
    # Initialize the SinglePipeline with sample configuration
    return SinglePipeline(config=sample_config)


@pytest.mark.benchmark(
    group="SinglePipeline Performance",
    warmup=True,
    warmup_iterations=1,
)
def test_single_pipeline_performance(
    benchmark: Callable,
    single_pipeline: SinglePipeline,
) -> None:
    # Generate synthetic data
    num_patients = 100  # Adjust based on the desired load
    num_notes = 10  # Number of notes per patient
    synthetic_patients = generate_synthetic_patients(num_patients, num_notes)

    # Benchmark the performance of the pipeline processing with the process method
    def run_single_pipeline() -> None:
        for patient in synthetic_patients:
            single_pipeline.process(patient)

    # Run the benchmark
    benchmark(run_single_pipeline)
