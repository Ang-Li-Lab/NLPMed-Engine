from collections.abc import Callable

import pytest

from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.batch_pipeline import BatchPipeline

# Sample configuration for SinglePipeline
sample_config = {
    "encoding_fixer": {"status": "enabled"},
    "pattern_replacer": {
        "status": "enabled",
        "patterns": [r"\s{4,}"],
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
        "device": "cpu",
        "ml_model_path": "prajjwal1/bert-mini",
        "ml_tokenizer_path": "prajjwal1/bert-mini",
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
def batch_pipeline() -> BatchPipeline:
    # Initialize the BatchPipeline with sample configuration
    return BatchPipeline(config=sample_config)


@pytest.mark.benchmark(
    group="BatchPipeline Performance",
    warmup=True,
    warmup_iterations=1,
)
def test_batch_pipeline_performance(
    benchmark: Callable,
    batch_pipeline: BatchPipeline,
) -> None:
    # Generate synthetic data
    num_patients = 100  # Adjust based on the desired load
    num_notes = 10  # Number of notes per patient
    synthetic_patients = generate_synthetic_patients(num_patients, num_notes)

    # Benchmark the performance of the pipeline processing with the process method
    def run_batch_pipeline() -> None:
        batch_pipeline.process(synthetic_patients, processes=4)

    # Run the benchmark
    benchmark(run_batch_pipeline)
