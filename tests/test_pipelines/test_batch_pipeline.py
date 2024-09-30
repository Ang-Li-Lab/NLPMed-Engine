import json

import pytest

from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.batch_pipeline import BatchPipeline


@pytest.fixture
def patient_data() -> dict:
    # Sample patient data for testing
    return {
        "patient_id": "123",
        "notes": [
            {
                "text": "Chief Complaint: This is the first note. It mentions DVT and other relevant medical terms.",
            },
            {
                "text": "Assessment: This is the second note. It discusses PE and similar conditions in detail.",
            },
        ],
    }


@pytest.fixture
def default_config() -> dict:
    # Default configuration with all components enabled
    return {
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
            "device": "cpu",
            "ml_model_path": "prajjwal1/bert-mini",
            "ml_tokenizer_path": "prajjwal1/bert-mini",
        },
    }


def test_process_batch_pipeline(patient_data: dict, default_config: dict) -> None:
    # Prepare multiple patients for testing
    patients = [Patient.from_json(json.dumps(patient_data)) for _ in range(5)]  # Create a batch of 5 patients

    # Initialize the BatchPipeline with the default configuration
    pipeline = BatchPipeline(config=default_config)

    # Process the batch of patients
    processed_patients = pipeline.process(patients, config=default_config)

    # Assertions to ensure each patient in the batch was processed correctly
    assert len(processed_patients) == len(
        patients,
    )  # Ensure the same number of patients were returned

    for processed_patient in processed_patients:
        assert processed_patient.patient_id == "123"
        assert processed_patient.notes  # Ensure notes were processed
        assert all(note.preprocessed_text for note in processed_patient.notes)  # Check that preprocessed text is set


def test_exclude_components_batch_pipeline(
    patient_data: dict,
    default_config: dict,
) -> None:
    # Test excluding specific components in BatchPipeline processing
    config = default_config.copy()
    config["duplicate_checker"]["status"] = "excluded"
    config["sentence_segmenter"]["status"] = "excluded"

    patients = [Patient.from_json(json.dumps(patient_data)) for _ in range(3)]
    pipeline = BatchPipeline(config=config)
    processed_patients = pipeline.process(patients, config=config)

    # Assertions to ensure excluded components did not alter processing
    assert all(pipeline.components["duplicate_checker"]["component"] is None for patient in processed_patients)
    assert all(pipeline.components["sentence_segmenter"]["component"] is None for patient in processed_patients)


def test_disabled_component_batch_pipeline(
    patient_data: dict,
    default_config: dict,
) -> None:
    # Test disabling specific components in BatchPipeline processing
    config = default_config.copy()
    config["sentence_filter"]["status"] = "disabled"

    patients = [Patient.from_json(json.dumps(patient_data)) for _ in range(3)]
    pipeline = BatchPipeline(config=config)
    processed_patients = pipeline.process(patients, config=config)

    # Assertions to ensure processing without disabled component
    assert processed_patients
    for processed_patient in processed_patients:
        for note in processed_patient.notes:
            for section in note.sections:
                assert not any(sentence.is_important for sentence in section.sentences)


def test_runtime_modification_batch_pipeline(
    patient_data: dict,
    default_config: dict,
) -> None:
    # Modify the configuration at runtime and test processing
    patients = [Patient.from_json(json.dumps(patient_data)) for _ in range(3)]
    pipeline = BatchPipeline(config=default_config)

    # Change config at processing time
    process_config = {
        "duplicate_checker": {"status": "enabled", "length_threshold": 10},
    }
    processed_patients = pipeline.process(patients, config=process_config)

    # Assertions to verify runtime modification was respected
    for processed_patient in processed_patients:
        for note in processed_patient.notes:
            for section in note.sections:
                assert all(
                    len(sentence.text) >= 10  # noqa: PLR2004
                    for sentence in section.sentences
                    if sentence.is_duplicate
                )


def test_phase_processing_multiprocessing(
    patient_data: dict,
    default_config: dict,
) -> None:
    # Test phase processing with multiprocessing
    patients = [Patient.from_json(json.dumps(patient_data)) for _ in range(5)]
    pipeline = BatchPipeline(config=default_config)
    preprocessed_params = pipeline.preprocess_params()

    # Manually test the _process_phase_multiprocessing function
    phase_components = ["encoding_fixer", "pattern_replacer", "word_masker"]
    processed_patients = pipeline._process_phase_multiprocessing(  # noqa: SLF001
        patients,
        preprocessed_params,
        phase_components,
    )

    # Assertions to check processing in phases
    assert len(processed_patients) == len(patients)
    assert all(patient.patient_id == "123" for patient in processed_patients)
