import pytest

from nlpmed_engine.pipelines.base_pipeline import BasePipeline


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
            "device": "cpu",
            "ml_model_path": "prajjwal1/bert-mini",
            "ml_tokenizer_path": "prajjwal1/bert-mini",
        },
    }


def test_exclude_components_base_pipeline(default_config: dict) -> None:
    # Exclude some components to ensure they are not initialized
    config = default_config.copy()
    config["section_splitter"]["status"] = "excluded"
    config["duplicate_checker"]["status"] = "excluded"

    pipeline = BasePipeline(config=config)

    # Assertions to ensure excluded components are not initialized
    assert pipeline.components["section_splitter"]["component"] is None
    assert pipeline.components["duplicate_checker"]["component"] is None
    assert pipeline.components["section_splitter"]["status"] == "excluded"
    assert pipeline.components["duplicate_checker"]["status"] == "excluded"


def test_disable_components_base_pipeline(default_config: dict) -> None:
    # Disable some components to ensure they are initialized but not processed
    config = default_config.copy()
    config["sentence_filter"]["status"] = "disabled"

    pipeline = BasePipeline(config=config)

    # Assertions to ensure components are initialized but disabled
    assert pipeline.components["sentence_filter"]["component"] is not None
    assert pipeline.components["sentence_filter"]["status"] == "disabled"


def test_initialization_of_components_base_pipeline(default_config: dict) -> None:
    # Ensure that all enabled components are initialized properly
    pipeline = BasePipeline(config=default_config)

    # Assertions to check initialization of all components
    for component_name, component_data in pipeline.components.items():
        if default_config[component_name]["status"] == "enabled":
            assert component_data["component"] is not None
            assert component_data["status"] == "enabled"
        else:
            assert component_data["component"] is None


def test_preprocess_params_base_pipeline(default_config: dict) -> None:
    # Check the behavior of preprocess_params method
    pipeline = BasePipeline(config=default_config)
    preprocessed_params = pipeline.preprocess_params()

    # Assertions to ensure preprocess_params reflects the enabled statuses correctly
    for component_name, component_data in pipeline.components.items():
        expected_status = component_data["status"]
        assert preprocessed_params[component_name]["should_process"] == (expected_status == "enabled")


def test_modify_component_status_at_runtime_base_pipeline(default_config: dict) -> None:
    # Modify component settings during processing
    pipeline = BasePipeline(config=default_config)
    config_modification = {"duplicate_checker": {"status": "disabled"}}
    preprocessed_params = pipeline.preprocess_params(config=config_modification)

    # Assertions to ensure runtime modification reflects correctly
    assert preprocessed_params["duplicate_checker"]["should_process"] is False
