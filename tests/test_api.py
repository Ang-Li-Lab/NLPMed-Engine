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

import pytest
from fastapi.testclient import TestClient

from nlpmed_engine.api.main import app
from nlpmed_engine.api.models import ConfigModel

HTTP_STATUS_OK = 200
client = TestClient(app)

# Sample patient and configuration data for testing
sample_patient_data = {
    "patient_id": "123",
    "notes": [
        {
            "text": "Chief Complaint: This is the first note. It mentions DVT, PE, and PE CT.",
            "sections": [],
        },
        {
            "text": "Assessment: This is the second note. It discusses PE, PE CT, and other conditions.",
            "sections": [],
        },
    ],
}

sample_text_data = {
    "text": "Chief Complaint: This is a test note mentioning DVT, PE, and PE CT.",
}

sample_config_data = {
    "encoding_fixer": {
        "status": "enabled",
    },
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


@pytest.fixture
def config() -> ConfigModel:
    # Create a configuration object from the sample data
    return ConfigModel(**sample_config_data)  # type: ignore[arg-type]


def test_health_check() -> None:
    response = client.get("/")
    assert response.status_code == HTTP_STATUS_OK
    assert response.json() == {"status": "API is running"}


def test_process_patient_with_disabled_word_masker(config: ConfigModel) -> None:
    # Disable the word_masker component
    modified_config = config.model_copy(update={"word_masker": {"status": "disabled"}})

    response = client.post(
        "/process_patient",
        json={"patient": sample_patient_data, "config": modified_config.model_dump()},
    )
    assert response.status_code == HTTP_STATUS_OK
    data = response.json()
    assert data["patient_id"] == sample_patient_data["patient_id"]
    assert len(data["notes"]) == len(sample_patient_data["notes"])

    # Assert that words like "PE CT" are not masked since word_masker is disabled
    assert "PE CT" in data["notes"][0]["text"]
    assert "PE CT" in data["notes"][1]["text"]


def test_process_text_with_disabled_word_masker(config: ConfigModel) -> None:
    # Disable the word_masker component
    modified_config = config.model_copy(update={"word_masker": {"status": "disabled"}})

    response = client.post(
        "/process_text",
        json={"input_data": sample_text_data, "config": modified_config.model_dump()},
    )
    assert response.status_code == HTTP_STATUS_OK
    data = response.json()

    # Assert that "PE CT" is not masked
    assert "PE CT" in data["preprocessed_text"]
    assert isinstance(data["predicted_label"], str)
    assert len(data["predicted_label"]) > 0  # Ensure some prediction label is returned
    assert isinstance(data["predicted_score"], float)  # Ensure a score is returned


def test_process_patient_with_enabled_word_masker(config: ConfigModel) -> None:
    response = client.post(
        "/process_patient",
        json={"patient": sample_patient_data, "config": config.model_dump()},
    )
    assert response.status_code == HTTP_STATUS_OK
    data = response.json()
    assert data["patient_id"] == sample_patient_data["patient_id"]
    assert len(data["notes"]) == len(sample_patient_data["notes"])

    # Assert that "PE CT" is masked since word_masker is enabled
    assert "PE CT" not in data["notes"][0]["text"]
    assert "*****" in data["notes"][0]["text"]
    assert "PE CT" not in data["notes"][1]["text"]
    assert "*****" in data["notes"][1]["text"]


def test_process_batch_patients_with_disabled_word_masker(config: ConfigModel) -> None:
    # Disable the word_masker component
    modified_config = config.model_copy(update={"word_masker": {"status": "disabled"}})
    batch_data = [sample_patient_data, sample_patient_data]  # Testing with two patients

    response = client.post(
        "/process_batch_patients",
        json={"patients": batch_data, "config": modified_config.model_dump()},
    )
    assert response.status_code == HTTP_STATUS_OK
    data = response.json()
    assert len(data) == len(batch_data)  # Ensure the batch size matches the input
    for patient in data:
        assert patient["patient_id"] == sample_patient_data["patient_id"]
        assert len(patient["notes"]) == len(sample_patient_data["notes"])
        # Assert that "PE CT" is not masked since word_masker is disabled
        assert "PE CT" in patient["notes"][0]["text"]
        assert "PE CT" in patient["notes"][1]["text"]
