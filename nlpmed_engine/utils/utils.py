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

import os
import warnings
from typing import Any
from typing import TypedDict


class ModelSpec(TypedDict):
    device: str
    model_path: str
    tokenizer_path: str
    max_length: int


def get_effective_param(
    instance_value: Any,  # noqa: ANN401
    provided_value: Any,  # noqa: ANN401
    *,
    required: bool = True,
) -> Any:  # noqa: ANN401
    if provided_value is not None:
        return provided_value

    if instance_value is not None:
        return instance_value

    if required:
        message = "A value must be provided."
        raise ValueError(message)

    return None


def read_models_from_env() -> dict[str, ModelSpec]:
    """Read API_ML_MODEL_NAMES and per-model env vars into a dictionary.

    Raises:
        RuntimeError: No model/tokenizer provided.

    Returns:
        dict[str, ModelSpec]: {
            "modelA": {"device": ..., "model_path": ..., "tokenizer_path": ..., "max_length": ...},
            "modelB": {"device": ..., "model_path": ..., "tokenizer_path": ..., "max_length": ...},
        }
    """
    names_str = os.getenv("API_ML_MODEL_NAMES", "")
    names = [n.strip() for n in names_str.split(",") if n.strip()]
    models: dict[str, ModelSpec] = {}

    for name in names:
        prefix = f"API_ML_{name}"
        device = os.getenv(f"{prefix}_DEVICE", "cpu")
        model_path = os.getenv(f"{prefix}_MODEL_PATH", "prajjwal1/bert-mini")
        tokenizer_path = os.getenv(f"{prefix}_TOKENIZER_PATH", "prajjwal1/bert-mini")
        max_length = int(os.getenv(f"{prefix}_MAX_LENGTH", "512"))

        if model_path is None or tokenizer_path is None:
            msg = f"Missing model/tokenizer path for model name {name!r}"
            raise RuntimeError(msg)

        models[name] = {
            "device": device,
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "max_length": max_length,
        }

        if not models:
            warnings.warn("API_ML_MODEL_NAMES is empty!", stacklevel=2)

    return models


def build_initial_config() -> dict[str, Any]:
    """Build the full pipeline config dict, including ml_inference models block.
    The first model name in API_ML_MODEL_NAMES will be treated as the default at runtime.

    Returns:
        dict[str, Any]: The configuration dictionary for the engine.
    """
    models = read_models_from_env()

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
            "models": models,
            "use_preped_text": True,
        },
        "debug": False,
    }
