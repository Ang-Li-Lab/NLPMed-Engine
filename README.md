<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Omid Jafari <omidjafari.com>
SPDX-License-Identifier: AGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-->

# NLPMed-Engine

**The NLP backend for NLPMed-Portal.**

NLPMed-Engine is a robust and extensible natural language processing engine tailored for medical text. It supports a range of NLP tasks commonly used in clinical and biomedical applications.

> ⚠️ **Important:** This software is intended for research use only. It must not be used in real-world medical or clinical decision-making settings.

![Static Badge](https://img.shields.io/badge/license-AGPLv3-blue)

---

## Installation

## Option 1 - Install nlpmed_engine Python package:

```bash
git clone https://github.com/ang-li-lab/NLPMed-Engine.git
cd NLPMed-Engine
```

Install based on your environment:

- **CPU**:
  ```bash
  pip install -e .
  ```

- **Apple GPU (MPS)**:
  ```bash
  pip install -e ".[gpu_apple]"
  ```

- **Apple CUDA 11**:
  ```bash
  pip install -e ".[gpu_cuda11]"
  ```

- **Apple CUDA 12**:
  ```bash
  pip install -e ".[gpu_cuda12]"

## Option 2 - Install dependencies only:

```bash
git clone https://github.com/ang-li-lab/NLPMed-Engine.git
cd NLPMed-Engine
```

Install dependencies based on your environment:

- **CPU**:
  ```bash
  pip install -r requirements/cpu.txt
  ```

- **Apple GPU (MPS)**:
  ```bash
  pip install -r requirements/gpu_apple.txt
  ```

- **Apple CUDA 11**:
  ```bash
  pip install -r requirements/gpu_guda11.txt
  ```

- **Apple CUDA 12**:
  ```bash
  pip install -r requirements/gpu_guda12.txt
  ```

## Usage

### Run REST API

1. Create a `.env` file with the following template:

    ```ini
    API_ML_MODEL_NAMES=VTE,BLEED

    API_ML_VTE_MULTICLASS_DEVICE=cpu
    API_ML_VTE_MULTICLASS_MODEL_PATH=/Users/model
    API_ML_VTE_MULTICLASS_TOKENIZER_PATH=/Users/tokenizer
    API_ML_VTE_MULTICLASS_MAX_LENGTH=512

    API_ML_BLEED_BINARY_DEVICE=cuda:0
    API_ML_BLEED_BINARY_MODEL_PATH=/Users/model
    API_ML_BLEED_BINARY_TOKENIZER_PATH=/Users/tokenizer
    API_ML_BLEED_BINARY_MAX_LENGTH=512

    API_HOST=127.0.0.1
    API_PORT=10010
    API_WORKERS=1
    ```

2. Run the API:

   ```bash
   python scripts/run_api.py
   ```

### Run Single or Batch Pipelines

Instead of using the API, you can directly use the `SinglePipeline` or `BatchPipeline` classes in your Python code.

- Sample Jupyter notebooks are provided under the `notebooks/` directory.
- **Note:** Since `BatchPipeline` uses parallel processing, the output order may differ from the input. Always use `patient_id, note_id` when merging results back with the input data.

## Resources

- **Demo**: [Visit our demo site](https://nlpmed.demo.angli-lab.com/)
- **VTE-BERT Model**: Our fine-tuned model optimized for VTE classification is available under gated access on [Hugging Face](https://huggingface.co/ang-li-lab/VTE-BERT).
- **Publication**: Development and Validation of VTE-BERT Natural Language Processing Model for Venous Thromboembolism ([Open Access](https://www.jthjournal.org/article/S1538-7836(25)00484-2/fulltext))

## Documentation

See [documentation](https://ang-li-lab.github.io/NLPMed-Engine/) for full API and module reference (generated with Sphinx).

## Terms of Use

This project includes software, models, or a federated learning framework that are governed by additional terms beyond the AGPLv3 license.

By using this software or model, you agree to the [Terms & Conditions](./TERMS.md).

## Citation

If you use NLPMed-Engine in your research or applications, please cite our paper:

```bibtex
@article{jafaridevelopment,
  title={Development and Validation of VTE-BERT Natural Language Processing Model for Venous Thromboembolism},
  author={Jafari, Omid and Ma, Shengling and Lam, Barbara D and Jiang, Jun Y and Zhou, Emily and Ranjan, Mrinal and Ryu, Justine and Bandyo, Raka and Maghsoudi, Arash and Peng, Bo and others},
  journal={Journal of Thrombosis and Haemostasis},
  publisher={Elsevier},
  year={2025},
  doi={10.1016/j.jtha.2025.07.021}
}
```

---
