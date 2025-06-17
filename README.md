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
   API_ML_DEVICE=cpu or mps or cuda
   API_ML_MODEL_PATH=path/to/model
   API_ML_TOKENIZER_PATH=path/to/tokenizer
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

- See `tests/test_single_pipeline.py` and `tests/test_batch_pipeline.py` for usage examples.

## Resources

- **Demo**: [Visit our demo site](https://nlpmed.demo.angli-lab.com/nlp/demo)
- **VTE-BERT Model**: Our fine-tuned model optimized for VTE classification is available under gated access on [Hugging Face](https://huggingface.co/ang-li-lab/VTE-BERT).

## Documentation

See [documentation](https://ang-li-lab.github.io/NLPMed-Engine/) for full API and module reference (generated with Sphinx).

## Terms of Use

This project includes software, models, or a federated learning framework that are governed by additional terms beyond the AGPLv3 license.

By using this software or model, you agree to the [Terms & Conditions](./TERMS.md).

## Citation

If you use NLPMed-Engine in your research or applications, please cite our paper:

```bibtex
@article{your_article_citation,
  author  = {Your Authors},
  title   = {Title of Your Paper},
  journal = {Journal Name},
  year    = {Year},
  volume  = {Volume},
  pages   = {Pages},
  doi     = {DOI}
}
```

---
