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

from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.base_pipeline import BasePipeline


class SinglePipeline(BasePipeline):
    def process(self, patient: Patient, config: dict | None = None) -> Patient:
        preprocessed_params = self.preprocess_params(config)

        if "duplicate_checker" in preprocessed_params and preprocessed_params["duplicate_checker"]["should_process"]:
            duplicate_checker = self.components.get(
                "duplicate_checker",
                {"component": None},
            ).get(
                "component",
            )
            if duplicate_checker:
                duplicate_checker.clear_lsh(  # type: ignore[attr-defined]
                    **preprocessed_params["duplicate_checker"]["params"],
                )

        for note in patient.notes:
            for component_name, component_data in self.components.items():
                component = component_data["component"]
                if not component:
                    continue

                should_process = preprocessed_params[component_name]["should_process"]
                params = preprocessed_params[component_name]["params"]

                if should_process:
                    processed_note = component.process(note, **params)  # type: ignore[attr-defined]

                    if processed_note is None:
                        note.preprocessed_text = ""
                        break

        return patient
