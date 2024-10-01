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
