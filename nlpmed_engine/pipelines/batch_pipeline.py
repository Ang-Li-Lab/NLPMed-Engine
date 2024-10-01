import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.base_pipeline import BasePipeline


def _partial_process_phase(serialized_patient: bytes, components: list) -> bytes:
    patient = pickle.loads(serialized_patient)  # noqa: S301

    for note in patient.notes:
        for component, params in components:
            processed_note = component.process(note, **params)

            if processed_note is None:
                note.preprocessed_text = ""
                break

    return pickle.dumps(patient, protocol=pickle.HIGHEST_PROTOCOL)


class BatchPipeline(BasePipeline):
    def process(
        self,
        patients: list[Patient],
        config: dict | None = None,
        processes: int = 4,
    ) -> list[Patient]:
        preprocessed_params = self.preprocess_params(config)

        phase1_components = [
            "encoding_fixer",
            "pattern_replacer",
            "word_masker",
            "note_filter",
            "section_splitter",
            "section_filter",
        ]
        patients = self._process_phase_multiprocessing(
            patients,
            preprocessed_params,
            phase1_components,
            processes,
        )

        if preprocessed_params.get("sentence_segmenter", {}).get(
            "should_process",
            False,
        ):
            sentence_segmenter = self.components["sentence_segmenter"]["component"]
            patients = sentence_segmenter.process_batch_patients(patients)  # type: ignore[attr-defined]

        phase3_components = [
            "duplicate_checker",
            "sentence_filter",
            "sentence_expander",
            "joiner",
        ]
        patients = self._process_phase_multiprocessing(
            patients,
            preprocessed_params,
            phase3_components,
            processes,
        )

        if preprocessed_params.get("ml_inference", {}).get("should_process", False):
            ml_inference = self.components["ml_inference"]["component"]
            patients = ml_inference.process_batch_patients(patients)  # type: ignore[attr-defined]

        return patients

    def _process_phase_multiprocessing(
        self,
        patients: list[Patient],
        preprocessed_params: dict,
        component_names: list[str],
        processes: int = 4,
    ) -> list[Patient]:
        components_to_process = [
            (
                self.components[component_name]["component"],
                preprocessed_params[component_name]["params"],
            )
            for component_name in component_names
            if preprocessed_params.get(component_name, {}).get("should_process", False)
        ]

        if not components_to_process:
            return patients

        serialized_patients = [pickle.dumps(patient, protocol=pickle.HIGHEST_PROTOCOL) for patient in patients]

        with ProcessPoolExecutor(max_workers=processes) as pool:
            process_func = partial(
                _partial_process_phase,
                components=components_to_process,
            )
            serialized_patients = list(
                pool.map(process_func, serialized_patients, chunksize=10),
            )

        return [
            pickle.loads(serialized_patient)  # noqa: S301
            for serialized_patient in serialized_patients
        ]
