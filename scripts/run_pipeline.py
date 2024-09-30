from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.pipelines.single_pipeline import SinglePipeline


def run_pipeline(patient_data: str, config: dict) -> Patient:
    patient = Patient.from_json(patient_data)
    pipeline = SinglePipeline(config)
    return pipeline.process(patient)
