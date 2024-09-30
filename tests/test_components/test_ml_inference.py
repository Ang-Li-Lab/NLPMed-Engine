import pytest

from nlpmed_engine.components.ml_inference import MLInference
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient


@pytest.fixture
def ml_inference() -> MLInference:
    # Initialize the MLInference component with the prajjwal1/bert-mini model
    ml_model_path = "prajjwal1/bert-mini"  # Small BERT model for testing
    ml_tokenizer_path = "prajjwal1/bert-mini"
    return MLInference(
        device="cpu",
        ml_model_path=ml_model_path,
        ml_tokenizer_path=ml_tokenizer_path,
    )


def test_ml_inference_process(ml_inference: MLInference) -> None:
    # Create a Note object with preprocessed text
    note = Note(text="Sample note text.")
    note.preprocessed_text = "This is a sample preprocessed note text that is expected to be positive."

    # Run the inference process
    processed_note = ml_inference.process(note, max_length=128)

    # Assertions to check if the predicted_label is set correctly
    assert processed_note.predicted_label is not None
    assert isinstance(
        processed_note.predicted_label,
        str,
    )  # Check that the label is a string


def test_ml_inference_process_use_preped_text_false(ml_inference: MLInference) -> None:
    # Create a Note object with text but no preprocessed text
    note = Note(text="This is a sample note text that is expected to be positive.")
    note.preprocessed_text = ""

    # Run the inference process with use_preped_text=False
    processed_note = ml_inference.process(note, use_preped_text=False, max_length=128)

    # Assertions to check if the predicted_label is set correctly
    assert processed_note.predicted_label is not None
    assert isinstance(processed_note.predicted_label, str)


def test_ml_inference_no_prediction(ml_inference: MLInference) -> None:
    # Create a Note object with empty preprocessed text
    note = Note(text="Sample note text.")
    note.preprocessed_text = ""  # No content to predict on

    # Run the inference process
    processed_note = ml_inference.process(note, max_length=128)

    # Assertions to check that predicted_label is None when no prediction is made
    assert processed_note.predicted_label is None


def test_process_batch_patients(ml_inference: MLInference) -> None:
    # Create patients with notes
    patient1 = Patient(
        patient_id="dummy1",
        notes=[
            Note(
                text="Patient 1 Note 1 text.",
                preprocessed_text="Preprocessed note 1 text.",
            ),
            Note(text="Patient 1 Note 2 text.", preprocessed_text=""),
        ],
    )
    patient2 = Patient(
        patient_id="dummy2",
        notes=[
            Note(
                text="Patient 2 Note 1 text.",
                preprocessed_text="Preprocessed note 2 text.",
            ),
            Note(text="", preprocessed_text=""),
        ],
    )
    patients = [patient1, patient2]

    # Run the batch inference process with use_preped_text=True
    ml_inference.process_batch_patients(patients, use_preped_text=True)

    # Collect notes that should have been processed
    expected_notes = [
        patient1.notes[0],  # Has preprocessed_text
        patient2.notes[0],  # Has preprocessed_text
    ]

    # Check that predictions were made for notes with preprocessed_text
    for note in expected_notes:
        assert note.predicted_label is not None
        assert isinstance(note.predicted_label, str)
        assert note.predicted_score is not None
        assert isinstance(note.predicted_score, float)

    # Check that notes without preprocessed_text were not processed
    assert patient1.notes[1].predicted_label is None
    assert patient2.notes[1].predicted_label is None

    # Now test with use_preped_text=False
    ml_inference.process_batch_patients(patients, use_preped_text=False)

    # All notes except the one with empty 'text' should be processed
    expected_notes = [
        patient1.notes[0],  # Has text
        patient1.notes[1],  # Has text
        patient2.notes[0],  # Has text
    ]

    # Check that predictions were made for notes with text
    for note in expected_notes:
        assert note.predicted_label is not None
        assert isinstance(note.predicted_label, str)
        assert note.predicted_score is not None
        assert isinstance(note.predicted_score, float)

    # Check that note with empty 'text' was not processed
    assert patient2.notes[1].predicted_label is None
