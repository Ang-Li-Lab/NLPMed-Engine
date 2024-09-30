"""Mapping functions between Pydantic models and internal data structures.

This module provides a set of mapping functions to convert between Pydantic models
used in the API layer and internal data structures used for processing within
NLPMed-Engine. These mappers ensure data consistency and facilitate seamless
transformation between API inputs/outputs and internal processing logic.

Functions:
    map_pydantic_to_internal_sentence:
        Maps a Pydantic SentenceModel to an internal Sentence object.
    map_internal_to_pydantic_sentence_model:
        Maps an internal Sentence object to a Pydantic SentenceModel.
    map_pydantic_to_internal_section:
        Maps a Pydantic SectionModel to an internal Section object.
    map_internal_to_pydantic_section_model:
        Maps an internal Section object to a Pydantic SectionModel.
    map_pydantic_to_internal_note:
        Maps a Pydantic NoteModel to an internal Note object.
    map_internal_to_pydantic_note_model:
        Maps an internal Note object to a Pydantic NoteModel.
    map_pydantic_to_internal_patient:
        Maps a Pydantic PatientModel to an internal Patient object.
    map_internal_to_pydantic_patient_model:
        Maps an internal Patient object to a Pydantic PatientModel.
"""

from nlpmed_engine.api.models import NoteModel
from nlpmed_engine.api.models import PatientModel
from nlpmed_engine.api.models import SectionModel
from nlpmed_engine.api.models import SentenceModel
from nlpmed_engine.data_structures.note import Note
from nlpmed_engine.data_structures.patient import Patient
from nlpmed_engine.data_structures.section import Section
from nlpmed_engine.data_structures.sentence import Sentence


def map_pydantic_to_internal_sentence(sentence_model: SentenceModel) -> Sentence:
    """Map a Pydantic SentenceModel to an internal Sentence object.

    Args:
        sentence_model (SentenceModel): The Pydantic model representing a sentence.

    Returns:
        Sentence: The internal Sentence object with the corresponding attributes.

    """
    return Sentence(
        text=sentence_model.text,
        start_index=sentence_model.start_index,
        end_index=sentence_model.end_index,
        is_duplicate=sentence_model.is_duplicate,
        is_important=sentence_model.is_important,
        is_expanded=sentence_model.is_expanded,
    )


def map_internal_to_pydantic_sentence_model(sentence: Sentence) -> SentenceModel:
    """Map an internal Sentence object to a Pydantic SentenceModel.

    Args:
        sentence (Sentence): The internal Sentence object to be converted.

    Returns:
        SentenceModel: The corresponding Pydantic model with mapped attributes.

    """
    return SentenceModel(
        text=sentence.text,
        start_index=sentence.start_index,
        end_index=sentence.end_index,
        is_duplicate=sentence.is_duplicate,
        is_important=sentence.is_important,
        is_expanded=sentence.is_expanded,
    )


def map_pydantic_to_internal_section(section_model: SectionModel) -> Section:
    """Map a Pydantic SectionModel to an internal Section object.

    Args:
        section_model (SectionModel): The Pydantic model representing a section.

    Returns:
        Section: The internal Section object with mapped sentences and attributes.

    """
    return Section(
        text=section_model.text,
        start_index=section_model.start_index,
        end_index=section_model.end_index,
        sentences=[map_pydantic_to_internal_sentence(sent) for sent in section_model.sentences],
        important_indices=section_model.important_indices,
        duplicate_indices=section_model.duplicate_indices,
        is_important=section_model.is_important,
    )


def map_internal_to_pydantic_section_model(section: Section) -> SectionModel:
    """Map an internal Section object to a Pydantic SectionModel.

    Args:
        section (Section): The internal Section object to be converted.

    Returns:
        SectionModel: The corresponding Pydantic model with mapped sentences and attributes.

    """
    return SectionModel(
        text=section.text,
        start_index=section.start_index,
        end_index=section.end_index,
        sentences=[map_internal_to_pydantic_sentence_model(sent) for sent in section.sentences],
        important_indices=section.important_indices,
        duplicate_indices=section.duplicate_indices,
        is_important=section.is_important,
    )


def map_pydantic_to_internal_note(note_model: NoteModel) -> Note:
    """Map a Pydantic NoteModel to an internal Note object.

    Args:
        note_model (NoteModel): The Pydantic model representing a note.

    Returns:
        Note: The internal Note object with mapped sections and attributes.

    """
    return Note(
        text=note_model.text,
        preprocessed_text=note_model.preprocessed_text,
        predicted_label=note_model.predicted_label,
        predicted_score=note_model.predicted_score,
        sections=[map_pydantic_to_internal_section(section) for section in note_model.sections],
    )


def map_internal_to_pydantic_note_model(note: Note) -> NoteModel:
    """Map an internal Note object to a Pydantic NoteModel.

    Args:
        note (Note): The internal Note object to be converted.

    Returns:
        NoteModel: The corresponding Pydantic model with mapped sections and attributes.

    """
    return NoteModel(
        text=note.text,
        preprocessed_text=note.preprocessed_text,
        predicted_label=note.predicted_label,
        predicted_score=note.predicted_score,
        sections=[map_internal_to_pydantic_section_model(section) for section in note.sections],
    )


def map_pydantic_to_internal_patient(patient_model: PatientModel) -> Patient:
    """Map a Pydantic PatientModel to an internal Patient object.

    Args:
        patient_model (PatientModel): The Pydantic model representing a patient.

    Returns:
        Patient: The internal Patient object with mapped notes.

    """
    notes = [map_pydantic_to_internal_note(note) for note in patient_model.notes]
    return Patient(
        patient_id=patient_model.patient_id,
        notes=notes,
    )


def map_internal_to_pydantic_patient_model(patient: Patient) -> PatientModel:
    """Map an internal Patient object to a Pydantic PatientModel.

    Args:
        patient (Patient): The internal Patient object to be converted.

    Returns:
        PatientModel: The corresponding Pydantic model with mapped notes.

    """
    notes = [map_internal_to_pydantic_note_model(note) for note in patient.notes]
    return PatientModel(
        patient_id=patient.patient_id,
        notes=notes,
    )
