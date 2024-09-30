from collections import OrderedDict
from typing import Any
from typing import TypedDict

from nlpmed_engine.components.duplicate_checker import DuplicateChecker
from nlpmed_engine.components.encoding_fixer import EncodingFixer
from nlpmed_engine.components.joiner import Joiner
from nlpmed_engine.components.ml_inference import MLInference
from nlpmed_engine.components.note_filter import NoteFilter
from nlpmed_engine.components.pattern_replacer import PatternReplacer
from nlpmed_engine.components.section_filter import SectionFilter
from nlpmed_engine.components.section_splitter import SectionSplitter
from nlpmed_engine.components.sentence_expander import SentenceExpander
from nlpmed_engine.components.sentence_filter import SentenceFilter
from nlpmed_engine.components.sentence_segmenter import SentenceSegmenter
from nlpmed_engine.components.word_masker import WordMasker


class ComponentDict(TypedDict):
    component: object | None
    status: str | None


class BasePipeline:
    def __init__(self, config: dict) -> None:
        self.components: OrderedDict[str, ComponentDict] = OrderedDict(
            [
                ("encoding_fixer", {"component": None, "status": None}),
                ("pattern_replacer", {"component": None, "status": None}),
                ("word_masker", {"component": None, "status": None}),
                ("note_filter", {"component": None, "status": None}),
                ("section_splitter", {"component": None, "status": None}),
                ("section_filter", {"component": None, "status": None}),
                ("sentence_segmenter", {"component": None, "status": None}),
                ("duplicate_checker", {"component": None, "status": None}),
                ("sentence_filter", {"component": None, "status": None}),
                ("sentence_expander", {"component": None, "status": None}),
                ("joiner", {"component": None, "status": None}),
                ("ml_inference", {"component": None, "status": None}),
            ],
        )

        for component_name, component_data in self.components.items():
            if component_name in config:
                settings = config[component_name]
                component_data["status"] = settings["status"]

                if settings["status"] != "excluded":  # Either enabled or disabled
                    component_data["component"] = self.initializer(
                        component_name,
                        settings,
                    )

    def initializer(self, component_name: str, settings: dict) -> object:
        component_map = {
            "encoding_fixer": self._init_encoding_fixer,
            "pattern_replacer": self._init_pattern_replacer,
            "word_masker": self._init_word_masker,
            "note_filter": self._init_note_filter,
            "section_splitter": self._init_section_splitter,
            "section_filter": self._init_section_filter,
            "sentence_segmenter": self._init_sentence_segmenter,
            "duplicate_checker": self._init_duplicate_checker,
            "sentence_filter": self._init_sentence_filter,
            "sentence_expander": self._init_sentence_expander,
            "joiner": self._init_joiner,
            "ml_inference": self._init_ml_inference,
        }

        if component_name not in component_map:
            message = f"Unknown component: {component_name}"
            raise ValueError(message)

        return component_map[component_name](settings)

    def _init_encoding_fixer(self, _: Any) -> EncodingFixer:  # noqa: ANN401
        return EncodingFixer()

    def _init_pattern_replacer(self, settings: dict) -> PatternReplacer:
        return PatternReplacer(
            patterns=settings["patterns"],
            target=settings.get("target", "\n\n"),
        )

    def _init_word_masker(self, settings: dict) -> WordMasker:
        return WordMasker(
            words_to_mask=settings["words_to_mask"],
            mask_char=settings.get("mask_char", "*"),
        )

    def _init_note_filter(self, settings: dict) -> NoteFilter:
        return NoteFilter(
            words_to_search=settings["words_to_search"],
        )

    def _init_section_splitter(self, settings: dict) -> SectionSplitter:
        return SectionSplitter(
            delimiter=settings.get("delimiter", "\n\n"),
        )

    def _init_section_filter(self, settings: dict) -> SectionFilter:
        return SectionFilter(
            section_inc_list=settings["section_inc_list"],
            section_exc_list=settings["section_exc_list"],
            fallback=settings["fallback"],
        )

    def _init_sentence_segmenter(self, settings: dict) -> SentenceSegmenter:
        return SentenceSegmenter(
            model_name=settings.get("model_name", "en_core_sci_lg"),
            batch_size=settings.get("batch_size", 10),
        )

    def _init_duplicate_checker(self, settings: dict) -> DuplicateChecker:
        return DuplicateChecker(
            num_perm=settings.get("num_perm", 256),
            sim_threshold=settings.get("sim_threshold", 0.9),
            length_threshold=settings["length_threshold"],
        )

    def _init_sentence_filter(self, settings: dict) -> SentenceFilter:
        return SentenceFilter(
            words_to_search=settings["words_to_search"],
        )

    def _init_sentence_expander(self, settings: dict) -> SentenceExpander:
        return SentenceExpander(
            length_threshold=settings["length_threshold"],
        )

    def _init_joiner(self, settings: dict) -> Joiner:
        return Joiner(
            sentence_delimiter=settings.get("sentence_delimiter", "\n"),
            section_delimiter=settings.get("section_delimiter", "\n\n"),
        )

    def _init_ml_inference(self, settings: dict) -> MLInference:
        return MLInference(
            device=settings["device"],
            ml_model_path=settings["ml_model_path"],
            ml_tokenizer_path=settings["ml_tokenizer_path"],
        )

    def preprocess_params(self, config: dict | None = None) -> dict:
        preprocessed_params: dict = {}
        for component_name, component_data in self.components.items():
            if component_data["component"] is None:
                continue

            initial_status = component_data["status"]
            params = {}

            if config and component_name in config:
                current_config = config[component_name]
                status = current_config.get("status", initial_status)
                params = {k: v for k, v in current_config.items() if k != "status"}
            else:
                status = initial_status

            preprocessed_params[component_name] = {
                "should_process": status == "enabled",
                "params": params,
            }
        return preprocessed_params
