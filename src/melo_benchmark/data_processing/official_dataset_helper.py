from dataclasses import dataclass
from typing import (
    List,
    Set
)

from melo_benchmark.data_processing.dataset_builder import DatasetBuilder


@dataclass
class MeloDatasetConfig:
    dataset_name: str
    crosswalk_name: str
    source_language: str
    target_languages: Set[str]
    dataset_dir_name: str = None
    is_ascii_normalizable: bool = True
    has_spacy_lemmatizer: bool = True


ONET_CROSSWALK = "usa_en"

EUROPEAN_CROSSWALKS = [
    "aut_de",
    "bel_fr",
    "bel_nl",
    "bgr_bg",
    "cze_cs",
    "deu_de",
    "dnk_da",
    "esp_es",
    "est_et",
    "fra_fr",
    "hrv_hr",
    "hun_hu",
    "ita_it",
    "ltu_lt",
    "lva_lv",
    "nld_nl",
    "nor_no",
    "pol_pl",
    "prt_pt",
    "rou_ro",
    "svk_sk",
    "svn_sl",
    "swe_sv",
]

TARGET_LANGUAGES_MULTILINGUAL = {
    "de",
    "en",
    "es",
    "fr",
    "it",
    "nl",
    "pl",
    "pt"
}

LANGUAGES_WITH_SPACY_MODEL = [
    "en",
    "da",
    "de",
    "es",
    "fr",
    "hr",
    "it",
    "lt",
    "nl",
    "no",
    "pt",
    "pl",
    "ro",
    "sl",
    "sv"
]


class OfficialDatasetHelper:

    def __init__(
                self,
                should_create_datasets: bool = True,
                only_monolingual: bool = False
            ):

        self.should_create_datasets = should_create_datasets
        self.only_monolingual = only_monolingual

    def get_dataset_configs(self) -> List[MeloDatasetConfig]:
        datasets = []

        usa_en_en = MeloDatasetConfig(
            dataset_name="USA-en-en",
            crosswalk_name=ONET_CROSSWALK,
            source_language="en",
            target_languages={"en"},
            is_ascii_normalizable=True,
            has_spacy_lemmatizer=True
        )
        datasets.append(usa_en_en)

        if not self.only_monolingual:
            usa_en_en = MeloDatasetConfig(
                dataset_name="USA-en-xx",
                crosswalk_name=ONET_CROSSWALK,
                source_language="en",
                target_languages=TARGET_LANGUAGES_MULTILINGUAL,
                is_ascii_normalizable=True,
                has_spacy_lemmatizer=True
            )
            datasets.append(usa_en_en)

        for crosswalk_name in EUROPEAN_CROSSWALKS:
            name_parts = crosswalk_name.split("_")
            country_code = name_parts[0].upper()
            source_language = name_parts[1]

            is_ascii_normalizable = True
            if source_language == "bg":
                # Avoid ASCII normalization for Bulgarian
                is_ascii_normalizable = False

            has_spacy_lemmatizer = True
            if source_language not in LANGUAGES_WITH_SPACY_MODEL:
                has_spacy_lemmatizer = False

            ds_name = f"{country_code}-{source_language}-{source_language}"
            monolingual_dataset = MeloDatasetConfig(
                dataset_name=ds_name,
                crosswalk_name=crosswalk_name,
                source_language=source_language,
                target_languages={source_language},
                is_ascii_normalizable=is_ascii_normalizable,
                has_spacy_lemmatizer=has_spacy_lemmatizer
            )
            datasets.append(monolingual_dataset)

            if not self.only_monolingual:
                cross_lingual_dataset = MeloDatasetConfig(
                    dataset_name=f"{country_code}-{source_language}-en",
                    crosswalk_name=crosswalk_name,
                    source_language=source_language,
                    target_languages={"en"},
                    is_ascii_normalizable=is_ascii_normalizable,
                    has_spacy_lemmatizer=has_spacy_lemmatizer
                )
                datasets.append(cross_lingual_dataset)

        self.create_datasets(datasets)

        return datasets

    def create_datasets(self, datasets: List[MeloDatasetConfig]):
        for dataset in datasets:
            source_taxonomy = dataset.crosswalk_name
            dataset_builder = DatasetBuilder(source_taxonomy)

            if self.should_create_datasets:
                # This creates the dataset in `data/processed/melo/`
                target_languages = list(dataset.target_languages)
                dataset_builder.build(
                    target_languages
                )

            dataset.dataset_dir_name = dataset_builder.compute_dataset_name(
                dataset.target_languages
            )
