import logging
import os
import re
from typing import (
    Any,
    Dict,
    Set
)

import numpy as np
import pandas as pd

import melo_benchmark.utils.helper as melo_utils
import melo_benchmark.utils.logging_config as melo_logging
from melo_benchmark.utils.metaclasses import Singleton


melo_logging.setup_logging()
logger = logging.getLogger(__name__)


# ESCO languages that are common to all ESCO versions
#   and with sufficient basic information
SUPPORTED_ESCO_LANGUAGES = [
    "en",
    # "ar",   # No ISCO Groups for this language
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    # "ga",   # No preferred names for this language
    "hr",
    "hu",
    # "is",   # No preferred names for this language
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
    # "uk",   # No info in older ESCO versions for this language
]


SUPPORTED_ESCO_VERSIONS = [
    "1.0.3",
    "1.0.8",
    "1.0.9",
    "1.1.0",
    "1.1.1",
    "1.2.0",
]


class EscoLoader(metaclass=Singleton):
    """
    A class to load and process ESCO data.
    """

    def __init__(self):
        self.loaded_esco_versions: Dict[str, Dict[str, Any]] = {}

    def load(self, esco_version: str) -> Dict[str, Any]:
        if esco_version in self.loaded_esco_versions:
            return self.loaded_esco_versions[esco_version]

        target_esco_version = self._do_load(esco_version)
        self.loaded_esco_versions[esco_version] = target_esco_version

        return target_esco_version

    def _do_load(self, esco_version: str) -> Dict[str, Any]:
        """
        Loads and processes ESCO data, then saves it to a JSON file if
        not already present.

        Returns:
            Dict[str, Any]: The ESCO data as a dictionary.
        """

        assert esco_version in SUPPORTED_ESCO_VERSIONS

        # Define the path for the JSON file
        json_file_path = self._get_json_file_path(
            esco_version
        )

        # Check if the JSON file already exists
        if os.path.exists(json_file_path):
            logger.info(f"Loading data from {json_file_path}...")
            concepts = melo_utils.load_content_from_json_file(json_file_path)
        else:
            # Load and process categories and occupations
            concepts = self._get_all_categories(esco_version)
            concepts = self._add_all_occupations(esco_version, concepts)

            # Format the concepts dictionary as a JSON string
            json_string = melo_utils.serialize_as_json(concepts)

            # Save the JSON string to a file
            with open(json_file_path, 'w', encoding="utf-8") as file:
                file.write(json_string)

            logger.info(f"Data saved to {json_file_path}...")

        return concepts

    def _get_all_categories(self, esco_version: str):
        categories = self._initialize_categories(esco_version)

        for language in SUPPORTED_ESCO_LANGUAGES:
            self._process_categories_for_language(
                esco_version,
                language,
                categories
            )

        return categories

    def _initialize_categories(self, esco_version: str) -> Dict[str, Any]:
        df = self._load_categories_dataset(esco_version, "en")
        categories = {}

        for _, row in df.iterrows():
            cat_id = row["conceptUri"]
            categories[cat_id] = {
                "id": cat_id,
                "std_name": {},
                "alt_names": {},
                "description": {}
            }

        return categories

    def _process_categories_for_language(
                self,
                esco_version: str,
                language: str,
                categories: Dict[str, Dict[str, Any]]
            ) -> None:

        df = self._load_categories_dataset(esco_version, language)
        category_ids: Set[str] = set()

        for _, row in df.iterrows():
            cat_id = row["conceptUri"]
            pref_label = row.get("preferredLabel", "").strip()
            alt_labels = row.get("altLabels", "")
            description = row.get("description", "").strip()

            if not pref_label and language in ["no"]:
                # For Norwegian, use English if no preferred name
                pref_label = categories[cat_id]["std_name"]["en"]
            else:
                pref_label = pref_label.strip()
                pref_label = self._clean_name(pref_label)
                assert pref_label, f"No preferred in {language} for {cat_id}"

            assert cat_id not in category_ids, f"Duplicate cat. ID: {cat_id}"
            category_ids.add(cat_id)

            # Process preferred label
            categories[cat_id]["std_name"][language] = pref_label

            # Process alternative labels
            alt_labels_set = self._process_alt_labels(
                alt_labels,
                pref_label
            )
            if alt_labels_set:
                categories[cat_id]["alt_names"][language] = alt_labels_set

            # Process description
            if description:
                categories[cat_id]["description"][language] = description

    def _load_categories_dataset(
                self,
                esco_version: str,
                language: str
            ) -> pd.DataFrame:

        file_name = f"ISCOGroups_{language}.csv"
        return self._load_dataset(
            esco_version,
            language,
            file_name
        )

    def _add_all_occupations(
                self,
                esco_version: str,
                concepts: Dict[str, Any]
            ):

        concepts = self._initialize_occupations(esco_version, concepts)

        for language in SUPPORTED_ESCO_LANGUAGES:
            self._process_occupations_for_language(
                esco_version,
                language,
                concepts
            )

        return concepts

    def _initialize_occupations(
                self,
                esco_version: str,
                concepts: Dict[str, Any]
            ) -> Dict[str, Any]:

        df = self._load_occupations_dataset(esco_version, "en")

        for _, row in df.iterrows():
            c_id = row["conceptUri"]
            assert c_id not in concepts.keys(), f"Duplicate occup. ID: {c_id}"
            concepts[c_id] = {
                "id": c_id,
                "std_name": {},
                "alt_names": {},
                "description": {}
            }

        return concepts

    def _process_occupations_for_language(
                self,
                esco_version: str,
                language: str,
                concepts: Dict[str, Dict[str, Any]]
            ) -> None:

        df = self._load_occupations_dataset(esco_version, language)
        occup_ids: Set[str] = set()

        for _, row in df.iterrows():
            c_id = row["conceptUri"]
            pref_label = row.get("preferredLabel", "")

            if not pref_label and language in ["no"]:
                # For Norwegian, use English if no preferred name
                pref_label = concepts[c_id]["std_name"]["en"]
            else:
                pref_label = pref_label.strip()
                pref_label = self._clean_name(pref_label)
                assert pref_label, f"No preferred occ in {language} for {c_id}"

            alt_labels = row.get("altLabels", "")
            hidden_labels = row.get("hiddenLabels", "")
            description = row.get("description", "").strip()

            assert c_id not in occup_ids, f"Duplicate cat. ID: {c_id}"
            occup_ids.add(c_id)

            # Process preferred label
            concepts[c_id]["std_name"][language] = pref_label

            # Process alternative labels
            alt_labels_set = self._process_alt_labels(
                alt_labels,
                pref_label
            )
            hidden_labels_set = self._process_alt_labels(
                hidden_labels,
                pref_label
            )
            alt_labels_set = alt_labels_set.union(hidden_labels_set)
            if alt_labels_set:
                concepts[c_id]["alt_names"][language] = alt_labels_set

            # Process description
            if description:
                concepts[c_id]["description"][language] = description

    def _load_occupations_dataset(
                self,
                esco_version: str,
                language: str
            ) -> pd.DataFrame:

        file_name = f"occupations_{language}.csv"
        return self._load_dataset(esco_version, language, file_name)

    @staticmethod
    def _clean_name(s: str) -> str:
        s = s.replace("\n", " ")
        s = s.replace("\t", " ")
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def _process_alt_labels(
                self,
                alt_labels: str,
                pref_label: str
            ) -> Set[str]:

        set_to_exclude = {pref_label, ""}
        alt_labels_split = []
        if alt_labels:
            alt_labels_split = alt_labels.split("\n")
        alt_labels_clean = {
            self._clean_name(label) for label in alt_labels_split
            if self._clean_name(label) not in set_to_exclude
        }
        return alt_labels_clean

    @staticmethod
    def _get_json_file_path(esco_version: str) -> str:
        ds_base_dir = melo_utils.get_data_raw_esco_standard_dir_base_path()

        file_dir = os.path.join(
            ds_base_dir,
            esco_version
        )
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_name = f"occupations.json"
        json_file_path = os.path.join(
            file_dir,
            file_name
        )

        return json_file_path

    @staticmethod
    def _load_dataset(
                esco_version: str,
                language: str,
                file_name: str
            ) -> pd.DataFrame:

        ds_base_dir = melo_utils.get_esco_original_dataset_path(
            esco_version,
            language
        )
        file_path = os.path.join(
            ds_base_dir,
            file_name
        )

        df = pd.read_csv(file_path)
        df = df.replace({np.nan: ""})

        return df
