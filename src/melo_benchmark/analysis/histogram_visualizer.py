import matplotlib.pyplot as plt
import os
from typing import (
    Dict,
    List,
    Tuple
)
import unicodedata

import pandas as pd
from rapidfuzz import fuzz
import seaborn as sns

from melo_benchmark.data_processing.official_dataset_helper import (
    MeloDatasetConfig,
    OfficialDatasetHelper
)
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class HistogramVisualizer:

    def __init__(self, custom_crosswalks: List[str] = None):
        dataset_helper = OfficialDatasetHelper(
            should_create_datasets=False,
            only_monolingual=True
        )

        # Default values
        self.melo_datasets = dataset_helper.get_dataset_configs()
        self.N = 6
        self.M = 4

        if custom_crosswalks is not None:
            crosswalk_name_to_dataset = {}
            for melo_dataset in self.melo_datasets:
                crosswalk_name = melo_dataset.crosswalk_name
                if crosswalk_name in custom_crosswalks:
                    crosswalk_name_to_dataset[crosswalk_name] = melo_dataset
            n_used = len(crosswalk_name_to_dataset)
            self.N = 1
            self.M = n_used
            self.melo_datasets = [
                crosswalk_name_to_dataset[crosswalk_name]
                for crosswalk_name in custom_crosswalks
            ]

    def create_histogram(self, output_file_path: str):
        sns.set_theme()
        sns.color_palette()
        sns.set(font_scale=1.3)

        # Create a grid of subplots
        fig, axes = plt.subplots(
            self.N,
            self.M,
            figsize=(4 * self.M, 4 * self.N),
            sharex=True,
            sharey=True
        )

        # Iterate over rows and columns to populate the grid
        for i in range(self.N):
            for j in range(self.M):
                if self.N == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                melo_dataset = self.melo_datasets[i * self.M + j]
                dataset_name = melo_dataset.dataset_name
                df_distances = self._compute_dist_distribution(melo_dataset)

                ax = sns.histplot(
                    data=df_distances,
                    x="min_dist",
                    stat="probability",
                    bins=20,
                    ax=ax
                )

                if i == self.N-1:
                    ax.set_xlabel('Distance')
                else:
                    ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel('Probability')
                else:
                    ax.set_ylabel('')

                ax.set_title(dataset_name)

        # Adjust layout
        plt.tight_layout()

        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    def _compute_dist_distribution(
                self,
                melo_dataset: MeloDatasetConfig
            ) -> pd.DataFrame:

        surface_forms_mapping, annotations_mapping = self._load_mappings(
            melo_dataset
        )

        ascii_normalize = True
        if melo_dataset.source_language == "bg":
            # Avoid ASCII normalization for Bulgarian
            ascii_normalize = False

        min_distances = []

        for q_key, c_keys in annotations_mapping.items():
            q_surface_form = self._preprocess(
                surface_forms_mapping[q_key],
                ascii_normalize=ascii_normalize
            )
            c_surface_forms = []
            for c_key in c_keys:
                c_surface_form = self._preprocess(
                    surface_forms_mapping[c_key],
                    ascii_normalize=ascii_normalize
                )
                c_surface_forms.append(c_surface_form)

            min_dist = 1.
            for c_surface_form in c_surface_forms:
                d = 1 - fuzz.ratio(c_surface_form, q_surface_form) / 100
                if d < min_dist:
                    min_dist = d

            min_distances.append({"min_dist": min_dist})

        return pd.DataFrame(min_distances)

    def _load_mappings(
                self,
                melo_dataset: MeloDatasetConfig
            ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:

        base_data_path = melo_utils.get_data_processed_melo_dir_base_path()
        dataset_name = melo_dataset.dataset_dir_name
        dataset_path = os.path.join(
            base_data_path,
            dataset_name
        )

        surface_forms_mapping = {}

        queries_file_path = os.path.join(
            dataset_path,
            "queries.tsv"
        )
        q_mapping = self._unpack_mapping_file(queries_file_path)

        for q_id, q_surface_form in q_mapping.items():
            if melo_dataset.crosswalk_name.startswith("esp_"):
                # For the Spanish crosswalk, use lower-case surface forms
                q_surface_form = q_surface_form.lower()
            surface_forms_mapping[q_id] = q_surface_form

        corpus_elements_file_path = os.path.join(
            dataset_path,
            "corpus_elements.tsv"
        )
        c_mapping = self._unpack_mapping_file(corpus_elements_file_path)

        for c_id, c_surface_form in c_mapping.items():
            assert c_id not in surface_forms_mapping.keys()
            if melo_dataset.crosswalk_name.startswith("esp_"):
                # For the Spanish crosswalk, use lower-case surface forms
                c_surface_form = c_surface_form.lower()
            surface_forms_mapping[c_id] = c_surface_form

        annotations_file_path = os.path.join(
            dataset_path,
            "annotations.tsv"
        )
        annotations_mapping = self._unpack_annotations_file(
            annotations_file_path
        )

        return surface_forms_mapping, annotations_mapping

    @staticmethod
    def _unpack_mapping_file(mapping_file_path) -> Dict[str, str]:
        mapping_ids_to_surface_forms = {}
        with open(mapping_file_path) as f_in:
            for line in f_in:
                item_id, item_surface_form = line.strip().split('\t')
                mapping_ids_to_surface_forms[item_id] = item_surface_form
        return mapping_ids_to_surface_forms

    @staticmethod
    def _unpack_annotations_file(
                annotations_file_path: str
            ) -> Dict[str, List[str]]:

        q_c_mapping = {}
        with open(annotations_file_path) as f_in:
            for line in f_in:
                q_key, _, c_key, _ = line.strip().split('\t')
                if q_key not in q_c_mapping.keys():
                    q_c_mapping[q_key] = []
                q_c_mapping[q_key].append(c_key)
        return q_c_mapping

    @staticmethod
    def _preprocess(text_element: str, ascii_normalize: bool = True) -> str:
        text_element = text_element.lower()
        if ascii_normalize:
            text_element = unicodedata.normalize('NFKD', text_element)
            text_element = text_element.encode('ASCII', 'ignore')
            text_element = text_element.decode("utf-8")
        return text_element
