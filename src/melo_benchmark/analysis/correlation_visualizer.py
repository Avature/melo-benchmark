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
from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class CorrelationVisualizer:

    def __init__(
                self,
                trec_metric: TrecMetric = None,
                show_labels: bool = True
            ):

        self.trec_metric = trec_metric
        self.show_labels = show_labels
        if trec_metric is None:
            self.trec_metric = TrecMetric.RECIP_RANK
        dataset_helper = OfficialDatasetHelper(
            should_create_datasets=False,
            only_monolingual=True
        )
        self.melo_datasets = dataset_helper.get_dataset_configs()
        self.dist_medians = {}
        for melo_dataset in self.melo_datasets:
            df_distances = self._compute_dist_distribution(melo_dataset)
            ds_id = melo_dataset.dataset_dir_name
            self.dist_medians[ds_id] = df_distances["min_dist"].median()
        self.N = 2
        self.M = 2

    def create_scatterplot(
                self,
                method_names: List[str],
                output_file_path: str
            ):

        for method_name in method_names:
            if method_name.endswith(" (lemmas)"):
                error_m = "Correlation visualization is not supported for " \
                          + "lemmatized methods"
                raise ValueError(error_m)

        # Setting for paper's format
        if len(method_names) % 2 != 0:
            self.N = 1
            self.M = 2
        elif len(method_names) == 4:
            self.N = 2
            self.M = 2
        else:
            self.N = 1
            self.M = len(method_names)

        metric_values = self._extract_trec_metrics(method_names)

        df_content = []
        for method_name, results in metric_values.items():
            for melo_dataset in self.melo_datasets:
                dataset_id = melo_dataset.dataset_dir_name
                df_row = {
                    "dataset": melo_dataset.dataset_name,
                    "median": self.dist_medians[dataset_id],
                    "model": method_name,
                    "result": results[dataset_id],
                }
                df_content.append(df_row)

        df = pd.DataFrame(df_content)

        sns.set_theme()
        sns.color_palette()
        sns.set(font_scale=1.3)

        # Create a grid of subplots
        fig, axes = plt.subplots(
            self.N,
            self.M,
            figsize=(4 * self.M + 0.4, 4 * self.N),
            sharex=True,
            sharey=True
        )

        plt.ylim(0, 1)

        # Iterate over rows and columns to populate the grid
        for i in range(self.N):
            for j in range(self.M):
                if self.N == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                target_model = method_names[i * self.M + j]
                df_method = df.loc[df["model"].isin([target_model])]

                sns.scatterplot(
                    data=df_method,
                    x="median",
                    y="result",
                    ax=ax,
                )

                if self.show_labels:
                    for _, row in df_method.iterrows():
                        ax.text(
                            row['median'] + 0.005,
                            row['result'] + 0.005,
                            str(row['dataset'][:3]),
                            fontdict=dict(size=8),
                            alpha=0.4
                        )

                if i == self.N - 1:
                    ax.set_xlabel('Lexical Distance')
                else:
                    ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel('MRR')
                else:
                    ax.set_ylabel('')

                ax.set_title(target_model)

                if i == 0 and j == self.M - 1 and False:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                    plt.setp(ax.get_legend().get_texts(), fontsize='8')
                    plt.setp(ax.get_legend().get_title(), fontsize='12')

        # Adjust layout
        plt.tight_layout()

        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    def create_scatterplot_deltas(
                self,
                method_names: List[str],
                baseline_method: str,
                output_file_path: str
            ):

        for method_name in method_names:
            if method_name.endswith(" (lemmas)"):
                error_m = "Correlation visualization is not supported for " \
                          + "lemmatized methods"
                raise ValueError(error_m)

        # Setting for paper's format
        if len(method_names) % 2 != 0:
            self.N = 1
            self.M = 2
        elif len(method_names) == 4:
            self.N = 2
            self.M = 2
        else:
            self.N = 1
            self.M = len(method_names)

        metric_values = self._extract_trec_metrics(method_names)
        baseline_values = self._extract_trec_metrics([baseline_method])

        df_content = []
        for method_name, results in metric_values.items():
            for melo_dataset in self.melo_datasets:
                dataset_id = melo_dataset.dataset_dir_name
                baseline_results = baseline_values[baseline_method]
                baseline_result = baseline_results[dataset_id]
                df_row = {
                    "dataset": melo_dataset.dataset_name,
                    "median": self.dist_medians[dataset_id],
                    "model": method_name,
                    "result": results[dataset_id] - baseline_result,
                }
                df_content.append(df_row)

        df = pd.DataFrame(df_content)

        sns.set_theme()
        sns.color_palette()
        sns.set(font_scale=1.3)

        # Create a grid of subplots
        fig, axes = plt.subplots(
            self.N,
            self.M,
            figsize=(4 * self.M + 0.4, 4 * self.N),
            sharex=True,
            sharey=True
        )

        plt.ylim(-0.5, 0.5)

        # Iterate over rows and columns to populate the grid
        for i in range(self.N):
            for j in range(self.M):
                if self.N == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                target_model = method_names[i * self.M + j]
                df_method = df.loc[df["model"].isin([target_model])]

                sns.scatterplot(
                    data=df_method,
                    x="median",
                    y="result",
                    ax=ax,
                    color=sns.color_palette()[1]
                )

                if self.show_labels:
                    for _, row in df_method.iterrows():
                        ax.text(
                            row['median'] + 0.005,
                            row['result'] + 0.005,
                            str(row['dataset'][:3]),
                            fontdict=dict(size=8),
                            alpha=0.4
                        )

                ax.axhline(y=0, linewidth=1, color='grey', linestyle='--')

                if i == self.N - 1:
                    ax.set_xlabel('Lexical Distance')
                else:
                    ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel('Δ MRR')
                else:
                    ax.set_ylabel('')

                ax.set_title(target_model)

                if i == 0 and j == self.M - 1 and False:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                    plt.setp(ax.get_legend().get_texts(), fontsize='8')
                    plt.setp(ax.get_legend().get_title(), fontsize='12')

        # Adjust layout
        plt.tight_layout()

        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    def create_correlation_table(
                self,
                method_names: List[str],
                output_file_path: str
            ):

        for method_name in method_names:
            if method_name.endswith(" (lemmas)"):
                error_m = "Correlation visualization is not supported for " \
                          + "lemmatized methods"
                raise ValueError(error_m)

        metric_values = self._extract_trec_metrics(method_names)

        df_content = []
        for method_name, results in metric_values.items():
            for melo_dataset in self.melo_datasets:
                dataset_id = melo_dataset.dataset_dir_name
                df_row = {
                    "dataset": melo_dataset.dataset_name,
                    "median": self.dist_medians[dataset_id],
                    "model": method_name,
                    "result": results[dataset_id],
                }
                df_content.append(df_row)

        df = pd.DataFrame(df_content)

        with open(output_file_path, "w") as f_out:
            for target_model in method_names:
                df_method = df.loc[df["model"].isin([target_model])]

                rho = df_method.corr(
                    method='spearman',
                    min_periods=1,
                    numeric_only=True
                )
                corr_value = round(float(rho["median"]["result"]), 4)
                f_out.write(f"{target_model} & {corr_value} //\n")

    def _compute_dist_distribution(
                self,
                melo_dataset: MeloDatasetConfig
            ) -> pd.DataFrame:

        surface_forms_mapping, annotations_mapping = self._load_mappings(
            melo_dataset
        )

        min_distances = []

        for q_key, c_keys in annotations_mapping.items():
            q_surface_form = self._preprocess(
                surface_forms_mapping[q_key]
            )
            c_surface_forms = []
            for c_key in c_keys:
                c_surface_form = self._preprocess(
                    surface_forms_mapping[c_key]
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
    def _preprocess(text_element: str) -> str:
        text_element = text_element.lower()
        text_element = unicodedata.normalize('NFKD', text_element)
        text_element = text_element.encode('ASCII', 'ignore')
        text_element = text_element.decode("utf-8")
        return text_element

    def _extract_trec_metrics(
                self,
                method_names: List[str]
            ) -> Dict[str, Dict[str, float]]:

        base_results_path = melo_utils.get_results_dir_path()

        dataset_names = [
            melo_dataset.dataset_dir_name
            for melo_dataset in self.melo_datasets
        ]

        metric_values = {}
        for method_name in method_names:
            metric_values[method_name] = {}
            for dataset_name in dataset_names:
                simplified_method_name = melo_utils.simplify_method_name(
                    method_name
                )
                results_file_path = os.path.join(
                    base_results_path,
                    simplified_method_name,
                    dataset_name,
                    "results.txt"
                )
                metric_value = self._extract_trec_metric_from_file(
                    results_file_path
                )
                metric_values[method_name][dataset_name] = float(metric_value)

        return metric_values

    def _extract_trec_metric_from_file(
                self,
                file_path: str,
            ) -> str:

        if not os.path.exists(file_path):
            return "-"

        with open(file_path) as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id == self.trec_metric:
                    return metric_value

        raise KeyError(f"Could not find metric {self.trec_metric}.")
