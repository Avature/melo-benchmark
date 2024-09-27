import matplotlib.pyplot as plt
import os
from typing import (
    Dict,
    List
)

import pandas as pd
import seaborn as sns

from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class AccuracyAtKVisualizer:

    def __init__(self):
        dataset_helper = OfficialDatasetHelper(
            should_create_datasets=False
        )
        self.melo_datasets = dataset_helper.get_dataset_configs()
        self.target_metrics = {
            "1": TrecMetric.A_1,
            "5": TrecMetric.A_5,
            "10": TrecMetric.A_10,
        }
        self.N = 2
        self.M = 5

    def create_plot(
                self,
                method_names: List[str],
                crosswalk_names: List[str],
                output_file_path: str
            ):

        crosswalk_to_ds_config = {}
        for ds_config in self.melo_datasets:
            is_monolingual = False
            if len(ds_config.target_languages) == 1 and \
                    ds_config.source_language in ds_config.target_languages:
                is_monolingual = True

            crosswalk_name = ds_config.crosswalk_name
            if crosswalk_name in crosswalk_names:

                if crosswalk_name not in crosswalk_to_ds_config.keys():
                    crosswalk_to_ds_config[crosswalk_name] = {}

                dict_key = "monolingual"
                if not is_monolingual:
                    dict_key = "cross-lingual"

                crosswalk_to_ds_config[crosswalk_name][dict_key] = ds_config

        target_datasets = []
        for crosswalk_name, ds_configs in crosswalk_to_ds_config.items():
            target_datasets.append(ds_configs["monolingual"])
        for crosswalk_name, ds_configs in crosswalk_to_ds_config.items():
            target_datasets.append(ds_configs["cross-lingual"])

        for method_name in method_names:
            if method_name.endswith(" (lemmas)"):
                error_m = "Correlation visualization is not supported for " \
                          + "lemmatized methods"
                raise ValueError(error_m)

        num_methods = len(method_names)

        # Setting for paper's format
        self.N = 2
        self.M = len(crosswalk_names)

        metric_values = self._extract_trec_metrics(method_names)

        df_content = []
        for method_name, results in metric_values.items():
            for melo_dataset in target_datasets:
                dataset_id = melo_dataset.dataset_dir_name
                acc_at_k = results[dataset_id]
                for k, acc in acc_at_k.items():
                    df_row = {
                        "dataset": melo_dataset.dataset_name,
                        "k": k,
                        "model": method_name,
                        "result": acc,
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
            figsize=(4 * self.M, 4 * self.N),
            sharex=True,
            sharey=True
        )

        plt.ylim(0, 1)
        plt.xticks([1, 5, 10])

        # Iterate over rows and columns to populate the grid
        for i in range(self.N):
            for j in range(self.M):
                if self.N == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                target_ds = target_datasets[i * self.M + j]
                dataset_name = target_ds.dataset_name

                df_dataset = df.loc[df["dataset"].isin([dataset_name])]

                legend = False
                if i == 0 and j == self.M - 1:
                    legend = True

                ax = sns.lineplot(
                    data=df_dataset,
                    x="k",
                    y="result",
                    hue="model",
                    style="model",
                    hue_order=method_names,
                    style_order=method_names,
                    palette=sns.color_palette()[:num_methods],
                    ax=ax,
                    legend=legend
                )

                if i == self.N - 1:
                    ax.set_xlabel('k')
                else:
                    ax.set_xlabel('')
                if j == 0:
                    ax.set_ylabel('A@k')
                else:
                    ax.set_ylabel('')

                ax.set_title(dataset_name)

                if i == 0 and j == self.M - 1:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                    plt.setp(ax.get_legend().get_texts(), fontsize='12')
                    plt.setp(ax.get_legend().get_title(), fontsize='12')

        # Adjust layout
        plt.tight_layout()

        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

    def _extract_trec_metrics(
                self,
                method_names: List[str]
            ) -> Dict[str, Dict[str, Dict[str, float]]]:

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
                target_results = self._extract_trec_metric_from_file(
                    results_file_path
                )
                metric_values[method_name][dataset_name] = {}
                for metric_name, result_value in target_results.items():
                    v = float(result_value)
                    metric_values[method_name][dataset_name][metric_name] = v

        return metric_values

    def _extract_trec_metric_from_file(
                self,
                file_path: str,
            ) -> Dict[str, str]:

        reverse_metrics_mapping = {
            trec_metric: metric_name
            for metric_name, trec_metric in self.target_metrics.items()
        }
        target_trec_metrics = reverse_metrics_mapping.keys()

        if not os.path.exists(file_path):
            result = {
                target_metric: "-"
                for target_metric in self.target_metrics.keys()
            }
            return result

        result = {}
        with open(file_path, encoding="utf-8") as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id in target_trec_metrics:
                    metric_name = reverse_metrics_mapping[metric_id]
                    result[metric_name] = metric_value

        for target_metric in self.target_metrics.keys():
            if target_metric not in result.keys():
                raise KeyError(f"Could not find metric {target_metric}.")

        return result
