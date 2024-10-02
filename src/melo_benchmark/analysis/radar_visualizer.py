import os
from typing import (
    Dict,
    List
)

import pandas as pd
import plotly.express as px

from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class RadarVisualizer:

    def __init__(
                self,
                trec_metric: TrecMetric = None
            ):

        self.trec_metric = trec_metric
        if trec_metric is None:
            self.trec_metric = TrecMetric.RECIP_RANK
        dataset_helper = OfficialDatasetHelper(
            should_create_datasets=False
        )
        self.melo_datasets = dataset_helper.get_dataset_configs()
        self.N = 1
        self.M = 2

    def create_radar_chart(
                self,
                method_names: List[str],
                crosswalk_names: List[str],
                output_file_monolingual_path: str,
                output_file_cross_lingual_path: str
            ):

        for method_name in method_names:
            if method_name.endswith(" (lemmas)"):
                error_m = "Radar chart visualization is not supported for " \
                          + "lemmatized methods"
                raise ValueError(error_m)

        metric_values = self._extract_trec_metrics(method_names)

        mling_dataset_crosswalk_to_id = {}
        xling_dataset_crosswalk_to_id = {}
        for melo_dataset in self.melo_datasets:
            crosswalk_name = melo_dataset.crosswalk_name
            source_lang = melo_dataset.source_language
            target_langs = list(melo_dataset.target_languages)
            if len(target_langs) == 1 and source_lang == target_langs[0]:
                mling_dataset_crosswalk_to_id[crosswalk_name] = melo_dataset
            else:
                xling_dataset_crosswalk_to_id[crosswalk_name] = melo_dataset

        mling_dataset_names = []
        xling_dataset_names = []
        for crosswalk_name in crosswalk_names:
            m_dataset_conf = mling_dataset_crosswalk_to_id[crosswalk_name]
            mling_dataset_names.append(m_dataset_conf.dataset_name)
            x_dataset_conf = xling_dataset_crosswalk_to_id[crosswalk_name]
            xling_dataset_names.append(x_dataset_conf.dataset_name)

        mling_r_per_method = {}
        xling_r_per_method = {}
        for method_name, results in metric_values.items():
            mling_r_per_method[method_name] = []
            xling_r_per_method[method_name] = []
            for crosswalk_name in crosswalk_names:
                m_dataset_conf = mling_dataset_crosswalk_to_id[crosswalk_name]
                m_dataset_id = m_dataset_conf.dataset_dir_name
                metric_value = results[m_dataset_id]
                mling_r_per_method[method_name].append(metric_value)
                x_dataset_conf = xling_dataset_crosswalk_to_id[crosswalk_name]
                x_dataset_id = x_dataset_conf.dataset_dir_name
                metric_value = results[x_dataset_id]
                xling_r_per_method[method_name].append(metric_value)

        # Monolingual Datasets
        r_vals = []
        theta_vals = []
        name_vals = []
        for method_name in method_names:
            n = len(mling_dataset_names)
            r_vals.extend(mling_r_per_method[method_name])
            theta_vals.extend(mling_dataset_names)
            name_vals.extend([method_name] * n)
        df = pd.DataFrame(dict(
            r=r_vals,
            theta=theta_vals,
            method_name=name_vals
        ))
        fig = px.line_polar(
            df,
            r="r",
            theta="theta",
            color="method_name",
            line_close=True
        )
        fig.update_layout(
            title=dict(
                text="MRR on Monolingual Datasets"
            ),
            legend_title_text="Method"
        )
        fig.update_polars(
            radialaxis=dict(
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
        )
        fig.write_image(output_file_monolingual_path)

        # Cross-lingual Datasets
        r_vals = []
        theta_vals = []
        name_vals = []
        for method_name in method_names:
            n = len(mling_dataset_names)
            r_vals.extend(xling_r_per_method[method_name])
            theta_vals.extend(xling_dataset_names)
            name_vals.extend([method_name] * n)
        df = pd.DataFrame(dict(
            r=r_vals,
            theta=theta_vals,
            method_name=name_vals
        ))
        fig = px.line_polar(
            df,
            r="r",
            theta="theta",
            color="method_name",
            line_close=True
        )
        fig.update_layout(
            title=dict(
                text="MRR on Cross-lingual Datasets"
            ),
            legend_title_text="Method"
        )
        fig.update_polars(
            radialaxis=dict(
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
        )
        fig.write_image(output_file_cross_lingual_path)

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

        with open(file_path, encoding="utf-8") as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id == self.trec_metric:
                    return metric_value

        raise KeyError(f"Could not find metric {self.trec_metric}.")
