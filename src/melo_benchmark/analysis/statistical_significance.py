import os
from typing import (
    Any,
    Dict,
    List
)

from scipy.stats import wilcoxon

from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class StatisticalSignificanceAnalyzer:

    @staticmethod
    def wilcoxon_test(
                method_a_results: List[float],
                method_b_results: List[float]
            ) -> Any:

        # Wilcoxon signed-rank test
        return wilcoxon(method_a_results, method_b_results)

    def test_over_benchmark(
                self,
                method_names: List[str],
                output_file_path: str,
                trec_metric: TrecMetric = None
            ):

        if trec_metric is None:
            trec_metric = TrecMetric.RECIP_RANK
        dataset_helper = OfficialDatasetHelper(
            should_create_datasets=False
        )
        melo_datasets = dataset_helper.get_dataset_configs()

        dataset_names = [
            melo_dataset.dataset_dir_name
            for melo_dataset in melo_datasets
        ]

        metric_values = self._extract_trec_metrics(
            method_names,
            dataset_names,
            trec_metric
        )

        n = len(method_names)

        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for i in range(n):
                for j in range(n-i-1):
                    method_name_a = method_names[i]
                    method_name_b = method_names[j+i+1]
                    results = self.wilcoxon_test(
                        metric_values[method_name_a],
                        metric_values[method_name_b]
                    )
                    f_out.write(f"\n\n\n")
                    test_header = (
                        "Wilcoxon signed-rank test for methods:\n"
                        f" - {method_name_a}\n"
                        f" - {method_name_b}\n\n"
                    )
                    f_out.write(test_header)
                    f_out.write(f"p-value: {float(results.pvalue)}")
                    f_out.write(f"\n\n\n")

    def _extract_trec_metrics(
                self,
                method_names: List[str],
                dataset_names: List[str],
                trec_metric: TrecMetric
            ) -> Dict[str, List[float]]:

        base_results_path = melo_utils.get_results_dir_path()

        metric_values = {}
        for method_name in method_names:
            metric_values[method_name] = []
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
                    results_file_path,
                    trec_metric
                )
                metric_values[method_name].append(float(metric_value))

        return metric_values

    @staticmethod
    def _extract_trec_metric_from_file(
                file_path: str,
                trec_metric: TrecMetric
            ) -> str:

        if not os.path.exists(file_path):
            return "-"

        with open(file_path, encoding="utf-8") as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id == trec_metric:
                    return metric_value

        raise KeyError(f"Could not find metric {trec_metric}.")
