import os
from typing import List

from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


# noinspection DuplicatedCode
class LatexTableHelper:

    def create_table_section(
                self,
                method_names: List[str],
                dataset_names: List[str],
                trec_metric: TrecMetric,
                output_report_name: str
            ):

        base_results_path = melo_utils.get_results_dir_path()

        report_values = {}
        for method_name in method_names:
            report_values[method_name] = {}
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
                report_values[method_name][dataset_name] = metric_value

        base_reports_path = melo_utils.get_reports_dir_path()
        report_file_path = os.path.join(
            base_reports_path,
            f"{output_report_name}.txt"
        )

        table_section_text = ""
        for method_name, method_results in report_values.items():
            # One row per method
            row_content = ["{:<20}".format(method_name)]
            for dataset_name, result_value in method_results.items():
                if result_value == "-":
                    result_value_str = " -         "
                else:
                    result_value_num = "{:.4f}".format(float(result_value))
                    result_value_str = f"{result_value_num}     "
                row_content.append(result_value_str)
            table_section_text += "  " + " & ".join(row_content) + " \\\\ \n"

        with open(report_file_path, "w") as report_file:
            report_file.write(table_section_text)

    @staticmethod
    def _extract_trec_metric_from_file(
                file_path: str,
                target_metric: TrecMetric
            ) -> str:

        if not os.path.exists(file_path):
            return "-"

        with open(file_path) as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id == target_metric:
                    return metric_value

        raise KeyError(f"Could not find metric {target_metric}.")
