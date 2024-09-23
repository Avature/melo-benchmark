from itertools import repeat
from operator import itemgetter
import os
from typing import (
    Any,
    List,
    Tuple
)

from melo_benchmark.evaluation.metrics import TrecMetric
from melo_benchmark.evaluation.scorer import BaseScorer
import melo_benchmark.utils.helper as melo_utils


TREC_METRICS_TO_EXTRACT = [
    TrecMetric.MAP,
    TrecMetric.P_5,
    TrecMetric.P_10,
    TrecMetric.P_20,
    TrecMetric.RECIP_RANK,
    TrecMetric.R_PREC,
    TrecMetric.A_1,
    TrecMetric.A_5,
    TrecMetric.A_10,
    TrecMetric.R_5,
    TrecMetric.R_10,
    TrecMetric.R_20,
]


class Evaluator:

    def __init__(
                self,
                queries_file_path: str,
                corpus_elements_file_path: str,
                annotations_file_path: str,
                output_dir_path: str,
                max_relevant_to_consider: int = 100
            ):

        if not os.path.exists(queries_file_path) \
                or not os.path.exists(corpus_elements_file_path) \
                or not os.path.exists(annotations_file_path):

            raise ValueError("Invalid evaluation dataset")

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        self.q_ids, self.q_surface_forms = self._unpack_mapping(
            queries_file_path
        )
        self.c_ids, self.c_surface_forms = self._unpack_mapping(
            corpus_elements_file_path
        )

        self.queries_file_path = queries_file_path
        self.corpus_elements_file_path = corpus_elements_file_path
        self.annotations_file_path = annotations_file_path

        self.output_dir_path = output_dir_path
        self.aux_scores_file_path = os.path.join(
            output_dir_path,
            "aux_scores.txt"
        )
        self.detailed_res_file = os.path.join(
            output_dir_path,
            "results.txt"
        )
        self.summary_res_file = os.path.join(
            output_dir_path,
            "summary.tsv"
        )

        self.max_relevant = max_relevant_to_consider

        self.trec_eval_binary_path = os.path.join(
            melo_utils.get_resources_dir_path(),
            "trec_eval"
        )

    def evaluate(self, scorer: BaseScorer) -> List[Tuple[str, Any]]:
        scores = scorer.compute_scores(
            self.q_ids,
            self.q_surface_forms,
            self.c_ids,
            self.c_surface_forms
        )
        metrics_values = self._evaluate_scores_with_trec_binary(scores)
        self._store_results(metrics_values)
        return metrics_values

    def _evaluate_scores_with_trec_binary(
                self,
                scores: List[List[float]]
            ) -> List[Tuple[str, Any]]:

        # Store ranking predictions in TREC format
        with open(self.aux_scores_file_path, 'w') as f_out:
            for q_key, scores_q in zip(self.q_ids, scores):
                # TREC expects ranking results as lines with format:
                #     {q_id} Q0 {c_id} 1 {score} STANDARD
                res = list(
                    zip(
                        repeat(q_key, len(self.c_ids)),
                        self.c_ids,
                        scores_q
                    )
                )
                res.sort(key=itemgetter(2), reverse=True)
                if (self.max_relevant > 0) and (len(res) > self.max_relevant):
                    # Limit per-query evaluation to top results
                    res = res[:self.max_relevant]
                for q_id, c_id, score in res:
                    f_out.write(
                        f'{q_id}\tQ0\t{c_id}\t1\t{score:.5f}\tSTANDARD\n'
                    )

        os.chdir(self.output_dir_path)

        # Extract sent metrics from the TREC generated file
        trec_bin = self.trec_eval_binary_path
        expect_qrels = self.annotations_file_path
        aux_file = self.aux_scores_file_path
        trec_command = f"{trec_bin} -m all_trec {expect_qrels} {aux_file}"
        command_to_execute = f"{trec_command} > {self.detailed_res_file}"
        status = os.system(command_to_execute)

        if status != 0:
            # Error case scenario
            print(f'TREC failed with status code: {status}')
            metrics_values = list(
                zip(
                    TREC_METRICS_TO_EXTRACT,
                    [-1] * len(TREC_METRICS_TO_EXTRACT)
                )
            )

        else:
            print(f"TREC scoring successfully computed")
            metrics_values = self._extract_trec_metrics_from_file(
                self.detailed_res_file
            )

            os.remove(aux_file)

        return metrics_values

    @staticmethod
    def _extract_trec_metrics_from_file(
                file_path: str
            ) -> List[Tuple[str, Any]]:

        retrieved_metrics = []
        with open(file_path) as f_in:
            for line in f_in:
                metric_id, _, metric_value = line.strip().split('\t')
                metric_id = metric_id.strip()
                metric_value = metric_value.strip()
                if metric_id in TREC_METRICS_TO_EXTRACT:
                    retrieved_metrics.append((metric_id, metric_value))
        return retrieved_metrics

    @staticmethod
    def _unpack_mapping(mapping_file_path) -> Tuple[List, List]:
        item_keys = []
        item_surface_forms = []
        with open(mapping_file_path) as f_in:
            for line in f_in:
                item_key, item_surface_form = line.strip().split('\t')
                item_keys.append(item_key)
                item_surface_forms.append(item_surface_form)
        return item_keys, item_surface_forms

    def _store_results(self, metrics_values: List):
        summary_file_name = os.path.join(self.summary_res_file)
        with open(summary_file_name, 'w') as f_out:
            f_out.write("metric_name\tmetric_val\n")
            for metric_name, metric_val in metrics_values:
                f_out.write(f"{metric_name}\t{metric_val}\n")
