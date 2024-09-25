import abc
from itertools import repeat
from operator import itemgetter
import os
from typing import (
    Any,
    List,
    Tuple
)

from melo_benchmark.data_processing.official_dataset_helper import \
    MeloDatasetConfig
from melo_benchmark.evaluation.metrics import TrecMetric
from melo_benchmark.evaluation.scorer import (
    BaseScorer,
    BiEncoderScorer
)
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


class BaseEvaluator(abc.ABC):

    def __init__(
                self,
                max_relevant_to_consider: int = 100
            ):

        self.max_relevant = max_relevant_to_consider

        self.trec_eval_binary_path = os.path.join(
            melo_utils.get_resources_dir_path(),
            "trec_eval"
        )

    def _evaluate_scores_with_trec_binary(
                self,
                q_ids: List[str],
                c_ids: List[str],
                annotations_file_path: str,
                scores: List[List[float]],
                output_dir_path: str
            ) -> List[Tuple[str, Any]]:

        aux_scores_file_path = os.path.join(
            output_dir_path,
            "aux_scores.txt"
        )
        detailed_res_file = os.path.join(
            output_dir_path,
            "results.txt"
        )

        # Store ranking predictions in TREC format
        with open(aux_scores_file_path, 'w') as f_out:
            for q_key, scores_q in zip(q_ids, scores):
                # TREC expects ranking results as lines with format:
                #     {q_id} Q0 {c_id} 1 {score} STANDARD
                res = list(
                    zip(
                        repeat(q_key, len(c_ids)),
                        c_ids,
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

        os.chdir(output_dir_path)

        # Extract sent metrics from the TREC generated file
        trec_bin = self.trec_eval_binary_path
        expect_qrels = annotations_file_path
        aux_file = aux_scores_file_path
        trec_command = f"{trec_bin} -m all_trec {expect_qrels} {aux_file}"
        command_to_execute = f"{trec_command} > {detailed_res_file}"
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
                detailed_res_file
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
    def _unpack_mapping(mapping_file_path) -> Tuple[List[str], List[str]]:
        item_keys = []
        item_surface_forms = []
        with open(mapping_file_path) as f_in:
            for line in f_in:
                item_key, item_surface_form = line.strip().split('\t')
                item_keys.append(item_key)
                item_surface_forms.append(item_surface_form)
        return item_keys, item_surface_forms

    @staticmethod
    def _store_results(metrics_values: List, output_dir_path: str):
        summary_res_file = os.path.join(
            output_dir_path,
            "summary.tsv"
        )

        summary_file_name = os.path.join(summary_res_file)
        with open(summary_file_name, 'w') as f_out:
            f_out.write("metric_name\tmetric_val\n")
            for metric_name, metric_val in metrics_values:
                f_out.write(f"{metric_name}\t{metric_val}\n")


class Evaluator(BaseEvaluator):

    def __init__(
                self,
                queries_file_path: str,
                corpus_elements_file_path: str,
                annotations_file_path: str,
                output_dir_path: str,
                max_relevant_to_consider: int = 100
            ):

        super().__init__(max_relevant_to_consider)

        if not os.path.exists(queries_file_path) \
                or not os.path.exists(corpus_elements_file_path) \
                or not os.path.exists(annotations_file_path):

            raise ValueError("Invalid evaluation dataset")

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        self.output_dir_path = output_dir_path

        self.q_ids, self.q_surface_forms = self._unpack_mapping(
            queries_file_path
        )
        self.c_ids, self.c_surface_forms = self._unpack_mapping(
            corpus_elements_file_path
        )

        self.queries_file_path = queries_file_path
        self.corpus_elements_file_path = corpus_elements_file_path
        self.annotations_file_path = annotations_file_path

    def evaluate(self, scorer: BaseScorer) -> List[Tuple[str, Any]]:
        scores = scorer.compute_scores(
            self.q_surface_forms,
            self.c_surface_forms
        )
        metrics_values = self._evaluate_scores_with_trec_binary(
            self.q_ids,
            self.c_ids,
            self.annotations_file_path,
            scores,
            self.output_dir_path
        )
        self._store_results(
            metrics_values,
            self.output_dir_path
        )
        return metrics_values


class BenchmarkEvaluator(BaseEvaluator):

    def __init__(
                self,
                datasets: List[MeloDatasetConfig],
                method_name: str,
                scorer: BiEncoderScorer,
                max_relevant_to_consider: int = 100
            ):

        super().__init__(max_relevant_to_consider)

        self.datasets = datasets
        self.method_name = method_name
        self.scorer = scorer

        self.dataset_elements = {}

        all_surface_forms = set()

        for dataset in self.datasets:
            base_data_path = melo_utils.get_data_processed_melo_dir_base_path()
            dataset_name = dataset.dataset_dir_name
            dataset_path = os.path.join(
                base_data_path,
                dataset_name
            )

            queries_file_path = os.path.join(
                dataset_path,
                "queries.tsv"
            )
            q_ids, q_surface_forms = self._unpack_mapping(
                queries_file_path
            )
            all_surface_forms.update(q_surface_forms)

            corpus_elements_file_path = os.path.join(
                dataset_path,
                "corpus_elements.tsv"
            )
            c_ids, c_surface_forms = self._unpack_mapping(
                corpus_elements_file_path
            )
            all_surface_forms.update(c_surface_forms)

            annotations_file_path = os.path.join(
                dataset_path,
                "annotations.tsv"
            )

            self.dataset_elements[dataset_name] = {
                "q_ids": q_ids,
                "q_surface_forms": q_surface_forms,
                "c_ids": c_ids,
                "c_surface_forms": c_surface_forms,
                "annotations_file_path": annotations_file_path
            }

        all_surface_forms = list(all_surface_forms)
        self.scorer.pre_compute_embeddings(all_surface_forms)

    def evaluate(self):
        for dataset in self.datasets:
            dataset_name = dataset.dataset_dir_name
            dataset_elements = self.dataset_elements[dataset_name]
            q_ids = dataset_elements["q_ids"]
            q_surface_forms = dataset_elements["q_surface_forms"]
            c_ids = dataset_elements["c_ids"]
            c_surface_forms = dataset_elements["c_surface_forms"]
            annotations_file_path = dataset_elements["annotations_file_path"]

            base_output_path = melo_utils.get_results_dir_path()
            method_name = melo_utils.simplify_method_name(self.method_name)
            output_path = os.path.join(
                base_output_path,
                method_name,
                dataset_name
            )

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            scores = self.scorer.compute_scores(
                q_surface_forms,
                c_surface_forms
            )

            metrics_values = self._evaluate_scores_with_trec_binary(
                q_ids,
                c_ids,
                annotations_file_path,
                scores,
                output_path
            )
            self._store_results(
                metrics_values,
                output_path
            )
