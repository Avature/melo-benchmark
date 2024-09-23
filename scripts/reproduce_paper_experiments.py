from dataclasses import dataclass
import os
from typing import Set

from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.lexical_baseline.bm25_baseline import \
    BM25BaselineScorer
from melo_benchmark.evaluation.lexical_baseline.edit_distance_baseline import \
    EditDistanceBaselineScorer
from melo_benchmark.evaluation.lexical_baseline.tf_idf_baseline import (
    CharTfIdfBaselineScorer,
    WordTfIdfBaselineScorer
)
from melo_benchmark.evaluation.evaluator import Evaluator
from melo_benchmark.evaluation.scorer import BaseScorer
import melo_benchmark.utils.helper as melo_utils
from melo_benchmark.utils.lemmatizer import Lemmatizer


@dataclass
class MeloDatasetConfig:
    dataset_name: str
    crosswalk_name: str
    source_language: str
    target_languages: Set[str]
    dataset_dir_name: str = None


LEXICAL_BASELINES = {
    "Edit Distance": EditDistanceBaselineScorer,
    "Word TF-IDF": WordTfIdfBaselineScorer,
    "Word TF-IDF (lemmas)": WordTfIdfBaselineScorer,
    "Char TF-IDF": CharTfIdfBaselineScorer,
    "Char TF-IDF (lemmas)": CharTfIdfBaselineScorer,
    "BM25": BM25BaselineScorer,
    "BM25 (lemmas)": BM25BaselineScorer,
}


def evaluate_baseline(
            dataset: MeloDatasetConfig,
            method_name: str,
            scorer: BaseScorer
        ):

    base_data_path = melo_utils.get_data_processed_melo_dir_base_path()
    dataset_name = dataset.dataset_dir_name
    dataset_path = os.path.join(
        base_data_path,
        dataset_name
    )

    base_output_path = melo_utils.get_results_dir_path()
    method_name = melo_utils.simplify_method_name(method_name)
    output_path = os.path.join(
        base_output_path,
        method_name,
        dataset_name
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    queries_file_path = os.path.join(
        dataset_path,
        "queries.tsv"
    )
    corpus_elements_file_path = os.path.join(
        dataset_path,
        "corpus_elements.tsv"
    )
    annotations_file_path = os.path.join(
        dataset_path,
        "annotations.tsv"
    )

    evaluator = Evaluator(
        queries_file_path,
        corpus_elements_file_path,
        annotations_file_path,
        output_path
    )

    evaluator.evaluate(scorer=scorer)


def main():
    melo_dataset_helper = OfficialDatasetHelper()
    melo_datasets = melo_dataset_helper.get_dataset_configs()

    print("\n\n")

    # Lexical baselines
    for baseline_name, scorer_class in LEXICAL_BASELINES.items():
        for dataset in melo_datasets:
            dataset_name = dataset.dataset_name
            print(f"Evaluating baseline {baseline_name} on {dataset_name}...")
            source_language = dataset.source_language
            scorer_params = {}
            if baseline_name != "Edit Distance":
                should_normalize_ascii = True
                if dataset.source_language == "bg":
                    # Avoid ASCII normalization for Bulgarian
                    should_normalize_ascii = False
                scorer_params["ascii_normalization"] = should_normalize_ascii
            if baseline_name.endswith(" (lemmas)"):
                try:
                    lemmatizer = Lemmatizer(source_language)
                except NotImplementedError:
                    # Skip for those languages with no spaCy lemmatizer
                    continue
                scorer_params["lemmatizer"] = lemmatizer
            scorer = scorer_class(**scorer_params)
            evaluate_baseline(dataset, baseline_name, scorer)


if __name__ == "__main__":
    main()
