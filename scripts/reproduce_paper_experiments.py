import argparse
import os
from typing import List

from melo_benchmark.data_processing.official_dataset_helper import (
    MeloDatasetConfig,
    OfficialDatasetHelper
)
from melo_benchmark.evaluation.lexical_baseline.bm25_baseline import \
    BM25BaselineScorer
from melo_benchmark.evaluation.lexical_baseline.edit_distance_baseline import \
    EditDistanceBaselineScorer
from melo_benchmark.evaluation.lexical_baseline.tf_idf_baseline import (
    CharTfIdfBaselineScorer,
    WordTfIdfBaselineScorer
)
from melo_benchmark.evaluation.semantic_baseline.openai_biencoder import \
    OpenAiBiEncoderScorer
from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
    SentenceTransformersBiEncoderScorer
from melo_benchmark.evaluation.semantic_baseline.tfhub_biencoder import \
    TFHubBiEncoderScorer
from melo_benchmark.evaluation.evaluator import (
    BenchmarkEvaluator,
    Evaluator
)
from melo_benchmark.evaluation.scorer import BiEncoderScorer
import melo_benchmark.utils.helper as melo_utils
from melo_benchmark.utils.lemmatizer import Lemmatizer


LEXICAL_BASELINES = {
    "Edit Distance": EditDistanceBaselineScorer,
    "Word TF-IDF": WordTfIdfBaselineScorer,
    "Word TF-IDF (lemmas)": WordTfIdfBaselineScorer,
    "Char TF-IDF": CharTfIdfBaselineScorer,
    "Char TF-IDF (lemmas)": CharTfIdfBaselineScorer,
    "BM25": BM25BaselineScorer,
    "BM25 (lemmas)": BM25BaselineScorer,
}


prompt_template = "The candidate's job title is \"{{job_title}}\". " \
                  + "What skills are likely required for this job?"


def build_escoxlm_r_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="jjzha/esco-xlm-roberta-large",
        prompt_template=prompt_template
    )


def build_muse_cnn_scorer() -> BiEncoderScorer:
    return TFHubBiEncoderScorer(
        model_name="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
        prompt_template="{{job_title}}"
    )


def build_paraph_mmpnet_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        prompt_template=prompt_template
    )


def build_bge_m3_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="BAAI/bge-m3",
        prompt_template=prompt_template
    )


def build_gist_embedding_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="avsolatorio/GIST-Embedding-v0",
        prompt_template=prompt_template
    )


def build_me5_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="intfloat/multilingual-e5-large",
        prompt_template=prompt_template
    )


def build_e5_scorer() -> BiEncoderScorer:
    return SentenceTransformersBiEncoderScorer(
        model_name="intfloat/e5-mistral-7b-instruct",
        prompt_template=prompt_template
    )


def build_openai_scorer() -> BiEncoderScorer:
    return OpenAiBiEncoderScorer(
        model_name="text-embedding-3-large",
        prompt_template=prompt_template
    )


SEMANTIC_BASELINES = {
    "ESCOXLM-R": build_escoxlm_r_scorer,
    "mUSE-CNN": build_muse_cnn_scorer,
    "Paraph-mMPNet": build_paraph_mmpnet_scorer,
    "BGE-M3": build_bge_m3_scorer,
    "GIST-Embedding": build_gist_embedding_scorer,
    "mE5": build_me5_scorer,
    "E5": build_e5_scorer,
    "OpenAI": build_openai_scorer,
}


def evaluate_lexical_baseline(baseline_name: str, dataset: MeloDatasetConfig):
    dataset_name = dataset.dataset_name
    print(f"Evaluating baseline {baseline_name} on {dataset_name}...")

    scorer_class = LEXICAL_BASELINES[baseline_name]

    source_language = dataset.source_language
    scorer_params = {}
    if baseline_name != "Edit Distance":
        should_normalize_ascii = dataset.is_ascii_normalizable
        scorer_params["ascii_normalization"] = should_normalize_ascii
    if baseline_name.endswith(" (lemmas)"):
        if not dataset.has_spacy_lemmatizer:
            # Skip for those languages with no spaCy lemmatizer
            return
        lemmatizer = Lemmatizer(source_language)
        scorer_params["lemmatizer"] = lemmatizer
    scorer = scorer_class(**scorer_params)

    base_data_path = melo_utils.get_data_processed_melo_dir_base_path()
    dataset_name = dataset.dataset_dir_name
    dataset_path = os.path.join(
        base_data_path,
        dataset_name
    )

    base_output_path = melo_utils.get_results_dir_path()
    method_name = melo_utils.simplify_method_name(baseline_name)
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


def evaluate_semantic_baseline(
            baseline_name: str,
            datasets: List[MeloDatasetConfig]
        ):

    print(f"Evaluating baseline {baseline_name} on all datasets...")

    scorer_builder = SEMANTIC_BASELINES[baseline_name]
    scorer: BiEncoderScorer = scorer_builder()

    evaluator = BenchmarkEvaluator(
        datasets,
        baseline_name,
        scorer
    )

    evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser(
        description="Script for reproducing the experiments from the paper."
    )

    parser.add_argument(
        "--baselines",
        choices=[
            "lexical",
            "semantic",
            "both"
        ],
        default="both",
        help="Choose the baseline to run. Options: 'lexical', "
             + "'semantic', or 'both' (default: 'both')."
    )

    args = parser.parse_args()
    selected_baselines = args.baselines
    print(f"Selected baselines: {selected_baselines}")

    melo_dataset_helper = OfficialDatasetHelper()
    melo_datasets = melo_dataset_helper.get_dataset_configs()

    print("\n\n")

    # Lexical baselines
    for baseline_name in LEXICAL_BASELINES.keys():
        for dataset in melo_datasets:
            evaluate_lexical_baseline(baseline_name, dataset)

    # Semantic baselines
    for baseline_name in SEMANTIC_BASELINES.keys():
        evaluate_semantic_baseline(baseline_name, melo_datasets)


if __name__ == "__main__":
    main()
