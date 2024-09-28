import argparse
import logging
import os
from typing import (
    Any,
    Dict,
    List,
    Tuple
)

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
from melo_benchmark.evaluation.evaluator import (
    BenchmarkEvaluator,
    Evaluator
)
from melo_benchmark.evaluation.scorer import BiEncoderScorer
import melo_benchmark.utils.helper as melo_utils
from melo_benchmark.utils.lemmatizer import Lemmatizer
import melo_benchmark.utils.logging_config as melo_logging

melo_logging.setup_logging()
logger = logging.getLogger(__name__)

LEXICAL_BASELINES: Dict[str, Any] = {
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

SEMANTIC_BASELINES: Dict[str, Any] = {}


def build_escoxlm_r_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="jjzha/esco-xlm-roberta-large",
        prompt_template=prompt_template,
        batch_size=1024
    )


def build_muse_cnn_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.tfhub_biencoder import \
        TFHubBiEncoderScorer

    return TFHubBiEncoderScorer(
        model_name="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
        prompt_template="{{job_title}}"
    )


def build_paraph_mmpnet_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        prompt_template=prompt_template,
        batch_size=1024
    )


def build_bge_m3_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="BAAI/bge-m3",
        prompt_template=prompt_template,
        batch_size=1024
    )


def build_gist_embedding_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="avsolatorio/GIST-Embedding-v0",
        prompt_template=prompt_template,
        batch_size=1024
    )


def build_me5_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="intfloat/multilingual-e5-large",
        prompt_template=prompt_template
    )


def build_e5_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
        SentenceTransformersBiEncoderScorer

    return SentenceTransformersBiEncoderScorer(
        model_name="intfloat/e5-mistral-7b-instruct",
        prompt_template=prompt_template,
        batch_size=1
    )


def build_openai_scorer() -> BiEncoderScorer:
    # Avoid importing this if not needed
    from melo_benchmark.evaluation.semantic_baseline.openai_biencoder import \
        OpenAiBiEncoderScorer

    return OpenAiBiEncoderScorer(
        model_name="text-embedding-3-large",
        prompt_template=prompt_template
    )


def evaluate_lexical_baseline(baseline_name: str, dataset: MeloDatasetConfig):
    dataset_name = dataset.dataset_name
    logger.info(f"Evaluating baseline {baseline_name} on {dataset_name}...")

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
    logger.info(f"Evaluating baseline {baseline_name} on all datasets...")

    scorer_builder = SEMANTIC_BASELINES[baseline_name]
    scorer: BiEncoderScorer = scorer_builder()

    evaluator = BenchmarkEvaluator(
        datasets,
        baseline_name,
        scorer
    )

    evaluator.evaluate()


def check_dl_frameworks_and_gpu_availability() -> Tuple[bool, bool]:
    tf_gpu = False
    torch_gpu = False

    try:
        # noinspection PyUnresolvedReferences
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info("CUDA is available for TensorFlow.")
            logger.info(f"GPUs available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                logger.info(f"GPU {i} name: {gpu.name}")
            tf_gpu = True
        else:
            logger.warning("No GPUs are available for TensorFlow.")
    except ImportError:
        pass

    try:
        # noinspection PyUnresolvedReferences
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA is available for PyTorch.")
            logger.info(f"GPUs available: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            torch_gpu = True
        else:
            logger.warning("No GPUs are available for PyTorch.")
    except ImportError:
        pass

    return tf_gpu, torch_gpu


def register_semantic_baselines(tf_with_gpu, torch_with_gpu):
    if torch_with_gpu:
        SEMANTIC_BASELINES["ESCOXLM-R"] = build_escoxlm_r_scorer

    if tf_with_gpu:
        SEMANTIC_BASELINES["mUSE-CNN"] = build_muse_cnn_scorer

    if torch_with_gpu:
        SEMANTIC_BASELINES["Paraph-mMPNet"] = build_paraph_mmpnet_scorer
        SEMANTIC_BASELINES["BGE-M3"] = build_bge_m3_scorer
        SEMANTIC_BASELINES["GIST-Embedding"] = build_gist_embedding_scorer
        SEMANTIC_BASELINES["mE5"] = build_me5_scorer
        SEMANTIC_BASELINES["E5"] = build_e5_scorer

    SEMANTIC_BASELINES["OpenAI"] = build_openai_scorer


def get_and_log_execution_plan(selected_baselines: str):
    tf_gpu, torch_gpu = check_dl_frameworks_and_gpu_availability()

    # Execute locally-hosted DL models only if GPU is available for them
    register_semantic_baselines(tf_gpu, torch_gpu)

    print(f"\nSelected baselines: {selected_baselines}\n")

    lexical_baselines = list(LEXICAL_BASELINES.keys())
    semantic_baselines = list(SEMANTIC_BASELINES.keys())

    planned_baselines = []
    if selected_baselines in ["lexical", "all"]:
        planned_baselines += lexical_baselines
    if selected_baselines in ["semantic", "all"]:
        planned_baselines += semantic_baselines

    print("Execution Plan: The following baselines will be evaluated:")
    for baseline_name in planned_baselines:
        print(f" - {baseline_name}")
    print()

    return lexical_baselines, semantic_baselines


def main():
    parser = argparse.ArgumentParser(
        description="Script for reproducing the experiments from the paper."
    )

    parser.add_argument(
        "--baselines",
        choices=[
            "lexical",
            "semantic",
            "all"
        ],
        default="all",
        help="Choose the baseline to run. Options: 'lexical', "
             + "'semantic', or 'all' (default: 'all')."
    )

    args = parser.parse_args()
    selected_baselines = args.baselines

    lexical_baselines, semantic_baselines = get_and_log_execution_plan(
        selected_baselines
    )

    melo_dataset_helper = OfficialDatasetHelper()
    melo_datasets = melo_dataset_helper.get_dataset_configs()

    if selected_baselines in ["lexical", "all"]:
        # Lexical baselines
        for baseline_name in lexical_baselines:
            print(f"\nEvaluating lexical baseline: {baseline_name}\n")
            for dataset in melo_datasets:
                evaluate_lexical_baseline(baseline_name, dataset)

    if selected_baselines in ["semantic", "all"]:
        # Semantic baselines
        for baseline_name in semantic_baselines:
            print(f"\nEvaluating semantic baseline: {baseline_name}\n")
            evaluate_semantic_baseline(baseline_name, melo_datasets)


if __name__ == "__main__":
    main()
