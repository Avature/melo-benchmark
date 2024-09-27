import os
import unittest

from melo_benchmark.evaluation.semantic_baseline.openai_biencoder import \
    OpenAiBiEncoderScorer
from melo_benchmark.evaluation.evaluator import Evaluator

import utils as test_utils


# noinspection DuplicatedCode
class TestSemanticBaselineEvaluator(unittest.TestCase):

    @staticmethod
    def _create_evaluator(output_path: str):
        base_data_path = test_utils.get_resources_folder_path()
        dataset_name = "mini_usa_q_en_c_en"
        dataset_path = os.path.join(
            base_data_path,
            dataset_name
        )

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

        return evaluator

    def test_openai_biencoder(self):
        output_path = test_utils.create_test_output_dir(self.id())
        representation_cache_path = os.path.join(
            output_path,
            "repr_cache.tsv"
        )

        evaluator = self._create_evaluator(output_path)

        model_name = "text-embedding-3-large"
        prompt_template = "The candidate's job title is \"{{job_title}}\". " \
                          + "What skills are likely required for this job?"
        scorer = OpenAiBiEncoderScorer(
            model_name=model_name,
            prompt_template=prompt_template,
            representation_cache_path=representation_cache_path
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertGreater(metric_value, 0.5)

    def test_sentence_transformers_biencoder(self):
        try:
            # noinspection PyUnresolvedReferences
            import torch
        except ImportError:
            # Ignore this test if PyTorch is not installed
            return

        print("\n\nTesting Sentence Transformer BiEncoder...\n\n")

        from melo_benchmark.evaluation.semantic_baseline \
            .stransf_biencoder import SentenceTransformersBiEncoderScorer

        output_path = test_utils.create_test_output_dir(self.id())
        representation_cache_path = os.path.join(
            output_path,
            "repr_cache.tsv"
        )

        evaluator = self._create_evaluator(output_path)

        model_name = "avsolatorio/GIST-Embedding-v0"
        prompt_template = "The candidate's job title is \"{{job_title}}\". " \
                          + "What skills are likely required for this job?"
        scorer = SentenceTransformersBiEncoderScorer(
            model_name=model_name,
            prompt_template=prompt_template,
            representation_cache_path=representation_cache_path
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertGreater(metric_value, 0.5)

    def test_tfhub_biencoder(self):
        try:
            # noinspection PyUnresolvedReferences
            import tensorflow as tf
        except ImportError:
            # Ignore this test if TensorFlow is not installed
            return

        print("\n\nTesting TF Hub BiEncoder...\n\n")

        from melo_benchmark.evaluation.semantic_baseline \
            .tfhub_biencoder import TFHubBiEncoderScorer

        output_path = test_utils.create_test_output_dir(self.id())
        representation_cache_path = os.path.join(
            output_path,
            "repr_cache.tsv"
        )

        evaluator = self._create_evaluator(output_path)

        model_name = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        scorer = TFHubBiEncoderScorer(
            model_name=model_name,
            prompt_template="{{job_title}}",
            representation_cache_path=representation_cache_path
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.5000)
            if metric_name == "recall_5":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.6667)
