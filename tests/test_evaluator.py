import os
import unittest

from melo_benchmark.evaluation.lexical_baseline.bm25_baseline import \
    BM25BaselineScorer
from melo_benchmark.evaluation.lexical_baseline.edit_distance_baseline import \
    EditDistanceBaselineScorer
from melo_benchmark.evaluation.lexical_baseline.random_baseline import \
    RandomBaselineScorer
from melo_benchmark.evaluation.lexical_baseline.tf_idf_baseline import (
    CharTfIdfBaselineScorer,
    WordTfIdfBaselineScorer
)
from melo_benchmark.evaluation.evaluator import Evaluator
import melo_benchmark.utils.helper as melo_utils
from melo_benchmark.utils.lemmatizer import Lemmatizer

import utils as test_utils


SHOULD_EVALUATE_LEXICAL_BASELINES = False


# noinspection DuplicatedCode
class TestEvaluator(unittest.TestCase):

    @staticmethod
    def _create_evaluator(output_path: str):
        base_data_path = melo_utils.get_data_processed_melo_dir_base_path()
        dataset_name = "usa_q_en_c_en"
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

    def test_random(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        scorer = RandomBaselineScorer()
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertLess(metric_value, 0.01)

    def test_bm25(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        scorer = BM25BaselineScorer()
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.2936)

    def test_bm25_lemma(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        lemmatizer = Lemmatizer("en")
        scorer = BM25BaselineScorer(
            lemmatizer=lemmatizer
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.6004)

    def test_edit_distance(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        scorer = EditDistanceBaselineScorer()
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.4858)

    def test_word_tf_idf(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        scorer = WordTfIdfBaselineScorer()
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.3250)

    def test_word_tf_idf_lemma(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        lemmatizer = Lemmatizer("en")
        scorer = WordTfIdfBaselineScorer(
            lemmatizer=lemmatizer
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.6056)

    def test_char_tf_idf(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        scorer = CharTfIdfBaselineScorer()
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.5800)

    def test_char_tf_idf_lemma(self):
        if not SHOULD_EVALUATE_LEXICAL_BASELINES:
            return

        output_path = test_utils.create_test_output_dir(self.id())
        evaluator = self._create_evaluator(output_path)
        lemmatizer = Lemmatizer("en")
        scorer = CharTfIdfBaselineScorer(
            lemmatizer=lemmatizer
        )
        result = evaluator.evaluate(scorer)

        for metric_name, metric_value in result:
            if metric_name == "recip_rank":
                metric_value = float(metric_value)
                self.assertEqual(metric_value, 0.5957)
