import os
import unittest

from melo_benchmark.analysis.accuracy_at_k_visualizer import \
    AccuracyAtKVisualizer
from melo_benchmark.analysis.correlation_visualizer import \
    CorrelationVisualizer
from melo_benchmark.analysis.histogram_visualizer import HistogramVisualizer

import utils as test_utils


# noinspection DuplicatedCode
class TestVisualization(unittest.TestCase):

    def test_histograms(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "histogram.pdf"
        )

        visualizer = HistogramVisualizer()
        visualizer.create_histogram(visualization_file_path)

    def test_histograms_simple(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "histogram.pdf"
        )

        crosswalks = [
            "usa_en",
            "deu_de",
            "esp_es",
            "nld_nl",
            "dnk_da",
        ]
        visualizer = HistogramVisualizer(crosswalks)
        visualizer.create_histogram(visualization_file_path)

    def test_correlation_plot_simple_big(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "corr_plot.pdf"
        )

        methods = [
            "Edit Distance",
            "Word TF-IDF",
            "Char TF-IDF",
            "BM25",
        ]
        visualizer = CorrelationVisualizer()
        visualizer.create_scatterplot(
            methods,
            visualization_file_path
        )

    def test_correlation_plot_simple_small(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "corr_plot.pdf"
        )

        methods = [
            "Edit Distance",
            "Word TF-IDF",
        ]
        visualizer = CorrelationVisualizer()
        visualizer.create_scatterplot(
            methods,
            visualization_file_path
        )

    def test_correlation_plot_deltas_small(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "corr_plot.pdf"
        )

        methods = [
            "mUSE-CNN",
            "Char TF-IDF",
        ]
        baseline_method = "Edit Distance"
        visualizer = CorrelationVisualizer()
        visualizer.create_scatterplot_deltas(
            methods,
            baseline_method,
            visualization_file_path
        )

    def test_correlation_table(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "table.txt"
        )

        methods = [
            "Edit Distance",
            "Word TF-IDF",
            "Char TF-IDF",
            "BM25",
            "mUSE-CNN",
        ]
        visualizer = CorrelationVisualizer()
        visualizer.create_correlation_table(
            methods,
            visualization_file_path
        )

    def test_acc_at_k_plot(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "acc_at_k.pdf"
        )

        crosswalks = [
            "usa_en",
            "deu_de",
            "esp_es",
            "nld_nl",
            "dnk_da",
        ]
        methods = [
            "Edit Distance",
            "Word TF-IDF",
            "Char TF-IDF",
            "BM25",
        ]
        visualizer = AccuracyAtKVisualizer()
        visualizer.create_plot(
            methods,
            crosswalks,
            visualization_file_path
        )
