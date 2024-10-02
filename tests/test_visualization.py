import os
import unittest

from melo_benchmark.analysis.accuracy_at_k_visualizer import \
    AccuracyAtKVisualizer
from melo_benchmark.analysis.correlation_visualizer import \
    CorrelationVisualizer
from melo_benchmark.analysis.histogram_visualizer import HistogramVisualizer
from melo_benchmark.analysis.radar_visualizer import RadarVisualizer
from melo_benchmark.analysis.statistical_significance import \
    StatisticalSignificanceAnalyzer

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

    def test_radar_chart(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_a_file_path = os.path.join(
            output_path,
            "monolingual_mrr.png"
        )
        visualization_b_file_path = os.path.join(
            output_path,
            "cross_lingual_mrr.png"
        )

        crosswalks = [
            "deu_de",
            "esp_es",
            "fra_fr",
            "ita_it",
            "nld_nl",
            "pol_pl",
            "prt_pt",
            "swe_sv",
        ]
        methods = [
            "Char TF-IDF",
            "mUSE-CNN",
            "OpenAI",
        ]

        visualizer = RadarVisualizer()
        visualizer.create_radar_chart(
            methods,
            crosswalks,
            visualization_a_file_path,
            visualization_b_file_path
        )

    def test_radar_chart_full(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_a_file_path = os.path.join(
            output_path,
            "monolingual_mrr.png"
        )
        visualization_b_file_path = os.path.join(
            output_path,
            "cross_lingual_mrr.png"
        )

        crosswalks = [
            "deu_de",
            "esp_es",
            "fra_fr",
            "ita_it",
            "nld_nl",
            "pol_pl",
            "prt_pt",
            "swe_sv",
        ]
        methods = [
            "Char TF-IDF",
            "BGE-M3",
            "OpenAI",
        ]

        visualizer = RadarVisualizer()
        visualizer.create_radar_chart(
            methods,
            crosswalks,
            visualization_a_file_path,
            visualization_b_file_path
        )

    def test_correlation_plot_deltas_alt(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        visualization_file_path = os.path.join(
            output_path,
            "corr_plot.pdf"
        )

        methods = [
            "mUSE-CNN",
            "BGE-M3",
            "OpenAI",
        ]
        baseline_method = "Char TF-IDF"
        visualizer = CorrelationVisualizer()
        visualizer.create_scatterplot_deltas(
            methods,
            baseline_method,
            visualization_file_path
        )

    def test_wilcoxon(self):
        output_path = test_utils.create_test_output_dir(self.id())

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        report_file_path = os.path.join(
            output_path,
            "statistical_analysis.txt"
        )

        methods = [
            "mUSE-CNN",
            "BGE-M3",
            "OpenAI",
        ]

        analyzer = StatisticalSignificanceAnalyzer()
        analyzer.test_over_benchmark(
            methods,
            report_file_path
        )
