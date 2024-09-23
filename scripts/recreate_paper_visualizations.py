import os

from melo_benchmark.analysis.accuracy_at_k_visualizer import \
    AccuracyAtKVisualizer
from melo_benchmark.analysis.correlation_visualizer import \
    CorrelationVisualizer
from melo_benchmark.analysis.histogram_visualizer import HistogramVisualizer
from melo_benchmark.analysis.latex_table_helper import LatexTableHelper
from melo_benchmark.evaluation.metrics import TrecMetric
import melo_benchmark.utils.helper as melo_utils


LEXICAL_BASELINES = [
    "Edit Distance",
    "Word TF-IDF",
    "Word TF-IDF (lemmas)",
    "Char TF-IDF",
    "Char TF-IDF (lemmas)",
    "BM25",
    "BM25 (lemmas)",
]

SEMANTIC_BASELINES = [
    "ESCOXLM-R",
    "mUSE-CNN",
    "Paraph-mMPNet",
    "BGE-M3",
    "GIST-Embedding",
    "mE5",
    "E5",
    "OpenAI",
]


def main():
    reports_base_dir_path = melo_utils.get_reports_dir_path()

    # - - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #
    #    Figures
    #
    # - - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    # Figure 1
    crosswalks = [
        "usa_en",
        "deu_de",
        "esp_es",
        "nld_nl",
        "dnk_da",
    ]
    histogram_visualizer = HistogramVisualizer(crosswalks)
    histogram_figure_path = os.path.join(
        reports_base_dir_path,
        "figure1.pdf"
    )
    histogram_visualizer.create_histogram(histogram_figure_path)

    # Figure 2
    visualizer = CorrelationVisualizer()

    # Figure 2a
    methods = [
        "Edit Distance",
        "Char TF-IDF",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure2a.pdf"
    )
    visualizer.create_scatterplot(
        methods,
        visualization_file_path
    )

    # Figure 2b
    methods = [
        "mUSE-CNN",
        "OpenAI",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure2b.pdf"
    )
    baseline_method = "Char TF-IDF"
    visualizer.create_scatterplot_deltas(
        methods,
        baseline_method,
        visualization_file_path
    )

    # Figure 3
    acc_at_k_visualizer = AccuracyAtKVisualizer()
    crosswalks = [
        "usa_en",
        "deu_de",
        "esp_es",
        "nld_nl",
        "dnk_da",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure3.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # Figure 4
    histogram_visualizer = HistogramVisualizer()
    histogram_figure_path = os.path.join(
        reports_base_dir_path,
        "figure4.pdf"
    )
    histogram_visualizer.create_histogram(histogram_figure_path)

    # Figure 5

    # Figure 5a
    methods = [
        "Edit Distance",
        "Char TF-IDF",
        "mUSE-CNN",
        "OpenAI",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure5a.pdf"
    )
    visualizer.create_scatterplot(
        methods,
        visualization_file_path
    )

    # Figure 5b
    methods = [
        "mUSE-CNN",
        "OpenAI",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure5b.pdf"
    )
    baseline_method = "Char TF-IDF"
    visualizer.create_scatterplot_deltas(
        methods,
        baseline_method,
        visualization_file_path
    )

    # Figure 6

    # Figure 6a
    crosswalks = [
        "usa_en",
        "aut_de",
        "bel_fr",
        "bel_nl",
        "bgr_bg",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure6a.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # Figure 6b
    crosswalks = [
        "cze_cs",
        "deu_de",
        "dnk_da",
        "esp_es",
        "est_et",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure6b.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # Figure 6c
    crosswalks = [
        "fra_fr",
        "hrv_hr",
        "hun_hu",
        "ita_it",
        "ltu_lt",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure6c.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # Figure 7

    # Figure 7a
    crosswalks = [
        "lva_lv",
        "nld_nl",
        "nor_no",
        "pol_pl",
        "prt_pt",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure7a.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # Figure 7b
    crosswalks = [
        "rou_ro",
        "svk_sk",
        "svn_sl",
        "swe_sv",
    ]
    methods = [
        "OpenAI",
        "mUSE-CNN",
        "Char TF-IDF",
        "Edit Distance",
    ]
    visualization_file_path = os.path.join(
        reports_base_dir_path,
        "figure7b.pdf"
    )
    acc_at_k_visualizer.create_plot(
        methods,
        crosswalks,
        visualization_file_path
    )

    # - - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #
    #    Tables
    #
    # - - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    table_helper = LatexTableHelper()

    # Table 2
    dataset_names = [
        "usa_q_en_c_en",
        "usa_q_en_c_de_en_es_fr_it_nl_pl_pt",
        "deu_q_de_c_de",
        "deu_q_de_c_en",
        "esp_q_es_c_es",
        "esp_q_es_c_en",
        "nld_q_nl_c_nl",
        "nld_q_nl_c_en",
        "dnk_q_da_c_da",
        "dnk_q_da_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 2 - Lexical Baselines
    table_section_name = "table_2_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 2 - Semantic Baselines
    table_section_name = "table_2_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5a
    dataset_names = [
        "usa_q_en_c_en",
        "usa_q_en_c_de_en_es_fr_it_nl_pl_pt",
        "aut_q_de_c_de",
        "aut_q_de_c_en",
        "bel_q_fr_c_fr",
        "bel_q_fr_c_en",
        "bel_q_nl_c_nl",
        "bel_q_nl_c_en",
        "bgr_q_bg_c_bg",
        "bgr_q_bg_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 5a - Lexical Baselines
    table_section_name = "table_5a_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5a - Semantic Baselines
    table_section_name = "table_5a_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5b
    dataset_names = [
        "cze_q_cs_c_cs",
        "cze_q_cs_c_en",
        "deu_q_de_c_de",
        "deu_q_de_c_en",
        "dnk_q_da_c_da",
        "dnk_q_da_c_en",
        "esp_q_es_c_es",
        "esp_q_es_c_en",
        "est_q_et_c_et",
        "est_q_et_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 5b - Lexical Baselines
    table_section_name = "table_5b_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5b - Semantic Baselines
    table_section_name = "table_5b_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5c
    dataset_names = [
        "fra_q_fr_c_fr",
        "fra_q_fr_c_en",
        "hrv_q_hr_c_hr",
        "hrv_q_hr_c_en",
        "hun_q_hu_c_hu",
        "hun_q_hu_c_en",
        "ita_q_it_c_it",
        "ita_q_it_c_en",
        "ltu_q_lt_c_lt",
        "ltu_q_lt_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 5c - Lexical Baselines
    table_section_name = "table_5c_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 5c - Semantic Baselines
    table_section_name = "table_5c_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 6a
    dataset_names = [
        "lva_q_lv_c_lv",
        "lva_q_lv_c_en",
        "nld_q_nl_c_nl",
        "nld_q_nl_c_en",
        "nor_q_no_c_no",
        "nor_q_no_c_en",
        "pol_q_pl_c_pl",
        "pol_q_pl_c_en",
        "prt_q_pt_c_pt",
        "prt_q_pt_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 6a - Lexical Baselines
    table_section_name = "table_6a_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 6a - Semantic Baselines
    table_section_name = "table_6a_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 6b
    dataset_names = [
        "rou_q_ro_c_ro",
        "rou_q_ro_c_en",
        "svk_q_sk_c_sk",
        "svk_q_sk_c_en",
        "svn_q_sl_c_sl",
        "svn_q_sl_c_en",
        "swe_q_sv_c_sv",
        "swe_q_sv_c_en",
    ]
    trec_metric = TrecMetric.RECIP_RANK

    # Table 6b - Lexical Baselines
    table_section_name = "table_6b_lexical"
    table_helper.create_table_section(
        LEXICAL_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )

    # Table 6b - Semantic Baselines
    table_section_name = "table_6b_semantic"
    table_helper.create_table_section(
        SEMANTIC_BASELINES,
        dataset_names,
        trec_metric,
        table_section_name
    )


if __name__ == "__main__":
    main()
