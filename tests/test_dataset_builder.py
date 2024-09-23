import unittest

from melo_benchmark.data_processing.dataset_builder import DatasetBuilder


# noinspection DuplicatedCode
class TestDatasetBuilder(unittest.TestCase):

    def test_build_usa_en_xx(self):
        source_taxonomy = "usa_en"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_usa_en_all(self):
        source_taxonomy = "usa_en"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_aut_de_xx(self):
        source_taxonomy = "aut_de"
        target_languages = ["de"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_aut_de_en(self):
        source_taxonomy = "aut_de"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bel_fr_xx(self):
        source_taxonomy = "bel_fr"
        target_languages = ["fr"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bel_fr_en(self):
        source_taxonomy = "bel_fr"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bel_nl_xx(self):
        source_taxonomy = "bel_nl"
        target_languages = ["nl"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bel_nl_en(self):
        source_taxonomy = "bel_nl"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bgr_bg_xx(self):
        source_taxonomy = "bgr_bg"
        target_languages = ["bg"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bgr_bg_en(self):
        source_taxonomy = "bgr_bg"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_cze_cs_xx(self):
        source_taxonomy = "cze_cs"
        target_languages = ["cs"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_cze_cs_en(self):
        source_taxonomy = "cze_cs"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_deu_de_xx(self):
        source_taxonomy = "deu_de"
        target_languages = ["de"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_deu_de_en(self):
        source_taxonomy = "deu_de"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_dnk_da_xx(self):
        source_taxonomy = "dnk_da"
        target_languages = ["da"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_dnk_da_en(self):
        source_taxonomy = "dnk_da"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_esp_es_xx(self):
        source_taxonomy = "esp_es"
        target_languages = ["es"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_esp_es_en(self):
        source_taxonomy = "esp_es"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_est_et_xx(self):
        source_taxonomy = "est_et"
        target_languages = ["et"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_est_et_en(self):
        source_taxonomy = "est_et"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_fra_fr_xx(self):
        source_taxonomy = "fra_fr"
        target_languages = ["fr"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_fra_fr_en(self):
        source_taxonomy = "fra_fr"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_hrv_hr_xx(self):
        source_taxonomy = "hrv_hr"
        target_languages = ["hr"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_hrv_hr_en(self):
        source_taxonomy = "hrv_hr"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_hun_hu_xx(self):
        source_taxonomy = "hun_hu"
        target_languages = ["hu"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_hun_hu_en(self):
        source_taxonomy = "hun_hu"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_ita_it_xx(self):
        source_taxonomy = "ita_it"
        target_languages = ["it"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_ita_it_en(self):
        source_taxonomy = "ita_it"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_ltu_lt_xx(self):
        source_taxonomy = "ltu_lt"
        target_languages = ["lt"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_ltu_lt_en(self):
        source_taxonomy = "ltu_lt"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_lva_lv_xx(self):
        source_taxonomy = "lva_lv"
        target_languages = ["lv"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_lva_lv_en(self):
        source_taxonomy = "lva_lv"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_nld_nl_xx(self):
        source_taxonomy = "nld_nl"
        target_languages = ["nl"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_nld_nl_en(self):
        source_taxonomy = "nld_nl"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_nor_no_xx(self):
        source_taxonomy = "nor_no"
        target_languages = ["no"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_nor_no_en(self):
        source_taxonomy = "nor_no"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_pol_pl_xx(self):
        source_taxonomy = "pol_pl"
        target_languages = ["pl"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_pol_pl_en(self):
        source_taxonomy = "pol_pl"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_prt_pt_xx(self):
        source_taxonomy = "prt_pt"
        target_languages = ["pt"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_prt_pt_en(self):
        source_taxonomy = "prt_pt"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_rou_ro_xx(self):
        source_taxonomy = "rou_ro"
        target_languages = ["ro"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_rou_ro_en(self):
        source_taxonomy = "rou_ro"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_svk_sk_xx(self):
        source_taxonomy = "svk_sk"
        target_languages = ["sk"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_svk_sk_en(self):
        source_taxonomy = "svk_sk"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_svn_sl_xx(self):
        source_taxonomy = "svn_sl"
        target_languages = ["sl"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_svn_sl_en(self):
        source_taxonomy = "svn_sl"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_swe_sv_xx(self):
        source_taxonomy = "swe_sv"
        target_languages = ["sv"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_swe_sv_en(self):
        source_taxonomy = "swe_sv"
        target_languages = ["en"]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_esp_es_all(self):
        # Custom dataset
        source_taxonomy = "esp_es"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_aut_de_all(self):
        # Custom dataset
        source_taxonomy = "aut_de"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_bel_fr_all(self):
        # Custom dataset
        source_taxonomy = "bel_fr"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_pol_pl_all(self):
        source_taxonomy = "pol_pl"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_prt_pt_all(self):
        source_taxonomy = "prt_pt"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_ita_it_all(self):
        source_taxonomy = "ita_it"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)

    def test_build_nld_nl_all(self):
        source_taxonomy = "nld_nl"
        target_languages = [
            "de",
            "en",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt"
        ]

        dataset_builder = DatasetBuilder(source_taxonomy)

        dataset_builder.build(target_languages)
