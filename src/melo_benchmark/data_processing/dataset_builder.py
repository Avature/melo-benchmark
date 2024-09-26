from dataclasses import dataclass
import json
import os
from typing import (
    Dict,
    List,
    Set
)
import statistics

from melo_benchmark.data_processing.crosswalk_loader import CrosswalkLoader
from melo_benchmark.data_processing.esco_loader import (
    EscoLoader,
    SUPPORTED_ESCO_LANGUAGES
)
import melo_benchmark.utils.helper as melo_utils


@dataclass
class CrosswalkConfig:
    query_language: str
    corresponding_esco_version: str
    min_acceptable_priority: int = 3
    melo_standard_output_language_sets: List[Set[str]] = None


SUPPORTED_CROSSWALKS = {
    "usa_en": CrosswalkConfig(
        query_language="en",
        corresponding_esco_version="1.1.0",
        melo_standard_output_language_sets=[
            {
                "de",
                "en",
                "es",
                "fr",
                "it",
                "nl",
                "pl",
                "pt"
            }
        ]
    ),
    "aut_de": CrosswalkConfig(
        query_language="de",
        corresponding_esco_version="1.1.0"
    ),
    "bel_fr": CrosswalkConfig(
        query_language="fr",
        corresponding_esco_version="1.0.3"
    ),
    "bel_nl": CrosswalkConfig(
        query_language="nl",
        corresponding_esco_version="1.0.3"
    ),
    "bgr_bg": CrosswalkConfig(
        query_language="bg",
        corresponding_esco_version="1.0.3"
    ),
    "cze_cs": CrosswalkConfig(
        query_language="cs",
        corresponding_esco_version="1.0.9"
    ),
    "deu_de": CrosswalkConfig(
        query_language="de",
        corresponding_esco_version="1.0.3"
    ),
    "dnk_da": CrosswalkConfig(
        query_language="da",
        corresponding_esco_version="1.0.8"
    ),
    "esp_es": CrosswalkConfig(
        query_language="es",
        corresponding_esco_version="1.0.8"
    ),
    "est_et": CrosswalkConfig(
        query_language="et",
        corresponding_esco_version="1.0.8"
    ),
    "fra_fr": CrosswalkConfig(
        query_language="fr",
        corresponding_esco_version="1.0.9"
    ),
    "hrv_hr": CrosswalkConfig(
        query_language="hr",
        corresponding_esco_version="1.0.3"
    ),
    "hun_hu": CrosswalkConfig(
        query_language="hu",
        corresponding_esco_version="1.0.8"
    ),
    "ita_it": CrosswalkConfig(
        query_language="it",
        corresponding_esco_version="1.0.8"
    ),
    "ltu_lt": CrosswalkConfig(
        query_language="lt",
        corresponding_esco_version="1.0.8"
    ),
    "lva_lv": CrosswalkConfig(
        query_language="lv",
        corresponding_esco_version="1.0.8"
    ),
    "nld_nl": CrosswalkConfig(
        query_language="nl",
        corresponding_esco_version="1.0.3"
    ),
    "nor_no": CrosswalkConfig(
        query_language="no",
        corresponding_esco_version="1.0.8",
        min_acceptable_priority=5
    ),
    "pol_pl": CrosswalkConfig(
        query_language="pl",
        corresponding_esco_version="1.0.3"
    ),
    "prt_pt": CrosswalkConfig(
        query_language="pt",
        corresponding_esco_version="1.0.3"
    ),
    "rou_ro": CrosswalkConfig(
        query_language="ro",
        corresponding_esco_version="1.0.8"
    ),
    "svk_sk": CrosswalkConfig(
        query_language="sk",
        corresponding_esco_version="1.0.8"
    ),
    "svn_sl": CrosswalkConfig(
        query_language="sl",
        corresponding_esco_version="1.0.8"
    ),
    "swe_sv": CrosswalkConfig(
        query_language="sv",
        corresponding_esco_version="1.1.1"
    )
}


RELATION_TYPE_PRIORITY = {
    "exactISCO": 1,
    "exactMatch": 2,
    "narrowMatch": 3,
    "broadMatch": 4,
    "closeMatch": 5
}


class DatasetBuilder:

    FILE_NAME_LOGGED_STATS = "logged_stats.txt"
    FILE_NAME_QUERIES = "queries.tsv"
    FILE_NAME_CORPUS_ELEMENTS = "corpus_elements.tsv"
    FILE_NAME_ANNOTATIONS = "annotations.tsv"
    FILE_NAME_SURFACE_FORMS = "surface_forms.json"

    def __init__(self, crosswalk_name: str):
        self.crosswalk_name = crosswalk_name

        if crosswalk_name not in SUPPORTED_CROSSWALKS.keys():
            m = f"Crosswalk `{crosswalk_name}` not supported"
            raise ValueError(m)
        self.crosswalk_name = crosswalk_name
        self.crosswalk_info = SUPPORTED_CROSSWALKS[self.crosswalk_name]

        crosswalk_loader = CrosswalkLoader(crosswalk_name)
        self.query_annotations = crosswalk_loader.load()

        esco_version = self.crosswalk_info.corresponding_esco_version
        esco_loader = EscoLoader(esco_version=esco_version)
        self.corpus = esco_loader.load()

        self.logged_stats = {}

    def build(self, target_languages: List[str]):
        target_languages = set(target_languages)
        self._assert_valid_target_langs(target_languages)

        output_path = self._compute_output_path(target_languages)

        processed_query_annotations = self._process_annotations()
        self._create_dataset_files(
            processed_query_annotations,
            target_languages,
            output_path
        )

        self._log_stats(output_path)

        return output_path

    def _process_annotations(self) -> List[Dict[str, str]]:
        n_inc_queries = 0
        n_rej_queries_too_many = 0
        n_rej_queries_too_few = 0

        processed_query_annotations = []

        for query_id, query_info in self.query_annotations.items():
            query_surface_form = query_info["title"]
            matches = query_info["matches"]
            best_matches = self._decide_best_matches(matches)

            if len(best_matches) == 0:
                n_rej_queries_too_few += 1

                # Reject queries that have zero acceptable matches
                continue

            elif len(best_matches) > 1:
                n_rej_queries_too_many += 1

                # Reject ambiguous queries
                continue

            n_inc_queries += 1

            relevant_element = best_matches[0]

            if self.crosswalk_name == "swe_sv" and \
                    relevant_element not in self.corpus.keys():
                # Swedish crosswalk includes an invalid (old) ESCO ID
                invalid_uri = "a9e54177-a185-404d-b6e6-15663d31137e"
                assert relevant_element.endswith(invalid_uri)
                continue

            new_query_annotation = {
                "query_id": query_id,
                "query_surface_form": query_surface_form,
                "rel_corpus_element_id": relevant_element,
            }

            processed_query_annotations.append(new_query_annotation)

        self.logged_stats["n_inc_queries"] = n_inc_queries
        self.logged_stats["n_rej_queries_too_many"] = n_rej_queries_too_many
        self.logged_stats["n_rej_queries_too_few"] = n_rej_queries_too_few

        return processed_query_annotations

    def _create_dataset_files(
                self,
                query_annotations: List[Dict[str, str]],
                target_languages: Set[str],
                output_path: str
            ):

        q_id_key_mapping = {}
        c_id_key_mapping = {}
        jt_key_sf_mapping = {}

        query_file_path = os.path.join(
            output_path,
            self.FILE_NAME_QUERIES
        )

        # Create file with list of queries
        with open(query_file_path, "w") as f_out:
            for i, query_info in enumerate(query_annotations):
                query_id = query_info["query_id"]
                query_sf = query_info["query_surface_form"]

                q_key = "Q{:06d}".format(i+1)
                q_id_key_mapping[query_id] = q_key

                f_out.write(f"{q_key}\t{query_sf}\n")

                jt_key_sf_mapping[q_key] = query_sf

        corpus_file_path = os.path.join(
            output_path,
            self.FILE_NAME_CORPUS_ELEMENTS
        )

        # Create file with list of corpus elements
        with open(corpus_file_path, "w") as f_out:
            i = 0
            for esco_id, node_info in self.corpus.items():
                i += 1
                c_id_key_mapping[esco_id] = set()
                standard_names = node_info["std_name"]
                alternative_names = node_info["alt_names"]
                for c_lang in standard_names.keys():
                    if c_lang in target_languages:
                        c_key = "C{:06d}_".format(i) + c_lang + "_000"
                        c_id_key_mapping[esco_id].add(c_key)
                        c_surf_form = standard_names[c_lang]
                        f_out.write(f"{c_key}\t{c_surf_form}\n")
                        jt_key_sf_mapping[c_key] = c_surf_form
                for c_lang in alternative_names.keys():
                    # Process alternative names, if any
                    if c_lang in target_languages:
                        alt_names = alternative_names[c_lang]
                        for j, alt_name in enumerate(alt_names):
                            c_key = "C{:06d}_".format(i) \
                                    + c_lang + "_" + \
                                    "{:03d}".format(j + 1)
                            c_id_key_mapping[esco_id].add(c_key)
                            c_surf_form = alt_name
                            f_out.write(f"{c_key}\t{c_surf_form}\n")
                            jt_key_sf_mapping[c_key] = c_surf_form

        qrel_file_path = os.path.join(
            output_path,
            self.FILE_NAME_ANNOTATIONS
        )

        els_per_query = []
        # Create file with annotations, with TREC format
        with open(qrel_file_path, "w") as f_out:
            for query_info in query_annotations:
                query_id = query_info["query_id"]
                q_key = q_id_key_mapping[query_id]
                corpus_element_id = query_info["rel_corpus_element_id"]
                c_keys = c_id_key_mapping[corpus_element_id]
                for c_key in sorted(c_keys):
                    f_out.write(f"{q_key}\t0\t{c_key}\t1\n")
                els_per_query.append(len(c_keys))

        median_els_per_query = statistics.median(els_per_query)
        self.logged_stats["median_elements_per_query"] = median_els_per_query
        self.logged_stats["n_corpus_elements"] = sum(
            [len(x) for x in c_id_key_mapping.values()]
        )

        surface_form_list_file_path = os.path.join(
            output_path,
            self.FILE_NAME_SURFACE_FORMS
        )

        new_dataset_elements = []

        for jt_key, jt_value in jt_key_sf_mapping.items():
            new_ds_element = {
                "id": jt_key,
                "job_title": jt_value,
            }
            new_dataset_elements.append(new_ds_element)

        with open(surface_form_list_file_path, "w", encoding='utf-8') as f_out:
            new_dataset = {
                "data": new_dataset_elements
            }
            json.dump(new_dataset, f_out, ensure_ascii=False, indent=4)

    def _decide_best_matches(self, matches: Dict[str, str]) -> List[str]:
        best_priority = float('inf')
        best_match_type = None

        for match_type in matches.values():
            current_priority = RELATION_TYPE_PRIORITY[match_type]
            if current_priority < best_priority:
                best_priority = current_priority
                best_match_type = match_type

        if best_priority > self.crosswalk_info.min_acceptable_priority:
            # Reject matches if none is acceptable (e.g. only closeMatch)
            return []

        # Collect all ESCO IDs with the best match type
        best_ids = [
            esco_id for esco_id, match_type in matches.items()
            if match_type == best_match_type
        ]

        return best_ids

    def _log_stats(self, output_path: str):
        log_file_path = os.path.join(
            output_path,
            self.FILE_NAME_LOGGED_STATS
        )

        n_inc_queries = self.logged_stats["n_inc_queries"]
        n_rej_queries_too_many = self.logged_stats["n_rej_queries_too_many"]
        n_rej_queries_too_few = self.logged_stats["n_rej_queries_too_few"]

        n_corpus_elements = self.logged_stats["n_corpus_elements"]

        median_els_per_query = self.logged_stats["median_elements_per_query"]

        total_rejected = n_rej_queries_too_many + n_rej_queries_too_few

        content = "\n" + \
            "=============== Statistics Log ===============\n" + \
            "\n" + \
            "Processed queries\n" + \
            f"    Accepted:                   {n_inc_queries}\n" + \
            f"    Total Rejected:             {total_rejected}\n" + \
            f"     - Rejected (ambiguous):    {n_rej_queries_too_many}\n" + \
            f"     - Rejected (vague):        {n_rej_queries_too_few}\n" + \
            "\n" + \
            "----------------------------------------------\n" + \
            "\n" + \
            f"Corpus elements:                {n_corpus_elements}\n" + \
            "\n" + \
            "----------------------------------------------\n" + \
            "\n" + \
            f"Median relevant per query:      {median_els_per_query}\n" + \
            "\n" + \
            "==============================================\n"

        with open(log_file_path, 'w') as log_file:
            log_file.write(content)

    @staticmethod
    def _assert_valid_target_langs(target_languages: Set[str]):
        for target_language in target_languages:
            if target_language not in SUPPORTED_ESCO_LANGUAGES:
                m = f"Target language `{target_language}` not supported"
                raise ValueError(m)

    def compute_dataset_name(self, target_languages: Set[str]) -> str:
        crosswalk_name_parts = self.crosswalk_name.split("_")
        assert len(crosswalk_name_parts) == 2
        country, source_lang = crosswalk_name_parts

        target_langs = "_".join(sorted(target_languages))

        dataset_name = f"{country}_q_{source_lang}_c_{target_langs}"

        return dataset_name

    def _compute_output_path(self, target_languages: Set[str]) -> str:
        implicit_sets = [
            {"en"},
            {self.crosswalk_info.query_language}
        ]
        explicit_sets = self.crosswalk_info.melo_standard_output_language_sets
        if not explicit_sets:
            explicit_sets = []
        if target_languages in implicit_sets + explicit_sets:
            # Datasets used in the MELO paper are considered "standard"
            base_path = melo_utils.get_data_processed_melo_dir_base_path()
        else:
            # Other datasets are considered "custom"
            base_path = melo_utils.get_data_processed_custom_dir_base_path()

        output_path = os.path.join(
            base_path,
            self.compute_dataset_name(target_languages)
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path
