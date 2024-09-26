from __future__ import annotations

from dataclasses import dataclass
import os
from typing import (
    Any,
    Dict
)

import numpy as np
import pandas as pd

import melo_benchmark.utils.helper as melo_utils
from melo_benchmark.utils.metaclasses import Singleton


@dataclass
class CrosswalkFileConfig:
    file_name: str
    group_name: str = "european_vs_esco"
    num_header_rows: int = 16
    num_footer_rows: int = 0
    type_of_match_col_name: str = "Mapping relation"
    query_id_col_name: str = "Classification 2 ID"
    query_surface_form_col_name: str = "Classification 2 PrefLabel"
    query_desc_col_name: str = None
    esco_relevant_id_col_name: str = "Classification 1 URI"
    should_clean_csv_file: bool = False


SUPPORTED_CROSSWALKS = {
    "usa_en": CrosswalkFileConfig(
        file_name="crosswalk_esco_onet.csv",
        group_name="onet_vs_esco",
        type_of_match_col_name="Type of Match",
        query_id_col_name="O*NET Id",
        query_surface_form_col_name="O*NET Title",
        query_desc_col_name="O*NET Description",
        esco_relevant_id_col_name="ESCO or ISCO URI"
    ),
    "aut_de": CrosswalkFileConfig(
        file_name="AT_mapping.csv",
        num_footer_rows=59
    ),
    "bel_fr": CrosswalkFileConfig(
        file_name="BLM_mapping.csv",
        query_surface_form_col_name="Classification 2 PrefLabel FR",
        esco_relevant_id_col_name="Classification 1 URL"
    ),
    "bel_nl": CrosswalkFileConfig(
        file_name="BLM_mapping.csv",
        query_surface_form_col_name="Classification 2 PrefLabel NL",
        esco_relevant_id_col_name="Classification 1 URL"
    ),
    "bgr_bg": CrosswalkFileConfig(
        file_name="BG_mapping_23.08.2021.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "cze_cs": CrosswalkFileConfig(
        file_name="CZ_mapping_25.11.2022.csv"
    ),
    "deu_de": CrosswalkFileConfig(
        file_name="DE_mapping_18.05.22.csv"
    ),
    "dnk_da": CrosswalkFileConfig(
        file_name="DK_mapping_10.11.2021.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "esp_es": CrosswalkFileConfig(
        file_name="ES_mapping_31.07.2021.csv"
    ),
    "est_et": CrosswalkFileConfig(
        file_name="EE_mapping_07.10.2021.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "fra_fr": CrosswalkFileConfig(
        file_name="FR_20230607_mapping_esco_rome.csv",
        num_header_rows=15,
        type_of_match_col_name="mapping relation",
        query_id_col_name="classification 2 ID",
        query_surface_form_col_name="classification 2 prefered label",
        esco_relevant_id_col_name="classification 1 ID"
    ),
    "hrv_hr": CrosswalkFileConfig(
        file_name="HR_mapping_29.06.2021.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "hun_hu": CrosswalkFileConfig(
        file_name="HU_mapping_17.01.2022.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "ita_it": CrosswalkFileConfig(
        file_name="IT_mapping_23.04.2021.csv"
    ),
    "ltu_lt": CrosswalkFileConfig(
        file_name="LT_mapping_table_updated_20240222.csv"
    ),
    "lva_lv": CrosswalkFileConfig(
        file_name="LV_mapping_30.03.2021.csv",
        should_clean_csv_file=True
    ),
    "nld_nl": CrosswalkFileConfig(
        file_name="NL_mapping.csv"
    ),
    "nor_no": CrosswalkFileConfig(
        file_name="NO_mapping_20.09.2022..csv",
        query_id_col_name="Classification 2 URI"
    ),
    "pol_pl": CrosswalkFileConfig(
        file_name="PL_mapping_31.10.2020.csv"
    ),
    "prt_pt": CrosswalkFileConfig(
        file_name="PT_mapping_20.08.2021.csv"
    ),
    "rou_ro": CrosswalkFileConfig(
        file_name="RO_mapping_26.11.2021.csv",
        query_id_col_name="Classification 2 URI"
    ),
    "svk_sk": CrosswalkFileConfig(
        file_name="SK_mapping_05.05.2022.csv"
    ),
    "svn_sl": CrosswalkFileConfig(
        file_name="SI_mapping_2021.csv"
    ),
    "swe_sv": CrosswalkFileConfig(
        file_name="SE_mapping Occupations v1.1.1.csv",
        num_header_rows=17,
        query_id_col_name="Classification 2 URI"
    )
}


SUPPORTED_RELATION_TYPES = [
    "exactISCO",
    "exactMatch",
    "narrowMatch",
    "broadMatch",
    "closeMatch"
]


class CrosswalkLoader(metaclass=Singleton):
    """
    A class to load and process crosswalk files mapping a national
    terminology into ESCO.
    """

    def __init__(self):
        self.loaded_crosswalks: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def load(self, crosswalk_name: str) -> Dict[str, Dict[str, Any]]:
        if crosswalk_name in self.loaded_crosswalks:
            return self.loaded_crosswalks[crosswalk_name]

        target_crosswalk = self._do_load(crosswalk_name)
        self.loaded_crosswalks[crosswalk_name] = target_crosswalk

        return target_crosswalk

    def _do_load(self, crosswalk_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Loads the crosswalk data. If a JSON file for the crosswalk
        exists, it loads the data from the file. Otherwise, it
        processes the crosswalk data and saves it to a JSON file.

        Returns:
            dict: A dictionary containing the processed crosswalk data.
        """

        assert crosswalk_name in SUPPORTED_CROSSWALKS.keys()
        crosswalk_info = SUPPORTED_CROSSWALKS[crosswalk_name]

        # Define the path for the JSON file
        json_file_path = self._get_json_file_path(crosswalk_name)

        # Check if the JSON file already exists
        if os.path.exists(json_file_path):
            print(f"Loading data from {json_file_path}...")
            queries = melo_utils.load_content_from_json_file(json_file_path)
        else:
            # Load and process crosswalk
            df = self._load_orig_crosswalk_file(
                crosswalk_info
            )
            queries = self._process_crosswalk(
                df,
                crosswalk_info
            )

            # Format the concepts dictionary as a JSON string
            json_string = melo_utils.serialize_as_json(queries)

            # Save the JSON string to a file
            with open(json_file_path, 'w') as file:
                file.write(json_string)

            print(f"Crosswalk saved to {json_file_path}...")

        return queries

    def _process_crosswalk(
                self,
                df: pd.DataFrame,
                crosswalk_info: CrosswalkFileConfig
            ) -> Dict[str, Any]:

        queries = {}

        for _, row in df.iterrows():
            type_of_match = row[crosswalk_info.type_of_match_col_name]
            type_of_match = type_of_match.strip()
            type_of_match = type_of_match.removeprefix("skos:")

            if type_of_match == "no relation":
                # Some European terminologies publish occupations with no
                #    corresponding ESCO concepts
                continue

            assert type_of_match in SUPPORTED_RELATION_TYPES

            query_id = row[crosswalk_info.query_id_col_name]
            query_id = query_id.strip()
            query_title = row[crosswalk_info.query_surface_form_col_name]

            if not query_title:
                # Some European terminologies publish relations items
                #    with occupation ID but no occupation title (e.g. Norway)
                continue

            query_title = query_title.strip()
            query_desc = ""
            if crosswalk_info.query_desc_col_name:
                # Include query description if available in the crosswalk
                query_desc = row[crosswalk_info.query_desc_col_name]
            query_desc = query_desc.strip()
            relevant_id = row[crosswalk_info.esco_relevant_id_col_name]
            if not relevant_id:
                print(row)
            relevant_id = relevant_id.strip()

            if query_id not in queries.keys():
                queries[query_id] = {
                    "title": query_title,
                    "description": query_desc.strip(),
                    "matches": {}
                }
            else:
                if query_title != queries[query_id]["title"]:
                    print(f"Warning: different titles for query {query_id}")
                assert query_desc == queries[query_id]["description"]

            if relevant_id in queries[query_id]["matches"].keys():
                old_type_of_match = queries[query_id]["matches"][relevant_id]
                assert type_of_match == old_type_of_match
            else:
                queries[query_id]["matches"][relevant_id] = type_of_match

        return queries

    @staticmethod
    def _get_json_file_path(crosswalk_name: str) -> str:
        ds_base_dir = melo_utils.get_data_raw_crosswalks_std_dir_base_path()

        file_dir = os.path.join(
            ds_base_dir,
            crosswalk_name
        )
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_name = f"crosswalk.json"
        json_file_path = os.path.join(
            file_dir,
            file_name
        )

        return json_file_path

    @staticmethod
    def _clean_csv_file(orig_file: str, clean_file: str):
        with open(orig_file, "r") as f_in, open(clean_file, "w") as f_out:
            for line in f_in:
                clean_line = line.strip()

                if len(clean_line) > 2 and \
                        clean_line[0] == '"' and \
                        clean_line[-1] == '"':
                    clean_line = clean_line[1:-1]
                    clean_line = clean_line.replace('""', '"')

                f_out.write(clean_line + "\n")

    def _load_orig_crosswalk_file(
                self,
                crosswalk_info: CrosswalkFileConfig
            ) -> pd.DataFrame:

        base_dir_path = melo_utils.get_data_raw_crosswalks_orig_dir_base_path()
        crosswalk_file_path = os.path.join(
            base_dir_path,
            crosswalk_info.group_name,
            crosswalk_info.file_name
        )

        if crosswalk_info.should_clean_csv_file:
            # This is needed for the Latvian national terminology file
            clean_crosswalk_file_path = crosswalk_file_path[:-4] + ".CLEAN.csv"
            if os.path.exists(clean_crosswalk_file_path):
                # Use the already existent clean version
                crosswalk_file_path = clean_crosswalk_file_path
            else:
                # Clean the CSV file and use it instead of the original
                self._clean_csv_file(
                    orig_file=crosswalk_file_path,
                    clean_file=clean_crosswalk_file_path
                )
                crosswalk_file_path = clean_crosswalk_file_path

        df = pd.read_csv(
            crosswalk_file_path,
            skiprows=crosswalk_info.num_header_rows,
            skipfooter=crosswalk_info.num_footer_rows,
            encoding="utf_8",
            dtype=str
        )
        df = df.replace({np.nan: None})

        return df
