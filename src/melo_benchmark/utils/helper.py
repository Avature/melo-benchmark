"""
This script provides helper functions.
"""

import json
import os
from typing import (
    Any,
    Dict
)

from dotenv import load_dotenv

from melo_benchmark.utils.json_encoder import SetsAsListsEncoder


def serialize_as_json(content: Dict[str, Any]) -> str:
    return json.dumps(
        content,
        sort_keys=True,
        indent=4,
        separators=(",", ": "),
        ensure_ascii=False,
        cls=SetsAsListsEncoder
    )


def get_data_dir_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "..", "data")


def get_dataset_path(*args) -> str:
    return os.path.abspath(os.path.join(get_data_dir_path(), *args))


def get_data_processed_dir_base_path() -> str:
    return get_dataset_path("processed")


def get_data_processed_custom_dir_base_path() -> str:
    return get_dataset_path("processed", "custom")


def get_data_processed_melo_dir_base_path() -> str:
    return get_dataset_path("processed", "melo")


def get_data_raw_dir_base_path() -> str:
    return get_dataset_path("raw")


def get_data_raw_crosswalks_orig_dir_base_path() -> str:
    return get_dataset_path("raw", "crosswalks_original")


def get_data_raw_crosswalks_std_dir_base_path() -> str:
    return get_dataset_path("raw", "crosswalks_standard")


def get_data_raw_esco_original_dir_base_path() -> str:
    return get_dataset_path("raw", "esco_original")


def get_esco_original_dataset_path(esco_version: str, language: str) -> str:
    return get_dataset_path(
        "raw",
        "esco_original",
        esco_version,
        language
    )


def get_data_raw_esco_standard_dir_base_path() -> str:
    return get_dataset_path("raw", "esco_standard")


def get_reports_dir_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "..", "reports")


def get_resources_dir_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "..", "resources")


def get_results_dir_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "..", "results")


def load_content_from_json_file(file_path: str) -> Dict:
    with open(file_path, encoding="utf-8") as f_in:
        return json.load(f_in)


def simplify_method_name(method_name: str) -> str:
    method_name = method_name.lower()
    method_name = method_name.replace(" ", "_")
    method_name = method_name.replace("-", "_")
    method_name = method_name.replace("(", "")
    method_name = method_name.replace(")", "")

    return method_name


def load_dotenv_variables():
    dotenv_path = os.path.join(
        os.path.realpath(__file__),
        "..",
        "..",
        "..",
        "..",
        ".env"
    )
    dotenv_path = os.path.abspath(dotenv_path)
    load_dotenv(dotenv_path=dotenv_path)
