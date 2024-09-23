import os
import psutil
import shutil


def create_test_output_dir(dir_name: str) -> str:
    dir_path = os.path.join(get_output_folder_path(), dir_name)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return dir_path


def get_base_test_path() -> str:
    return os.path.abspath(os.path.join(os.path.realpath(__file__), ".."))


def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss / 1024 / 1024
    return mem


def get_output_folder_path() -> str:
    return os.path.join(get_base_test_path(), "output")


def get_resources_folder_path() -> str:
    return os.path.join(get_base_test_path(), "resources")
