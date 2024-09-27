import logging
import os
import requests
import zipfile

import melo_benchmark.utils.helper as melo_utils
import melo_benchmark.utils.logging_config as melo_logging


melo_logging.setup_logging()
logger = logging.getLogger(__name__)


record_url = "https://zenodo.org/records/13830968"
data_urls = [
    f"{record_url}/files/melo_benchmark_raw.zip?download=1",
    f"{record_url}/files/melo_benchmark_processed.zip?download=1"
]


# Directory where the files will be downloaded and extracted
data_dir = melo_utils.get_data_dir_path()


def download_and_unzip(url: str, output_dir: str):
    zip_file_name = url.split('/')[-1].replace('?download=1', '')
    zip_file_path = os.path.join(output_dir, zip_file_name)

    # Download the file if it doesn't already exist
    if not os.path.exists(zip_file_path):
        logger.info(f"Downloading {zip_file_name}...")
        response = requests.get(url)

        # Raise an error for bad responses (e.g., 404)
        response.raise_for_status()

        with open(zip_file_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {zip_file_name}.")
    else:
        logger.info(f"{zip_file_name} already exists, skipping download.")

    dir_name = zip_file_name.replace('melo_benchmark_', '')
    dir_name = dir_name.replace('.zip', '')
    dir_path = os.path.join(output_dir, dir_name)

    # Unzip the file if it hasn't already been unzipped
    if not os.path.exists(dir_path):
        logger.info(f"Unzipping {zip_file_name}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"Unzipped {zip_file_name} to {dir_path}.")
    else:
        logger.info(f"{zip_file_name} already unzipped, skipping extraction.")


# Process each Zenodo URL
def main():
    for url in data_urls:
        download_and_unzip(url, data_dir)


if __name__ == "__main__":
    main()
