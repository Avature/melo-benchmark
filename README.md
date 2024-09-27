# MELO Benchmark: Multilingual Entity Linking of Occupations

This is a companion repository for the paper 
[*MELO: An Evaluation Benchmark for Multilingual Entity Linking of Occupations*](https://github.com/Avature/melo-benchmark), 
which introduces the new **Multilingual Entity Linking of Occupations (MELO)** 
benchmark for evaluating the linking of occupation mentions across 21 languages 
to the multilingual ESCO Occupations taxonomy. The benchmark is composed of 48 
datasets built from high-quality, pre-existing human annotations.

In addition to providing these datasets and code for the standardized 
evaluation of models, this repository offers the source code necessary for 
generating new datasets. Researchers can use this code to define custom 
language combinations, enabling the creation of new task instances by selecting 
any subset of ESCO languages as targets for any of the source national 
classifications, allowing for multilingual variations of the tasks presented 
in the paper.

Specifically, this repository provides:

- **MELO Benchmark**: A collection of 48 datasets for entity linking of 
        occupations across 21 languages.
- **Custom Dataset Creation**: Tools to generate new datasets by combining 
        different target languages for each source terminology.
- **Standard Evaluation**: Tools to perform standard evaluation of entity 
        linking methods using the MELO benchmark.
- **Baselines**: Implementation of the baseline methods described in the 
        paper: lexical models and general-purpose sentence encoders in a 
        zero-shot setup.


## Repository Structure

```bash
melo-benchmark
 ├── data/                      # Directory for datasets
 │   ├── raw/                   # Raw data from public sources (e.g., ESCO, national terminologies)
 │   └── processed/             # Processed datasets ready for experiments
 ├── docs/                      # Documentation files
 ├── reports/                   # Output reports and visualizations
 ├── resources/                 # External resources
 ├── results/                   # Results from experiments and evaluations
 ├── scripts/                   # Scripts for creating datasets and reproducing experiments
 ├── src/                       # Source code
 │   └── melo_benchmark/        # Main package
 │       ├── analysis/          # Module for creating visualizations and reports
 │       ├── data_processing/   # Module for dataset creation and processing
 │       ├── evaluation/        # Module for evaluating models and generating metrics
 │       └── utils/             # Utility functions and helpers
 ├── tests/                     # Unit and integration tests
 ├── install_spacy_models.sh    # Script for installing spaCy models (lemmatizers)
 ├── LICENSE                    # License file
 ├── pyproject.toml             # Project metadata and dependencies
 ├── README.md                  # Project overview (this file)
 └── tox.ini                    # Configuration for automated testing
```


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/melo-benchmark.git
   cd melo-benchmark
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Basic Installation**: To install the project with default settings 
         (excluding `torch` and `tensorflow`), run:
     ```bash
     pip install -e .
     ```

   - To use spaCy lemmatizers, execute the following script
     ```bash
     sh install_spacy_models.sh
     ```
   - For reproducing experiments from the paper, you will need to install the 
         project with one of the backend frameworks. (**Note**: Currently, 
         simultaneous installation of both PyTorch and TensorFlow with GPU 
         support is not supported and you must choose one backend framework.)

     For **PyTorch** (with GPU support if available):
     ```bash
     pip install -e .[with-torch]
     ```

     For **TensorFlow** (with GPU support if available):
     ```bash
     pip install -e .[with-tf]
     ```

   - To reproduce experiments that include OpenAI models, you will need to 
         create a `.env` file in the root directory of the project. This file 
         should contain your OpenAI API credentials as environment variables. 
         The file should be structured as follows (replacing 
         `<your_openai_api_key>` with your actual API key):
     ```plaintext
     OPENAI_API_KEY=<your_openai_api_key>
     ```


## Datasets

The `data/` directory contains all the data needed for the MELO benchmark, 
organized into two main subdirectories: `raw/` and `processed/`.

### `raw/`
The `raw/` subdirectory holds the original files used to build the datasets, 
including data from the ESCO taxonomy and crosswalks between national 
occupational terminologies and ESCO. Additionally, for ease of use, we provide 
**standardized** versions of these files, which organize the information in a 
consistent format and simplify its processing in the project’s code.

The `raw/` directory is structured as follows:
- **crosswalks_original/**: Contains original crosswalks between national 
        terminologies and ESCO, as found online.
- **crosswalks_standard/**: Standardized versions of the national-to-ESCO 
        crosswalks, with each file collected and formatted into JSON for 
        consistency.
- **esco_original/**: Original versions of the ESCO occupations taxonomy, 
        corresponding to different versions of the taxonomy (1.0.3, 1.0.8, 
        etc.).
- **esco_standard/**: Standardized versions of the ESCO occupations data in 
        JSON format, ensuring uniformity across taxonomy versions.

### `processed/`
The `processed/` directory contains the final datasets used for 
experimentation. It is divided into two subdirectories:
- **melo/**: This contains the *official* datasets described in the MELO 
        benchmark paper. These datasets are based on specific language 
        combinations of source national terminology and target languages 
        and are meant to serve as standard evaluation datasets for the 
        benchmark.
- **custom/**: This directory stores any custom datasets generated by users. 
        These datasets are based on arbitrary combinations of national 
        terminologies and target languages, allowing for flexible 
        experimentation beyond the official benchmark configurations.

Each dataset in the `processed/` directory, regardless of the subdirectory,
is named using the following name template: 
`<SOURCE_COUNTRY>_q_<SOURCE_LANG>_c_[LIST_TARGET_LANGS]`, where 
`<SOURCE_COUNTRY>` is the lowercase ISO 3166-1 alpha-3 code of the country 
corresponding to the source national terminology (e.g. `usa` for O*NET), 
`<SOURCE_LANG>` is the lowercase ISO 639 2-character code for the source 
language (e.g. `en` for English) which corresponds to the national 
terminology, and `[LIST_TARGET_LANGS]` is the underscore-separated list of
ISO 639 2-character code for the set of ESCO languages that compose the corpus 
elements for the dataset, in alphabetical order (e.g. `de_es_fr` for German, 
Spanish, and French). The dataset is a folder with the following files:
- **annotations.tsv**: ground truth annotations in the form of binary relevance 
        signals, with (query ID, corpus element ID) pairs.
- **corpus_elements.tsv**: mapping between the corpus element IDs and
        corpus element surface forms.
- **queries.tsv**: mapping between the query IDs and the query surface forms.
- **logged_stats.txt**: A log file containing statistics and metadata about 
        the dataset.

### Download the Datasets

The raw and processed datasets are also available for download:

- **Raw data**: [Download from Zenodo (DOI: 10.5281/zenodo.13830967)](https://zenodo.org/records/13830968/files/melo_benchmark_raw.zip?download=1)
- **Processed datasets**: [Download from Zenodo (DOI: 10.5281/zenodo.13830967)](https://zenodo.org/records/13830968/files/melo_benchmark_processed.zip?download=1)

Alternatively, you can use the following script to automatically download the processed datasets:
```bash
python scripts/download_datasets.py
```

### Generate Custom Datasets

You can create custom datasets by specifying a source taxonomy (national 
terminology) and a set of target languages. This allows for flexible 
experimentation beyond the predefined MELO benchmark datasets.

To generate a new dataset, use the provided script and specify the source 
taxonomy and target languages. For example, to create a dataset using the 
German crosswalk (`deu_de`) and English, Spanish, and French as the target 
languages, you would run the following command:

```bash
python scripts/create_new_dataset.py --source_taxonomy deu_de --target_languages en,es,fr
```

The resulting dataset will be saved in the `data/processed/custom/` directory, 
following the same structure as the official datasets, making it easy to 
integrate with the evaluation scripts.



## Experiments

To reproduce the experiments described in the paper, use the following command:

```bash
python reproduce_paper_experiments.py
```

Make sure to have the processed datasets in the `data/processed/` directory.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) 
file for more information.


## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{retyk2024melo,
    title={{MELO: An Evaluation Benchmark for Multilingual Entity Linking of Occupations}},
    author={Federico Retyk and Luis Gasco and Casimiro Pio Carrino and Daniel Deniz and Rabih Zbib},
    journal={The 4th Workshop on Recommender Systems for Human Resources (RecSys in HR’24), in conjunction with the 18th ACM Conference on Recommender Systems},
    year={2024}
}
```


## Contact

For any questions or feedback, feel free to open an issue.
