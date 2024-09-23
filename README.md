# MELO Benchmark: Multilingual Entity Linking of Occupations


## Introduction

The **Multilingual Entity Linking of Occupations (MELO) Benchmark** is a new collection of datasets designed to evaluate entity linking in the domain of occupations, specifically linked to the multilingual ESCO Occupations taxonomy. MELO includes **48 datasets** in **21 languages**, making it an essential resource for research in cross-lingual and multilingual entity linking. This repository contains the code and resources to reproduce the creation of the datasets, as well as experiments with baseline models.

The primary goal of this benchmark is to provide researchers with a standard dataset to evaluate their entity linking models and to allow easy customization for creating new cross-lingual linking tasks.


## Features

- **48 Datasets**: Entity linking datasets across 21 languages.
- **Custom Dataset Creation**: Generate new datasets by combining different languages.
- **Baselines**: Baseline results using lexical models and general-purpose sentence encoders in a zero-shot setup.
- **Open Source**: All datasets and code are publicly available for reproduction and further research.


## Repository Structure

```bash
melo-benchmark
 ├── data/                      # Directory for datasets
 │   ├── raw/                   # Raw data from public sources (ESCO, national terminologies)
 │   └── processed/             # Processed datasets ready for experiments
 ├── docs/                      # Documentation
 ├── reports/                   # Output reports
 ├── resources/                 # External resources
 ├── results/                   # Output results
 ├── results/                   # Scripts for creating datasets and reproducing experiments
 ├── src/                       # Source code
 │   └── melo_benchmark/        # Main package
 │       ├── data_processing/   # Module for dataset creation
 │       ├── evaluation/        # Module for evaluating models
 │       ├── experiments/       # Scripts to reproduce the experiments
 │       └── utils/             # Utility functions
 ├── tests/                     # Unit and integration tests
 ├── install_spacy_models.sh    # Script for installing lemmatizers
 ├── LICENSE                    # License file
 ├── pyproject.toml             # Project metadata and dependencies
 ├── README.md                  # Project overview (this file)
 └── tox.ini                    # Configuration for testing
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
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install project**:
     ```bash
     pip install -e .
     ```
   - To use spaCy lemmatizers, execute the following script
     ```bash
     sh install_spacy_models.sh
     ```


## Datasets

### Download the Datasets

The raw and processed datasets are available for download:

- **Raw data**: [Download from Zenodo (DOI: 10.5281/zenodo.13830967)](https://zenodo.org/records/13830968/files/melo_benchmark_raw.zip?download=1)
- **Processed datasets**: [Download from Zenodo (DOI: 10.5281/zenodo.13830967)](https://zenodo.org/records/13830968/files/melo_benchmark_processed.zip?download=1)

Alternatively, you can use the following script to automatically download the processed datasets:
```bash
python scripts/download_datasets.py
```


### Generate Custom Datasets

You can generate new datasets by specifying a crosswalk and the set of target languages. 
Here's an example of how you can generate a dataset using the German crosswalk and the target
languages English, Spanish, and French:

```bash
python scripts/create_new_dataset.py --source_taxonomy deu_de --target_languages en,es,fr
```


## Experiments

To reproduce the experiments described in the paper, use the following command:

```bash
python reproduce_paper_experiments.py
```

Make sure to have the processed datasets in the `data/processed/` directory.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


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
