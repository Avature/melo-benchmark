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

The MELO datasets are located in `data/processed/melo`. Each dataset consists
on the following files:
 - `corpus_elements.tsv`: mapping between the corpus element IDs and
        corpus element surface forms.
 - `queries.tsv`: mapping between the query IDs and the query surface forms.
 - `annotations.tsv`: annotations in the form of binary relevance signals, with 
        (query ID, corpus element ID) pairs.

Those files define a dataset. Additionally, we also include the following extra
files in the directory for each dataset:
 - `logged_stats.txt`: Statistics generated during the creation of the MELO dataset,
        starting from the publicly available crosswalk and ESCO occupation taxonomy
        version.
 - `surface_forms.json`: Mapping between query or corpus element IDs with their
        corresponding surface form, which may be useful for external systems computing
        representations for each dataset element with the aim of evaluation.


### Download the Datasets

The raw and processed datasets are also available for download:

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
