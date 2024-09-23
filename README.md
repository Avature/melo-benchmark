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
 ├── src/                       # Source code
 │   └── melo_benchmark/        # Main package
 │       ├── data_processing/   # Module for dataset creation
 │       ├── evaluation/        # Module for evaluating models
 │       ├── experiments/       # Scripts to reproduce the experiments
 │       └── utils/             # Utility functions
 ├── tests/                     # Unit and integration tests
 ├── .github/workflows/ci.yml   # CI/CD pipeline configuration
 ├── README.md                  # Project overview (this file)
 ├── pyproject.toml             # Project metadata and dependencies
 └── tox.ini                    # Configuration for testing and linting
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

- **Raw data**: [Download from Google Drive](https://drive.google.com/uc?id=RAW_FILE_ID)
- **Processed datasets**: [Download from Zenodo (DOI: 10.5281/zenodo.123456)](https://zenodo.org/record/123456)

Alternatively, you can use the following script to automatically download the processed datasets:
```bash
python src/melo_benchmark/download_datasets.py
```

### Generate Custom Datasets

You can generate new datasets by specifying different languages for the query and corpus elements. The tool supports multiple configurations, such as creating cross-lingual tasks (e.g., Italian to Greek). Here's how you can generate a dataset:

```bash
python src/melo_benchmark/data_processing/dataset_builder.py --esco-file path/to/esco.json --terminology-file path/to/terminology.json --output-dir data/processed --languages en fr
```

## Experiments

To reproduce the experiments described in the paper, use the following command:

```bash
python src/melo_benchmark/experiments/run_experiments.py
```

Make sure to have the processed datasets in the `data/processed/` directory.

## Evaluation

The evaluation module provides standardized metrics to evaluate your entity linking models. Here's an example usage:

```python
from src.melo_benchmark.evaluation.evaluator import Evaluator

ground_truth = [...]  # List of ground truth labels
predictions = [...]   # List of predicted labels

evaluator = Evaluator(ground_truth, predictions)
results = evaluator.evaluate()

print(results)  # {'accuracy': 0.85, 'f1_score': 0.82}
```

## Contributing

We welcome contributions! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Run tests (`pytest`).
5. Push to your branch (`git push origin feature/your-feature`).
6. Create a pull request.

Please ensure that your code adheres to the coding standards outlined in the `pyproject.toml` file.

## Running Tests

To run the unit and integration tests, use:

```bash
pytest
```

Alternatively, you can use `tox` to run tests in multiple environments:

```bash
tox
```

## Continuous Integration

This repository uses GitHub Actions for continuous integration. Every push to the `main` branch and every pull request triggers a workflow that runs the unit tests. You can view the status of the workflow [here](https://github.com/yourusername/melo-benchmark/actions).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{retyk2024melo,
    title={{MELO: An Evaluation Benchmark for Multilingual Entity Linking of Occupations}},
    author={Federico Retyk and Luis Gasco and Casimiro Pio Carrino and Daniel Deniz and Rabih Zbib},
    journal={The 4rd Workshop on Recommender Systems for Human Resources (RecSys in HR’24), in conjunction with the 18th ACM Conference on Recommender Systems},
    year={2024}
}
```

## Contact

For any questions or feedback, feel free to open an issue or contact the project maintainer at your.email@example.com.
