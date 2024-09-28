
# Standardized Evaluation of Models

This project provides a flexible framework for evaluating models with the
MELO benchmark. The evaluation process involves two key components:

1. **Scorer**: Computes ranking scores for each (query, corpus element) pair 
         of surface forms.
2. **Evaluator**: Loads the dataset, supplies the relevant data to the Scorer, 
         and aggregates the scores to compute the final evaluation metrics.

Depending on the researcher's use-case, various options are available for both 
Scorers and Evaluators.

### Scorer Options

- **Custom Scorer**: For general use cases, implement a custom Scorer by 
        creating a class that inherits from `BaseScorer` and implements 
        the `compute_scores` method to compute scores for 
        each (query, corpus element) pair.
- **Bi-Encoder Scorer**: If you are using a bi-encoder model, the 
        `BiEncoderScorer` class provides efficiency optimizations such as 
        computing all required embeddings only once and parallelizing score 
        computations (if a GPU is available).

### Evaluator Options

- **Standard Evaluator**: The `Evaluator` class can be used for computing 
        ranking metrics on any dataset (official MELO datasets or 
        custom datasets) with any Scorer implementation.
- **Benchmark Evaluator**: For bi-encoder models, the `BenchmarkEvaluator` 
        class allows comprehensive benchmarking using MELO by computing 
        embeddings for all query and corpus element surface forms across all 
        datasets at once. This evaluator only supports `BiEncoderScorer`
        instances.

### Extensibility

Both the Scorer and Evaluator classes are designed to be easily extendable, 
allowing you to adapt them for new use cases, such as evaluating 
cross-encoders or using occupation descriptions for evaluation.


## Examples

This section demonstrates how to evaluate various models using the MELO 
benchmark by leveraging different combinations of Scorers and Evaluators.


### Evaluate a Model from `sentence-transformers`

To perform a zero-shot evaluation of a `sentence-transformer` model as a
bi-encoder, you can follow the procedure outlined below. This example 
demonstrates how to use the `BenchmarkEvaluator` class and a 
`SentenceTransformersBiEncoderScorer` (which inherits from `BiEncoderScorer`). 
This combination is intended to 
evaluate the specified model across all official MELO datasets in a procedure
that is similar to the experiments presented in the paper.

```python
from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.evaluator import BenchmarkEvaluator
from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
    SentenceTransformersBiEncoderScorer

# Specify the name of the method
baseline_name = "BGE-M3"

# Specify the Hugging Face model name
hf_model_name = "BAAI/bge-m3"

# Define the prompt template; ensure it includes `{{job_title}}`
prompt_template = "The candidate's job title is: {{job_title}}."

melo_dataset_helper = OfficialDatasetHelper()
melo_datasets = melo_dataset_helper.get_dataset_configs()

# Initialize the scorer with the specified model and prompt template
scorer = SentenceTransformersBiEncoderScorer(
    model_name=hf_model_name,
    prompt_template=prompt_template,
    batch_size=16
)

# Initialize the evaluator with the datasets, baseline name, and scorer
evaluator = BenchmarkEvaluator(
    melo_datasets,
    baseline_name,
    scorer
)

# Perform the evaluation
evaluator.evaluate()
```

The `BenchmarkEvaluator` in conjunction with the any implementation of the 
`BiEncoderScorer` abstract class offers several optimizations to ensure 
efficient evaluation:
 - It first identifies all surface forms present in the evaluation datasets 
 - It computes their embeddings in batches
 - For each dataset, cosine similarity scores between queries and corpus 
       elements are calculated in parallel (if GPU is available)

When using `BenchmarkEvaluator`, the evaluation results will be saved 
in the `results` directory. You can 
access the output files to analyze the performance of your models on the 
benchmark datasets.


### Evaluate a Model from OpenAI

Similarly, you can use the `BenchmarkEvaluator` class and a 
`OpenAiBiEncoderScorer` to evaluate an OpenAI model. (Note that this 
requires a `.env` file in the root directory of the project with the OpenAI
API key.)


```python
from melo_benchmark.data_processing.official_dataset_helper import \
    OfficialDatasetHelper
from melo_benchmark.evaluation.evaluator import BenchmarkEvaluator
from melo_benchmark.evaluation.semantic_baseline.openai_biencoder import \
    OpenAiBiEncoderScorer

# Specify the name of the method
baseline_name = "OpenAI"

# Specify the Hugging Face model name
openai_model_name = "text-embedding-3-large"

# Define the prompt template; ensure it includes `{{job_title}}`
prompt_template = "The candidate's job title is: {{job_title}}."

melo_dataset_helper = OfficialDatasetHelper()
melo_datasets = melo_dataset_helper.get_dataset_configs()

# Initialize the scorer with the specified model and prompt template
scorer = OpenAiBiEncoderScorer(
    model_name=openai_model_name,
    prompt_template=prompt_template
)

# Initialize the evaluator with the datasets, baseline name, and scorer
evaluator = BenchmarkEvaluator(
    melo_datasets,
    baseline_name,
    scorer
)

# Perform the evaluation
evaluator.evaluate()
```


### Evaluate a BiEncoder Model for a Custom Dataset

When evaluating a custom dataset (a non-official combination of crosswalk and
target ESCO languages), you need to use the more general `Evaluator` class.

This class allows to evaluate a model (represented by any kind of Scorer) 
on a single dataset, without assuming its location nor the output directory.

For example, in order to evaluate a dataset that was created using the German
national terminology and languages Finnish, Hungarian, and Spanish as the
target languages. If this dataset was created using the 
`scripts/create_new_dataset.py` tool provided in this project, then the path 
of the directory containing the dataset files would be 
`data/processed/custom/deu_q_de_c_fi_hu_es`, but the `Evaluator` class accepts 
any path.

```python
import os

from melo_benchmark.evaluation.evaluator import Evaluator
from melo_benchmark.evaluation.semantic_baseline.stransf_biencoder import \
    SentenceTransformersBiEncoderScorer

# Specify the name of the method
baseline_name = "ESCOXLM-R"

# Specify the Hugging Face model name
hf_model_name = "jjzha/esco-xlm-roberta-large"

# Define the prompt template; ensure it includes `{{job_title}}`
prompt_template = "The candidate's job title is: {{job_title}}."

# Initialize the scorer with the specified model and prompt template
scorer = SentenceTransformersBiEncoderScorer(
    model_name=hf_model_name,
    prompt_template=prompt_template,
    batch_size=16
)

dataset_dir = "/path/to/dataset"
queries_file_path = os.path.join(
    dataset_dir,
    "queries.tsv"
)
corpus_elements_file_path = os.path.join(
    dataset_dir,
    "corpus_elements.tsv"
)
annotations_file_path = os.path.join(
    dataset_dir,
    "annotations.tsv"
)

output_path = "/path/for/storing/output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize the evaluator with the datasets, baseline name, and scorer
evaluator = Evaluator(
    queries_file_path,
    corpus_elements_file_path,
    annotations_file_path,
    output_path
)

# Perform the evaluation
evaluator.evaluate(scorer=scorer)
```

Although it is possible to replicate the functionality of `BatchEvaluator` in 
this way, evaluating every official MELO dataset one by one, the latter class 
offer important efficiency improvements, especially when a GPU is available.



### Evaluate any Model for any Dataset

The most general way to evaluate a model for any dataset is using `Evaluator` 
with a custom implementation of the abstract class `BaseScorer`. This class 
has an abstract method `compute_scores` which should be implemented such that
the evaluated model computes the ranking score for each (query, corpus 
element) pair of surface form involved in a dataset. 

```python
import abc
from typing import List

class BaseScorer(abc.ABC):

    @abc.abstractmethod
    def compute_scores(
                self,
                q_surface_forms: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        raise NotImplementedError()
```

This is an example of how to evaluate a custom model.


```python
import os
import random
from typing import List

from melo_benchmark.evaluation.evaluator import Evaluator
from melo_benchmark.evaluation.scorer import BaseScorer


def my_score_computation_function(q, c) -> float:
    return random.random()


class MyNewScorer(BaseScorer):

    def compute_scores(
                self,
                q_surface_forms: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:
        
        results = []

        for q_surface_form in q_surface_forms:
            results_for_q = []
            for c_surface_form in c_surface_forms:
                score = my_score_computation_function(
                    q_surface_form,
                    c_surface_form
                )
                results_for_q.append(score)
            results.append(results_for_q)
        
        # Needs to return, for each query, a scores for each corpus element
        return results


# Initialize the custom scorer
scorer = MyNewScorer()

dataset_dir = "/path/to/dataset"
queries_file_path = os.path.join(
    dataset_dir,
    "queries.tsv"
)
corpus_elements_file_path = os.path.join(
    dataset_dir,
    "corpus_elements.tsv"
)
annotations_file_path = os.path.join(
    dataset_dir,
    "annotations.tsv"
)

output_path = "/path/for/storing/output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize the evaluator with the datasets, baseline name, and scorer
evaluator = Evaluator(
    queries_file_path,
    corpus_elements_file_path,
    annotations_file_path,
    output_path
)

# Perform the evaluation
evaluator.evaluate(scorer=scorer)
```
