
# Evaluation of Models with the MELO Benchmark

## Standardized Evaluation Procedure

To evaluate your models using the MELO benchmark, you can follow the 
procedure outlined below. This example demonstrates how to set up and run a 
standardized evaluation using the `BenchmarkEvaluator` class and a 
`SentenceTransformersBiEncoderScorer`.

### Example Code

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

### Output

The evaluation results will be saved in the `results` directory. You can 
access the output files to analyze the performance of your models on the 
benchmark datasets.
