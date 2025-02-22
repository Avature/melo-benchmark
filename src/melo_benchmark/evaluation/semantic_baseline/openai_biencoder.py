import os
from typing import List

import openai

from melo_benchmark.evaluation.scorer import BiEncoderScorer
import melo_benchmark.utils.helper as melo_utils


class OpenAiBiEncoderScorer(BiEncoderScorer):

    def __init__(
                self,
                model_name: str,
                prompt_template: str,
                representation_cache_path: str = None,
                lowercase: bool = False,
                ascii_normalization: bool = False,
                batch_size: int = 32
            ):

        # This class assumes that the prompt template includes a variable
        #    with key {{job_title}}
        super().__init__(
            prompt_template=prompt_template,
            representation_cache_path=representation_cache_path,
            lowercase=lowercase,
            ascii_normalization=ascii_normalization,
            batch_size=batch_size
        )

        self.model_name = model_name

        melo_utils.load_dotenv_variables()

        open_ai_token = os.environ.get("OPENAI_API_KEY")
        if open_ai_token is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        openai.api_key = open_ai_token

        self.client = openai.OpenAI()

    def _free_model_resources(self):
        pass

    def _compute_embeddings(
                self,
                rendered_prompts: List[str]
            ) -> List[List[float]]:

        results = []
        for i in range(0, len(rendered_prompts), self.batch_size):
            batch = rendered_prompts[i:i + self.batch_size]

            batch_embeddings = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )

            batch_embeddings = batch_embeddings.dict()
            batch_embeddings = batch_embeddings['data']

            for embedding in batch_embeddings:
                results.append(embedding['embedding'])

        return results
