import csv
import os
from typing import (
    Dict,
    List
)

import numpy as np
from numpy.typing import NDArray
import openai

from melo_benchmark.evaluation.scorer import BiEncoderScorer
import melo_benchmark.utils.helper as melo_utils


class OpenAiBiEncoderScorer(BiEncoderScorer):

    def __init__(
                self,
                model_name: str,
                prompt_template: str,
                representation_cache_path: str,
                lowercase: bool = False,
                ascii_normalization: bool = True,
            ):

        # This class assumes that the prompt template includes a variable
        #    with key {{job_title}}
        super().__init__(
            prompt_template,
            representation_cache_path,
            lowercase,
            ascii_normalization
        )

        self.model_name = model_name

        melo_utils.load_dotenv_variables()

        open_ai_token = os.environ.get("OPENAI_API_KEY")
        if open_ai_token is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        openai.api_key = open_ai_token

        self.client = openai.OpenAI()

    def _compute_representations(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        sf_repr_mapping = {}

        with open(self.repr_mapping_cache_path, "a") as f_out:
            tsv_writer = csv.writer(f_out, delimiter='\t')

            for surface_form in surface_forms:
                text_prompt = self._render_template(
                    job_title=surface_form
                )
                response = self.client.embeddings.create(
                    input=[text_prompt],
                    model=self.model_name
                )
                response = response.dict()
                embedding = response['data'][0]['embedding']

                tsv_writer.writerow(
                    [surface_form] + [str(x) for x in embedding]
                )

                embedding = np.array([float(x) for x in embedding])
                sf_repr_mapping[surface_form] = embedding

        return sf_repr_mapping
