import csv
from typing import (
    Dict,
    List
)

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from melo_benchmark.evaluation.scorer import BiEncoderScorer


# noinspection DuplicatedCode
class SentenceTransformersBiEncoderScorer(BiEncoderScorer):

    def __init__(
                self,
                model_name: str,
                prompt_template: str,
                representation_cache_path: str,
                lowercase: bool = False,
                ascii_normalization: bool = False,
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
        self.model = SentenceTransformer(model_name)

    def _compute_embedding(self, prompt_text: str) -> List[int]:
        normalized_prompt_embedding = self.model.encode(
            prompt_text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        return normalized_prompt_embedding.squeeze().tolist()

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
                embedding = self._compute_embedding(text_prompt)

                tsv_writer.writerow(
                    [surface_form] + [str(x) for x in embedding]
                )

                embedding = np.array([float(x) for x in embedding])
                sf_repr_mapping[surface_form] = embedding

        return sf_repr_mapping
