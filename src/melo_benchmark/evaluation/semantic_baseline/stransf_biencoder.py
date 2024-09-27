from typing import List

from sentence_transformers import SentenceTransformer

from melo_benchmark.evaluation.scorer import BiEncoderScorer


class SentenceTransformersBiEncoderScorer(BiEncoderScorer):

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
        self.model = SentenceTransformer(model_name)

    def _compute_embeddings(
                self,
                rendered_prompts: List[str]
            ) -> List[List[float]]:

        results = []
        for i in range(0, len(rendered_prompts), self.batch_size):
            batch = rendered_prompts[i:i + self.batch_size]

            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size
            )

            for embedding in batch_embeddings:
                results.append(embedding.squeeze().tolist())

        return results
