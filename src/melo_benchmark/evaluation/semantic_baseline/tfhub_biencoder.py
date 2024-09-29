from typing import List

import tensorflow as tf
import tensorflow_hub as hub

# noinspection PyUnresolvedReferences
import tensorflow_text

from melo_benchmark.evaluation.scorer import BiEncoderScorer


class TFHubBiEncoderScorer(BiEncoderScorer):

    def __init__(
                self,
                model_name: str,
                prompt_template: str,
                representation_cache_path: str = None,
                lowercase: bool = False,
                ascii_normalization: bool = True,
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

        self.backbone = hub.KerasLayer(
            model_name,
            trainable=False,
            name="tf_hub_encoder"
        )

        self.l2_normalization_layer = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=-1),
            name="l2_norm"
        )

    def _free_model_resources(self):
        del self.backbone
        del self.l2_normalization_layer

    def _compute_embeddings(
                self,
                rendered_prompts: List[str]
            ) -> List[List[float]]:

        results = []
        for i in range(0, len(rendered_prompts), self.batch_size):
            batch = rendered_prompts[i:i + self.batch_size]

            # noinspection PyCallingNonCallable
            batch_embeddings = self.backbone(batch)

            batch_embeddings = self.l2_normalization_layer(batch_embeddings)

            for embedding in batch_embeddings:
                results.append(embedding.numpy().flatten().tolist())

        return results
