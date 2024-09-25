import csv
from typing import (
    Dict,
    List
)

import numpy as np
from numpy.typing import NDArray
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

        self.backbone = hub.KerasLayer(
            model_name,
            trainable=False,
            name="tf_hub_encoder"
        )

        self.l2_normalization_layer = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=-1),
            name="l2_norm"
        )

    def _compute_embedding(self, prompt_text: str) -> List[int]:
        sentence_tensor = tf.constant([prompt_text])
        x = self.backbone(sentence_tensor)
        if (type(x) is list) and len(x) == 1:
            x = x[0]
        x = self.l2_normalization_layer(x)
        x = x.numpy().flatten().tolist()
        return x

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
