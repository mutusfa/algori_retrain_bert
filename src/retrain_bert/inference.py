import functools
import itertools
from typing import Iterable, Generator

import numpy as np
import tensorflow_text as text  # tf registers ops on import
import tensorflow as tf
import tensorflow_hub as hub

from retrain_bert import preprocessor, settings


def chunk_it(sequence, chunksize):
    """
    Split an iterable into chunks of fixed length.
    """
    it = iter(sequence)
    return iter(lambda: tuple(itertools.islice(it, chunksize)), ())


class InferenceModel:
    def __init__(
        self,
        model_path=settings.INFERENCE_MODEL_PATH,
    ):
        self.model_path = model_path
        self.labels = preprocessor.load_labels()
        self.labels_conf = preprocessor.get_labels_conf(self.labels)

    @functools.cached_property
    def model(self):
        custom_objects = {"KerasLayer": hub.KerasLayer}
        with tf.keras.utils.custom_object_scope(custom_objects):
            return tf.keras.models.load_model(settings.INFERENCE_MODEL_PATH)

    def predict(self, *args, **kwds):
        return self.infer(*args, **kwds)

    def infer(self, sentences) -> list:
        if isinstance(sentences, str):
            sentences = [sentences]
        inferences = self.model(tf.constant(sentences), training=False)
        inferred_categories = np.array(
            [np.argmax(p, axis=1).tolist() for p in inferences]
        ).T
        assert inferred_categories.shape[0] == len(sentences)
        assert inferred_categories.shape[1] == settings.DEEPEST_LEVEL

        # model was trained on 0 based categories while in database we have 1 based
        # category codes
        inferred_categories += 1

        # transform inferred labels into category codes
        category_codes = []
        for line in inferred_categories:
            codes = []
            for level, code in enumerate(line):
                # if unknown category
                if code == self.labels_conf[level]["num_classes"]:
                    break
                codes.append(f"{code:02d}")
            category_codes.append("".join(codes))

        return category_codes
