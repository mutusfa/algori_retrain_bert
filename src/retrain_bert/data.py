import tensorflow as tf


class AugmentedClassifiedOcr(tf.data.Dataset):
    def __init__(self, ocr_lines, labels, *args, random_state=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocr_lines = ocr_lines
        self.labels = labels
