import numpy as np

from retrain_bert import settings


class ExactCategoryScore:
    def __init__(self, labels_conf):
        self.labels_conf = labels_conf
        self.unknown_str = "".join([f"{l['num_classes'] - 1:02d}" for l in labels_conf])

    def compare_cat_codes(self, pred_code, true_code):
        for level in range(settings.DEEPEST_LEVEL):
            true_cat = true_code[2 * level : 2 * level + 2]
            unknown_cat = self.unknown_str[2 * level : 2 * level + 2]
            if true_cat == unknown_cat:
                break
        true_code = true_code[: level * 2]
        pred_code = pred_code[: len(true_code)]
        return pred_code == true_code

    def exact_category(self, preds, true, level):
        pred_code = np.array([np.argmax(p, axis=1).tolist() for p in preds]).T
        true_code = np.array([np.argmax(t, axis=1).tolist() for t in true]).T

        pred_cat_str = [
            "".join([f"{c:02d}" for c in line]) for line in pred_code[:, : level + 1]
        ]
        true_cat_str = [
            "".join([f"{c:02d}" for c in line]) for line in true_code[:, : level + 1]
        ]

        return [
            self.compare_cat_codes(pred, true)
            for pred, true in zip(pred_cat_str, true_cat_str)
        ]

    def exact_category_score(self, preds, true, level):
        return np.mean(self.exact_category(preds, true, level))

    def exact_category_score_from_codes(self, pred_codes, true_codes, level):
        pred_codes = [p[: level * 2] for p in pred_codes]
        true_codes = [t[: level * 2] for t in true_codes]
        return np.mean(
            [
                self.compare_cat_codes(pred, true)
                for pred, true in zip(pred_codes, true_codes)
            ]
        )


def make_targets(data, labels_conf):
    labels = data.drop("OcrValue", axis="columns")
    targets = []
    for level, col in zip(range(settings.DEEPEST_LEVEL), labels.columns):
        level_targets = np.zeros(
            (len(labels), labels_conf[level]["num_classes"]), dtype=np.float32
        )
        for i, label in enumerate(labels[col]):
            level_targets[i, label] = 1
        targets.append(level_targets)

    return targets
