import torch


def _correct_level_predictions(targets, predictions, level):
    targets = targets[level]
    predictions = predictions[level]

    targets = targets.argmax(dim=1)
    predictions = predictions.argmax(dim=1)
    return targets == predictions


def cumulative_accuracy(targets, predictions):
    """
    Calculate the cumulative precision of the predictions.

    Args:
        targets (torch.Tensor): The true labels of the data.
        predictions (torch.Tensor): The predicted labels of the data.

    Returns:
        float: The cumulative precision of the predictions.
    """
    num_samples = len(targets[0])

    cumulative_correct_predictions = torch.ones(
        num_samples, dtype=torch.bool, device=predictions[0].device
    )

    level_accuracy = []

    for level in range(len(targets)):
        correct_predictions = _correct_level_predictions(targets, predictions, level)
        cumulative_correct_predictions &= correct_predictions
        level_accuracy.append(cumulative_correct_predictions.float().mean().item())

    return level_accuracy
