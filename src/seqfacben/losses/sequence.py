import torch.nn.functional as F


def cross_entropy(logits, targets):
    """
    Cross-entropy for sequence predictions.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )