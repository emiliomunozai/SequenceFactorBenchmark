import torch
import torch.nn.functional as F


def cross_entropy(logits, targets):
    """
    Cross-entropy for sequence predictions.
    logits: [B, L, V]   targets: [B, L]
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )


def shift_tolerant_ce(logits, targets, window=2, sigma=1.0):
    """
    Shift-tolerant cross-entropy.  At each output position t the target is a
    soft distribution that blends the one-hot targets from positions
    [t-window … t+window], weighted by a Gaussian kernel with the given sigma.

    A perfect copy shifted by one position is penalised much less than with
    standard CE, while a perfectly aligned copy still achieves the optimum.

    logits: [B, L, V]   targets: [B, L]
    """
    B, L, V = logits.shape

    offsets = torch.arange(-window, window + 1, device=targets.device)
    weights = torch.exp(-offsets.float() ** 2 / (2 * sigma ** 2))
    weights = weights / weights.sum()  # [2*window+1]

    one_hot = F.one_hot(targets, V).float()                     # [B, L, V]
    padded = F.pad(one_hot, (0, 0, window, window))             # [B, L+2w, V]

    soft_targets = torch.zeros_like(one_hot)
    for i, off in enumerate(range(-window, window + 1)):
        soft_targets += weights[i] * padded[:, window + off : window + off + L, :]

    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()