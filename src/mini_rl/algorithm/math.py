from collections import defaultdict
import torch


def group_normalize_advantages(
    rewards: torch.Tensor, group_ids: list[str]
) -> torch.Tensor:
    if rewards.numel() != len(group_ids):
        raise ValueError("rewards and group_ids length mismatch")

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        grouped_indices[gid].append(idx)

    advantages = torch.zeros_like(rewards)
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            for idx in indices:
                advantages[idx] = 0.0
            continue
        values = rewards[indices]
        std = values.std(unbiased=False)
        if std.item() == 0.0:
            for idx in indices:
                advantages[idx] = 0.0
        else:
            normed = (values - values.mean()) / (std + 1e-8)
            for i, idx in enumerate(indices):
                advantages[idx] = normed[i]

    return advantages
