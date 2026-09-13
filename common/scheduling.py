"""Learning-rate policies shared by all benchmark scripts."""

from torch.optim.lr_scheduler import CosineAnnealingLR


def build_learning_rate_scheduler(optimizer, schedule, total_steps, min_learning_rate):
    """Build a step-wise scheduler while preserving the legacy constant-LR default."""
    normalized = schedule.strip().lower()
    if normalized == 'constant':
        return None
    if normalized == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_learning_rate,
        )
    raise ValueError(f"Unknown learning-rate schedule '{schedule}'")
