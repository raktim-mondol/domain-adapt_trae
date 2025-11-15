from .mmd import mmd_loss
from .orthogonal import orthogonal_constraint
from .entropy import entropy_loss, entropy_loss_from_logits
from .consistency import consistency_mse, consistency_kl

__all__ = [
    'mmd_loss',
    'orthogonal_constraint',
    'entropy_loss',
    'entropy_loss_from_logits',
    'consistency_mse',
    'consistency_kl'
]
