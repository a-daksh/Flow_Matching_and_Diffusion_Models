import jax
from base import Sampleable
from typing import List, Tuple, Optional

class IsotropicGaussian(Sampleable):
    """
    Sampleable wrapper around jax.random.normal
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std

    def sample(self, key, num_samples) -> Tuple[jax.Array, Optional[jax.Array]]:
        return self.std * jax.random.normal(key,shape=(num_samples, *self.shape)), None

