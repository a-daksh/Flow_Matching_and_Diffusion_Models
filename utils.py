import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

MiB = 1024 ** 2


def model_size_b(model) -> int:
    """
    Returns model size in bytes. 
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    # TODO: Check this once 
    for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)):
        size += leaf.nbytes
    return size

def visualize_probability_path(path, num_rows=3, num_cols=3, num_timesteps=5, output_path=None):
    """
    Visualize samples from the conditional probability path at different time steps.
    
    Args:
        path: GaussianConditionalProbabilityPath instance
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        num_timesteps: Number of time steps to visualize
        output_path: Optional path to save the figure
    """
    num_samples = num_rows * num_cols
    key = jax.random.PRNGKey(0)
    
    key, subkey = jax.random.split(key)
    z, _ = path.p_data.sample(subkey, num_samples)
    
    if num_timesteps == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6 * num_cols, 6 * num_rows))
        axes = [axes]  # Make it iterable
    else:
        fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = jnp.linspace(0, 1, num_timesteps)
    keys = jax.random.split(key, num_timesteps)
    for tidx, t in enumerate(ts):
        tt = jnp.full((num_samples, 1, 1, 1), float(t))
        xt = path.sample_conditional_path(z, tt, keys[tidx])  # (num_samples, 1, 32, 32)
        
        # NOTE: Convert JAX array to torch for make_grid (visualization only)
        xt_torch = torch.from_numpy(np.asarray(xt).copy())
        grid = make_grid(xt_torch, nrow=num_cols, normalize=True, value_range=(-1, 1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        axes[tidx].axis("off")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    plt.show()

