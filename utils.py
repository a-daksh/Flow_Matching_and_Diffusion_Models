import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

MiB = 1024 ** 2

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

def visualize_probability_path(path, device, num_rows=3, num_cols=3, num_timesteps=5, output_path=None):
    """
    Visualize samples from the conditional probability path at different time steps.
    
    Args:
        path: GaussianConditionalProbabilityPath instance
        device: torch.device to use
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        num_timesteps: Number of time steps to visualize
        output_path: Optional path to save the figure
    """
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, 32, 32)

    if num_timesteps == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6 * num_cols, 6 * num_rows))
        axes = [axes]  # Make it iterable
    else:
        fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1) # (num_samples, 1, 1, 1)
        xt = path.sample_conditional_path(z, tt) # (num_samples, 1, 32, 32)
        grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[tidx].axis("off")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    plt.show()

