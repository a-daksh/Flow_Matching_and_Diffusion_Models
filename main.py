import os
import argparse
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from base import Sampleable, Trainer, ODE, ConditionalVectorField
from probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from simulators import EulerSimulator
from models import MNISTUNet
from utils import visualize_probability_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z,y=self.path.p_data.sample(batch_size)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask=torch.rand(batch_size)<self.eta
        y[mask]=10

        # Step 3: Sample t and x
        t=torch.rand(batch_size,1,1,1, device= z.device)
        x=self.path.sample_conditional_path(z,t)

        # Step 4: Regress and output loss
        loss=torch.nn.functional.mse_loss(self.model(x,t,y),  self.path.conditional_vector_field(x,z,t))
        return loss


def train(args):
    """Training function"""
    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    unet = MNISTUNet(
        channels = args.channels,
        num_residual_layers = args.num_residual_layers,
        t_embed_dim = args.t_embed_dim,
        y_embed_dim = args.y_embed_dim,
    )

    # Initialize trainer
    trainer = CFGTrainer(path=path, model=unet, eta=args.eta)

    checkpoint_path = args.checkpoint_path
    checkpoint_every = args.checkpoint_every
    
    def checkpoint_callback(epoch, model, optimizer, loss):
        if epoch % checkpoint_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_type': model.__class__.__name__,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model_config': {
                    'channels': args.channels,
                    'num_residual_layers': args.num_residual_layers,
                    't_embed_dim': args.t_embed_dim,
                    'y_embed_dim': args.y_embed_dim,
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"\n !!Checkpoint saved at epoch {epoch} to {checkpoint_path}!!")

    # Train!
    trainer.train(
        num_epochs=args.num_epochs,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_callback=checkpoint_callback,
    )
    
    print(f"\nTraining completed! Checkpoints saved to {checkpoint_path}")

def visualize_path(args):
    """Visualize the probability path"""
    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)
    
    visualize_probability_path(
        path=path,
        device=device,
        num_rows=args.vis_num_rows,
        num_cols=args.vis_num_cols,
        num_timesteps=args.vis_num_timesteps,
        output_path=args.output_path if hasattr(args, 'output_path') else None
    )

def inference(args):
    """Inference/generation function"""
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type')
    
    if model_type == 'MNISTUNet':
        model = MNISTUNet(
            channels = model_config.get('channels'),
            num_residual_layers = model_config.get('num_residual_layers'),
            t_embed_dim = model_config.get('t_embed_dim'),
            y_embed_dim = model_config.get('y_embed_dim'),
        )
        # NOTE: Add more model types here as needed!
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"Model {model_type} loaded from {args.checkpoint_path} (epoch {epoch}, loss: {loss:.4f})")
    
    model = model.to(device)
    model.eval()
    
    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)
    
    samples_per_class = args.samples_per_class
    num_timesteps = args.num_timesteps
    guidance_scales = args.guidance_scales

    # Graph
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(model, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        num_samples = y.shape[0]
        x0, _ = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

        # Simulate
        ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)

        # Plot
        grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
    
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        plt.savefig(args.output_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run inference on Flow/Diffusion models")
    parser.add_argument("mode", choices=["train", "inference", "viz_path"], help="Mode: train, inference, or visualize-path")
    
    # Training args
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eta", type=float, default=0.1, help="Label dropout probability for CFG")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint.pth", help="Path to save/load model checkpoint")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every N epochs")
    
    # Model args
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128], help="U-Net channel sizes")
    parser.add_argument("--num_residual_layers", type=int, default=2, help="Number of residual layers")
    parser.add_argument("--t_embed_dim", type=int, default=40, help="Time embedding dimension")
    parser.add_argument("--y_embed_dim", type=int, default=40, help="Label embedding dimension")
    
    # Inference args
    parser.add_argument("--samples_per_class", type=int, default=10, help="Samples per class for inference")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps for ODE integration")
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0, 3.0, 5.0], help="Guidance scales to test")
    parser.add_argument("--output_path", type=str, default="outputs/output.png", help="Path to save inference output image")
    
    # Visualization args
    parser.add_argument("--vis_num_rows", type=int, default=3, help="Number of rows for probability path visualization")
    parser.add_argument("--vis_num_cols", type=int, default=3, help="Number of columns for probability path visualization")
    parser.add_argument("--vis_num_timesteps", type=int, default=5, help="Number of timesteps for probability path visualization")
    
    args = parser.parse_args()
    
    if args.mode == "inference" and not args.checkpoint_path:
        parser.error("--checkpoint_path is required for inference mode")
    
    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "viz_path":
        visualize_path(args)