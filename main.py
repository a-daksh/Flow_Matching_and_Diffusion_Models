import os
import argparse
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

# NOTE: Prevent TensorFlow from grabbing GPU memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

from base import Sampleable, Trainer, ODE, ConditionalVectorField
from probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from simulators import EulerSimulator, CFGVectorFieldODE
from models import UNet
from utils import visualize_probability_path

class MNISTSampler(Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        print("Loading dataset...")
        # Load dataset as numpy arrays
        ds = tfds.load('fashion_mnist', split='train', as_supervised=True, batch_size=-1)
        ds = tfds.as_numpy(ds)
        
        images, labels = ds
        # Preprocess: resize to 32x32 and normalize to [-1, 1]
        images = tf.image.resize(images, [32, 32]).numpy()
        images = images.astype(np.float32) / 255.0
        images = (images - 0.5) / 0.5
        
        self.images = images  # (60000, 32, 32, 1) - NHWC format
        self.labels = labels.astype(np.int32)  # (60000,)
        self.ds_size = len(self.images)

    def sample(self, key: jax.random.PRNGKey, num_samples: int) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Args:
            - key: JAX PRNG key
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size,)
        """
        if num_samples > self.ds_size:
            raise ValueError(f"num_samples exceeds dataset size: {self.ds_size}")

        rng = np.random.default_rng(int(key[0]))
        indices = rng.choice(self.ds_size, size=num_samples, replace=False)
        
        images_batch = self.images[indices]  # (batch_size, 32, 32, 1)
        labels_batch = self.labels[indices]  # (batch_size,)
        
        # Convert batch to JAX (GPU)
        images_jax = jnp.array(images_batch)
        images_jax = jnp.transpose(images_jax, (0, 3, 1, 2))  # NHWC -> NCHW
        labels_jax = jnp.array(labels_batch, dtype=jnp.int32)
        
        return images_jax, labels_jax

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int):
        # Step 1: Sample z,y from p_data
        key1, key2, key3, key4 = jax.random.split(key, 4)
        z, y = self.path.p_data.sample(key1, batch_size)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = jax.random.uniform(key2, shape=(batch_size,)) < self.eta
        y = jnp.where(mask, 10, y)

        # Step 3: Sample t and x
        t = jax.random.uniform(key3, shape=(batch_size, 1, 1, 1))
        x = self.path.sample_conditional_path(z, t, key4)
        
        return x, z, t, y

    def get_train_loss(self, model: eqx.Module, x: jax.Array, z: jax.Array, t: jax.Array, y: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        batch_size = x.shape[0]
        # UNet expects: (t_scalar, y_image, y_label, key) where y_image is (C, H, W), y_label is scalar
        def single_sample_model(x_i, t_i, y_i, key_i):
            t_scalar = t_i[0, 0, 0]  # Extract scalar from (1,1,1) shape
            return model(t_scalar, x_i, y_i, key=key_i)
        
        vmapped_model = jax.vmap(single_sample_model, in_axes=(0, 0, 0, 0))
        
        model_keys = jax.random.split(key, batch_size)        
        pred = vmapped_model(x, t, y, model_keys)  # (batch_size, C, H, W)
        target = self.path.conditional_vector_field(x, z, t)
        loss = jnp.mean((pred - target) ** 2)
        return loss


def train(args):
    """Training function"""
    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    )

    # Initialize model
    # Map old args to UNet parameters
    # channels = [32, 64, 128] -> dim_mults = [2, 4] (relative to hidden_size)
    # hidden_size = channels[0] = 32
    # dim_mults = [c // hidden_size for c in channels[1:]] = [64//32, 128//32] = [2, 4]
    hidden_size = args.channels[0] if args.channels else 32
    dim_mults = [c // hidden_size for c in args.channels[1:]] if len(args.channels) > 1 else [2, 4]
    
    init_key = jax.random.PRNGKey(args.seed)
    unet = UNet(
        data_shape = (1, 32, 32),
        is_biggan = False,
        dim_mults = dim_mults,
        hidden_size = hidden_size,
        y_emb_dim = args.y_embed_dim,
        heads = 4,
        dim_head = 32,
        dropout_rate = 0.1,
        num_res_blocks = args.num_residual_layers,
        attn_resolutions = [16],  # Attention at 16x16 resolution
        key = init_key,
    )

    # Initialize trainer
    trainer = CFGTrainer(path=path, model=unet, eta=args.eta)

    checkpoint_path = args.checkpoint_path
    checkpoint_every = args.checkpoint_every
    
    def checkpoint_callback(epoch, model, opt_state, loss):
        if epoch % checkpoint_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_type': model.__class__.__name__,
                'opt_state': opt_state,
                'loss': loss,
                'model_config': {
                    'data_shape': (1, 32, 32),
                    'is_biggan': False,
                    'dim_mults': dim_mults,
                    'hidden_size': hidden_size,
                    'y_emb_dim': args.y_embed_dim,
                    'heads': 4,
                    'dim_head': 32,
                    'dropout_rate': 0.1,
                    'num_res_blocks': args.num_residual_layers,
                    'attn_resolutions': [16],
                }
            }
            eqx.tree_serialise_leaves(checkpoint_path, (checkpoint, model))
            print(f"\n !!Checkpoint saved at epoch {epoch} to {checkpoint_path}!!")

    # Train!
    trainer.train(
        num_epochs=args.num_epochs,
        lr=args.lr,
        checkpoint_callback=checkpoint_callback,
        key=jax.random.PRNGKey(args.seed + 1),
        batch_size=args.batch_size,
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
    )
    
    visualize_probability_path(
        path=path,
        num_rows=args.vis_num_rows,
        num_cols=args.vis_num_cols,
        num_timesteps=args.vis_num_timesteps,
        output_path=args.output_path if hasattr(args, 'output_path') else None
    )

def inference(args):
    """Inference/generation function"""
    # We need to deserialize just the dict part first to read the config
    dummy_model = None  # Placeholder for first deserialization
    checkpoint_meta, _ = eqx.tree_deserialise_leaves(args.checkpoint_path, (dict, dummy_model))
    
    # Extract model config from checkpoint
    model_config = checkpoint_meta.get('model_config')
    if model_config is None:
        raise ValueError(f"Checkpoint at {args.checkpoint_path} is missing 'model_config'. ")
    
    print(f"Loading model with config: {model_config}")
    
    init_key = jax.random.PRNGKey(0)
    model_template = UNet(
        data_shape=tuple(model_config['data_shape']),
        is_biggan=model_config['is_biggan'],
        dim_mults=model_config['dim_mults'],
        hidden_size=model_config['hidden_size'],
        y_emb_dim=model_config['y_emb_dim'],
        heads=model_config['heads'],
        dim_head=model_config['dim_head'],
        dropout_rate=model_config['dropout_rate'],
        num_res_blocks=model_config['num_res_blocks'],
        attn_resolutions=model_config['attn_resolutions'],
        key=init_key,
    )
    
    checkpoint, model = eqx.tree_deserialise_leaves(args.checkpoint_path, (dict, model_template))
    
    model_type = checkpoint.get('model_type', 'UNet')
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"Model {model_type} loaded from {args.checkpoint_path} (epoch {epoch}, loss: {loss:.4f})")
    
    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    )
    
    samples_per_class = args.samples_per_class
    num_timesteps = args.num_timesteps
    guidance_scales = args.guidance_scales

    # Graph
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

    # NOTE: Created a wrapper that implements ConditionalVectorField interface for UNet
    # CFGVectorFieldODE expects batches (x, t, y) but UNet is single-sample
    # NOTE: We disable dropout during inference
    class UNetWrapper(ConditionalVectorField):
        def __init__(self, unet_model):
            self.unet = unet_model
        
        def __call__(self, x: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
            """
            Args:
            - x: (bs, c, h, w)
            - t: (bs, 1, 1, 1)
            - y: (bs,)
            Returns:
            - u_t^theta(x|y): (bs, c, h, w)
            """
            def single_sample(x_i, t_i, y_i):
                t_scalar = t_i[0, 0, 0]
                return eqx.nn.inference_mode(self.unet)(t_scalar, x_i, y_i, key=None)
            
            vmapped_model = jax.vmap(single_sample, in_axes=(0, 0, 0))
            return vmapped_model(x, t, y)
    
    wrapped_model = UNetWrapper(model)

    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(wrapped_model, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        y_labels = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32)
        y = jnp.repeat(y_labels, samples_per_class)  # (num_samples,)
        num_samples = y.shape[0]
        
        key = jax.random.PRNGKey(42)
        x0, _ = path.p_simple.sample(key, num_samples)  # (num_samples, 1, 32, 32)

        # Simulate
        ts_base = jnp.linspace(0, 1, num_timesteps)  # (num_timesteps,)
        ts = jnp.broadcast_to(ts_base[None, :, None, None, None], (num_samples, num_timesteps, 1, 1, 1))
        
        sim_key = jax.random.PRNGKey(0)
        x1 = simulator.simulate(x0, ts, key=sim_key, y=y)  # (num_samples, 1, 32, 32)

        # Plot
        x1_torch = torch.from_numpy(np.asarray(x1))
        grid = make_grid(x1_torch, nrow=samples_per_class, normalize=True, value_range=(-1,1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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