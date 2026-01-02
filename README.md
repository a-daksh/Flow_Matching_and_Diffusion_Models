# Flow Matching and Diffusion Models

Personal repository for experimenting with flow matching and diffusion models, based on the [MIT 6.S184 course](https://diffusion.csail.mit.edu/2026/index.html).

## Overview

This repo contains implementations of flow matching and diffusion models for image generation. Currently working on training both models on MNIST, tuning hyperparameters, and exploring the differences between flow and diffusion approaches.

## Structure

- `main.py` - Main training and inference script
- `base.py` - Abstract base classes for models, trainers, and simulators
- `models.py` - Model architectures (U-Net, etc.)
- `probability_paths.py` - Probability path implementations
- `simulators.py` - ODE/SDE solvers
- `distributions.py` - Distribution implementations
- `utils.py` - Utility functions

## Usage

```bash
python main.py train --num_epochs 5000 --batch_size 250 --checkpoint_path checkpoints/flow_model.pth

python main.py inference --checkpoint_path checkpoints/flow_model.pth --output_path outputs/samples.png

python main.py visualize-path --vis_num_timesteps 5
```

## Status

Course materials completed. Currently experimenting with:
- Flow matching models and Diffusion models  on FashionMNIST 

