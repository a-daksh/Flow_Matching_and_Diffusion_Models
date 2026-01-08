from abc import ABC, abstractmethod
from typing import Optional, Tuple
from tqdm import tqdm
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from utils import model_size_b, MiB

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, key: jax.random.PRNGKey, num_samples: int) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass

# Abstract class for scheduler functions
class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert jnp.allclose(
            self(jnp.zeros((1,1,1,1))), jnp.zeros((1,1,1,1))
        )
        # Check alpha_1 = 1
        assert jnp.allclose(
            self(jnp.ones((1,1,1,1))), jnp.ones((1,1,1,1))
        )

    @abstractmethod
    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: jax.Array) -> jax.Array:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t_flat = t.squeeze()
        dt = jax.vmap(jax.grad(lambda s: self(s[None, None, None, None]).squeeze()))(t_flat)
        return dt.reshape(-1, 1, 1, 1)

class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert jnp.allclose(
            self(jnp.zeros((1,1,1,1))), jnp.ones((1,1,1,1))
        )
        # Check beta_1 = 0
        assert jnp.allclose(
            self(jnp.ones((1,1,1,1))), jnp.zeros((1,1,1,1))
        )

    @abstractmethod
    def __call__(self, t: jax.Array) -> jax.Array:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: jax.Array) -> jax.Array:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """
        t_flat = t.squeeze()
        dt = jax.vmap(jax.grad(lambda s: self(s[None, None, None, None]).squeeze()))(t_flat)
        return dt.reshape(-1, 1, 1, 1)

# Abstract class for both ODE and SDE
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: jax.Array, t: jax.Array, **kwargs) -> jax.Array:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: jax.Array, t: jax.Array, **kwargs) -> jax.Array:
        """
        Returns the drift coefficient of the SDE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: jax.Array, t: jax.Array, **kwargs) -> jax.Array:
        """
        Returns the diffusion coefficient of the SDE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass

# Abstract class for simulators
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: jax.Array, t: jax.Array, dt: jax.Array, key: jax.Array, **kwargs) -> jax.Array:
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time step, shape (bs, 1, 1, 1)
            - key: JAX PRNG key (for SDE simulators that need randomness)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass

    def simulate(self, x: jax.Array, ts: jax.Array, key: jax.Array, **kwargs) -> jax.Array:
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
            - key: JAX PRNG key
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        keys = jax.random.split(key, nts - 1)
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, keys[t_idx], **kwargs)
        return x

    def simulate_with_trajectory(self, x: jax.Array, ts: jax.Array, key: jax.Array, **kwargs) -> jax.Array:
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
            - key: JAX PRNG key
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x]
        nts = ts.shape[1]
        keys = jax.random.split(key, nts - 1)
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, keys[t_idx], **kwargs)
            xs.append(x)
        return jnp.stack(xs, axis=1)

# Abstract class for conditional 
class ConditionalProbabilityPath(ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: jax.Array, key: jax.Array) -> jax.Array:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
            - key: JAX PRNG key
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        key1, key2 = jax.random.split(key)
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(key1, num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t, key2) # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, key: jax.Array, num_samples: int) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Samples the conditioning variable z and label y
        Args:
            - key: JAX PRNG key
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: jax.Array, t: jax.Array, key: jax.Array) -> jax.Array:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
            - key: JAX PRNG key
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: jax.Array, z: jax.Array, t: jax.Array) -> jax.Array:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: jax.Array, z: jax.Array, t: jax.Array) -> jax.Array:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        pass

class ConditionalVectorField(ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def __call__(self, x: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass

# Abstract class for training models
class Trainer(ABC):
    def __init__(self, model: eqx.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, model: eqx.Module, **kwargs) -> jax.Array:
        """Compute loss given a model. Must be a pure function."""
        pass

    def get_optimizer(self, lr: float):
        return optax.adam(lr)

    def train(self, num_epochs: int, lr: float = 1e-3, checkpoint_callback=None, **kwargs):
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')

        # Initialize optimizer
        opt = self.get_optimizer(lr)
        opt_state = opt.init(eqx.filter(self.model, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, opt_state, **loss_kwargs):
            loss, grads = eqx.filter_value_and_grad(self.get_train_loss)(model, **loss_kwargs)
            updates, opt_state = opt.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        # Train loop
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            self.model, opt_state, loss = make_step(self.model, opt_state, **kwargs)
            loss_val = float(loss)
            pbar.set_description(f'Epoch {epoch}, loss: {loss_val:.3f}')
            
            if checkpoint_callback is not None:
                checkpoint_callback(epoch, self.model, opt_state, loss_val)