import jax
import jax.numpy as jnp
from base import Simulator, ODE, SDE, ConditionalVectorField

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: jax.Array, t: jax.Array, h: jax.Array, key: jax.Array, **kwargs) -> jax.Array:
        return xt + self.ode.drift_coefficient(xt,t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: jax.Array, t: jax.Array, h: jax.Array, key: jax.Array, **kwargs) -> jax.Array:
        return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * jnp.sqrt(h) * jax.random.normal(key, shape=xt.shape)

class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: jax.Array, t: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = jnp.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field


class CFGFlowSDE(SDE):
    """
    Stochastic SDE using trained FLOW model with score derived from vector field.
    
    Conversion (Proposition 1): s_t = (α_t * u_t - α̇_t * x) / (β_t² * α̇_t - α_t * β̇_t * β_t)
    SDE Extension (Theorem 13): dx = [u_t + 0.5*σ²*s_t] dt + σ dW
    """
    def __init__(self, net: ConditionalVectorField, alpha, beta, guidance_scale: float = 1.0, sigma: float = 0.1):
        self.net = net
        self.alpha = alpha
        self.beta = beta
        self.guidance_scale = guidance_scale
        self.sigma = sigma

    def drift_coefficient(self, x: jax.Array, t: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
        # Get vector field from flow model with CFG
        guided_vf = self.net(x, t, y)
        unguided_y = jnp.ones_like(y) * 10
        unguided_vf = self.net(x, t, unguided_y)
        vector_field = (1 - self.guidance_scale) * unguided_vf + self.guidance_scale * guided_vf
        
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        alpha_dt = self.alpha.dt(t)
        beta_dt = self.beta.dt(t)
        
        denominator = beta_t ** 2 * alpha_dt - alpha_t * beta_dt * beta_t
        denominator_safe = jnp.where(jnp.abs(denominator) < 1e-6, 1e-6 * jnp.sign(denominator + 1e-10), denominator)
        score = (alpha_t * vector_field - alpha_dt * x) / denominator_safe
        
        return vector_field + 0.5 * self.sigma ** 2 * score

    def diffusion_coefficient(self, x: jax.Array, t: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - diffusion: scalar or (bs, c, h, w)
        """
        return self.sigma
