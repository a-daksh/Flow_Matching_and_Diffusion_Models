import jax
import jax.numpy as jnp
from base import Simulator, ODE, SDE
from models import ConditionalVectorField

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
