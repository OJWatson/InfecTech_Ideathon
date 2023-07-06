from .train import get_train_state
from jax.random import PRNGKey
from jax import numpy as jnp

if __name__ == '__main__':
    state = get_train_state(PRNGKey(0), 7)
    print(
        state.apply_fn(
            state.params,
            jnp.ones((1, 100, 10)),
            jnp.ones((1, 8, 2))
        ).shape
    )
