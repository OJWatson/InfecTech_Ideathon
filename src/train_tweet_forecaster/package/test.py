from .train import get_train_state, train
from jax.random import PRNGKey
from jax import numpy as jnp
import jax

if __name__ == '__main__':
    # test that I can make a train state
    lag = 7
    state = get_train_state(PRNGKey(0), 10, lag)
    print(
        state.apply_fn(
            state.params,
            jnp.ones((1, 100, 10)),
            jnp.ones((1, lag + 1, 2))
        ).shape
    )

    # test that I can run a training loop
    X = jax.random.uniform(PRNGKey(1), (100, 100, 10))
    y_train = jax.random.uniform(PRNGKey(2), (100, 100, 1))
    y_test = jax.random.uniform(PRNGKey(2), (100, lag, 1))

    train(PRNGKey(3), X, y_train, y_test, epochs=5, batch_size=10)
