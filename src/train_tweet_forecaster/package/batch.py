from jax import numpy as jnp
from typing import Tuple
from jaxtyping import Array

def windowing(
    train_set: Array,
    test_set: Array,
    lag: int
    ) -> Tuple[Array, Array, Array]:
    ''' turns a pair of timeseries into windowed batches for training'''
    max_length = train_set.shape[0] - lag
    n_features = train_set.shape[1]
    starts = jnp.arange(1, max_length)
    X = jnp.full((len(starts), max_length, n_features), -1)
    y_input = jnp.full((len(starts), max_length, 1), -1)
    y_output = jnp.zeros((len(starts), lag, 1))
    for i, start in enumerate(starts):
        for j in range(start):
            X = X.at[i, j, :n_features].set(train_set[j])
            y_input = y_input.at[i, j, 1].set(test_set[j])
        y_output = y_output.at[i].set(test_set[start:start+lag, jnp.newaxis])
    return X, y_input, y_output
