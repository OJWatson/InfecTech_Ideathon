from jax import numpy as jnp

def windowing(train_set, test_set, lag):
    ''' turns a pair of timeseries into windowed batches for training'''
    max_length = train_set.shape[0] - lag
    n_features = train_set.shape[1]
    starts = jnp.arange(1, max_length)
    X = jnp.full((len(starts), max_length, n_features), -1)
    y = jnp.zeros((len(starts), lag, 1))
    for i, start in enumerate(starts):
        for j in range(start):
            X = X.at[i, j].set(train_set[j])
        y = y.at[i].set(test_set[start:start+lag, jnp.newaxis])
    return X, y
