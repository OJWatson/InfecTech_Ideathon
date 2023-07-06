# Adapted from seq2seq training example:
# https://github.com/google/flax/blob/main/examples/seq2seq/train.py

from typing import Tuple, Dict, Any
from jaxtyping import Array
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import random
import optax
from .rnn import Seq2seq

def get_train_state(rng: Any, n_features: int, lag: int) -> train_state.TrainState:
    """Returns a train state."""
    model = Seq2seq(
        teacher_force = True,
        hidden_size = 128,
        eos_id = -1
    )
    params = model.init(
        rng,
        jnp.ones((1, 100, n_features)),
        jnp.ones((1, lag + 1, 2))
    )
    tx = optax.adam(1e-2)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state

def nll_loss(y_hat: Array, y: Array) -> Array:
    """Returns gauss negative log likelihood."""
    mu, sigma = y_hat[0,:], y_hat[1,:]
    return jnp.sum(jnp.log(sigma) / 2 + jnp.square(y - mu) / (2 * sigma))

def compute_metrics(y_hat: Array, y: Array) -> Dict[str, Array]:
    """Computes metrics and returns them."""
    loss = nll_loss(y_hat, y)
    mse = jnp.mean(jnp.square(y - y_hat))
    metrics = {
        'loss': loss,
        'mse': mse,
    }
    return metrics


@jax.jit
def train_step(
        state: train_state.TrainState,
        X: Array,
        y_input: Array,
        y_test: Array
    ) -> Tuple[train_state.TrainState, Dict[str, Array]]:
    """Trains one step."""

    def loss_fn(params):
        y_hat = state.apply_fn(
            params,
            X,
            y_input
        )
        loss = nll_loss(y_hat, y_test)
        return loss, y_hat

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, y_hat), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(y_hat, y_test)

    return state, metrics

def train(
    key: Any,
    X: Array,
    y_train: Array,
    y_test: Array,
    epochs: int = 100,
    batch_size = 100
    ):
    # get the lag
    lag = y_test.shape[1]

    # Create input sequences 
    X_input = jnp.concatenate([X, y_train], axis=-1)

    # Create output sequences 
    y_input = jnp.concatenate([y_train[:,-1:,:], y_test], axis=1)
    y_std = jnp.ones_like(y_input)
    y_input = jnp.concatenate([y_input, y_std], axis=-1)

    # Batch the sequences
    X_batched = jnp.split(X_input, batch_size)
    y_input_batched = jnp.split(y_input, batch_size)
    y_test_batched = jnp.split(y_test, batch_size)

    # Ceate the train state
    key, key_i = random.split(key)
    state = get_train_state(key_i, X_input.shape[-1], lag)

    # Train
    for step in range(epochs):
        key, key_i = random.split(key)
        for b in random.permutation(key_i, len(X_batched)):
            state, metrics = train_step(
                state,
                X_batched[b],
                y_input_batched[b],
                y_test_batched[b]
            )
        print(metrics)
    return state
