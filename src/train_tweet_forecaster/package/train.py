# Adapted from seq2seq training example:
# https://github.com/google/flax/blob/main/examples/seq2seq/train.py

from typing import Tuple, Dict, Any
from jaxtyping import Array
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from .rnn import Seq2seq

def get_train_state(rng: Any) -> train_state.TrainState:
    """Returns a train state."""
    model = Seq2seq(
        teacher_force = True,
        hidden_size = 10,
        eos_id = -1
    )
    params = model.init(
        rng,
        jnp.ones((1, 100, 10)),
        jnp.ones((1, 100, 1))
    )
    tx = optax.adam(1e-2)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state

def nll_loss(y_hat: Array, y: Array) -> float:
    """Returns gauss negative log likelihood."""
    mu, sigma = y_hat[0,:], y_hat[1,:]
    return jnp.log(sigma) / 2 + jnp.square(y - mu) / (2 * sigma)

def compute_metrics(y_hat: Array, y: Array) -> Dict[str, float]:
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
        batch: Array,
        eos_id: int
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Trains one step."""
    y = batch['answer'][:, 1:]

    def loss_fn(params):
        y_hat = state.apply_fn(
                params,
                batch['query'],
                batch['answer']
                )
        loss = nll_loss(y_hat, y)
        return loss, y_hat

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, y_hat), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(y_hat, y)

    return state, metrics

def train(key: Any, X: Array, y: Array, epochs: int = 100):
    state = get_train_state(key)
    for step in range(epochs):
        #TODO: minibatching
        state, metrics = train_step(state, X, y)
        print(metrics)
    return state
