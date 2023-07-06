from jax import numpy as jnp
from jaxtyping import Array
import pandas as pd
from .train import get_model
from jax.scipy.stats.norm import cdf

def make_forecasts(
    state,
    X: Array,
    y: Array,
    y_test: Array
    ) -> pd.DataFrame:

    X_input = jnp.concatenate([X, y], axis=-1)
    y_input = jnp.concatenate([
        jnp.ones((X.shape[0], 1, 2)).at[:,0,0].set(y[:,-1,0]),
        y
    ])
    y_hat = get_model().apply(state.params, X_input, y_input)
    mean = y_hat[:,:,0]
    lower_ci = cdf(.1, loc=mean, scale=y_hat[:,:,1])
    upper_ci = cdf(.9, loc=mean, scale=y_hat[:,:,1])
    df = pd.DataFrame(
        y_hat = y_hat,
        lower_ci = lower_ci,
        upper_ci = upper_ci,
        y = y
    )
    print(df)
    return df
