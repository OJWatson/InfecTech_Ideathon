# Adapted from flax seq2seq examples:
# https://github.com/google/flax/blob/main/examples/seq2seq/

import functools
from typing import Any, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
PRNGKey = jax.random.KeyArray
LSTMCarry = Tuple[Array, Array]

class DecoderLSTMCell(nn.RNNCellBase):
	"""DecoderLSTM Module wrapped in a lifted scan transform.

  Attributes:
	teacher_force: See docstring on Seq2seq module.
	feature_size: Feature size of the output sequence
  """
  features: int
  teacher_force: bool
  feature_size: int

  @nn.compact
  def __call__(
		  self,
          carry: Tuple[LSTMCarry, Array],
          x: Array
		  ) -> Tuple[LSTMCarry, Array]:
	  """Applies the DecoderLSTM model."""
	lstm_state, last_prediction = carry
	if not self.teacher_force:
		x = last_prediction
	lstm_state, y = nn.LSTMCell(self.features)(lstm_state, x)
	predictions = nn.Dense(features=self.feature_size)(y)
	return lstm_state, predictions

	@property
	def num_feature_axes(self) -> int:
		return 1


class Seq2seq(nn.Module):
    """Sequence-to-sequence class using encoder/decoder architecture.

    Attributes:
    teacher_force: whether to use `decoder_inputs` as input to the decoder at
      every step. If False, only the first input (i.e., the "=" token) is used,
      followed by samples taken from the previous output logits.
    hidden_size: int, the number of hidden dimensions in the encoder and decoder
      LSTMs.

    lag: int, the number of timesteps to predict
    """
    teacher_force: bool
    hidden_size: int
    lag: int

    @nn.compact
    def __call__(self, encoder_inputs: Array,
               decoder_inputs: Array) -> Tuple[Array, Array]:
          """Applies the seq2seq model.

        Args:
          encoder_inputs: [batch_size, max_input_length, vocab_size].
            padded batch of input sequences to encode.
          decoder_inputs: [batch_size, max_output_length, vocab_size].
            padded batch of expected decoded sequences for teacher forcing.
            When sampling (i.e., `teacher_force = False`), only the first token is
            input into the decoder (which is the token "="), and samples are used
            for the following inputs. The second dimension of this tensor determines
            how many steps will be decoded, regardless of the value of
            `teacher_force`.

        Returns:
          predictions, an array of length `batch_size`
          containing the predicted mean and variance of the output sequence
        """
        # Encode inputs.
        encoder = nn.RNN(
                nn.LSTMCell(self.hidden_size),
                return_carry=True,
                name='encoder'
                )
        decoder = nn.RNN(
                DecoderLSTMCell(
                    decoder_inputs.shape[-1],
                    self.teacher_force,
                    2
                    ),
                name='decoder'
                )

        seq_lengths = self.get_seq_lengths(encoder_inputs)

        encoder_state, _ = encoder(encoder_inputs, seq_lengths=seq_lengths)
        _, predictions = decoder(
                decoder_inputs[:, :-1],
                initial_carry=(encoder_state, decoder_inputs[:, 0])
                )

        return predictions

    def get_seq_lengths(self, inputs: Array) -> Array:
        """Get segmentation mask for inputs."""
        # undo one-hot encoding
        inputs = jnp.argmax(inputs, axis=-1)
        # calculate sequence lengths
        seq_lengths = jnp.argmax(inputs == self.eos_id, axis=-1)

        return seq_lengths
