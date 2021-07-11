# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class GRUEncoder(tf.keras.layers.Layer):
	"""GRU encoder layer implementation.

	Args:
		input_vocab_size (int): Vocabulary size of the input language.
		embedding_dim (int): Embedding dimension.
		units (int, optinal): Number of nodes. Defaults to 512.

	Input shape:
		2D input tensor of shape `batch_size, input_size`.

	Output shape:
		3D output tensor of shape `batch_size, input_size, units` and
		a 2D hidden state tensor of shape `batch_size, units`.
	"""

	def __init__(self, input_vocab_size, embedding_dim, units=512, **kwargs):
		super(GRUEncoder, self).__init__(name="GRUEncoder", **kwargs)
		self.embedding = layers.Embedding(
			input_vocab_size,
			embedding_dim
		)
		self.gru_layer_1 = layers.GRU(
			units,
			return_sequences=True,
			recurrent_initializer="glorot_uniform"
		)
		self.gru_layer_2 = layers.GRU(
			units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer="glorot_uniform"
		)

	def call(self, tokens):
		vectors = self.embedding(tokens)
		output = self.gru_layer_1(vectors)
		output, state = self.gru_layer_2(output)
		return output, state


# Test
if __name__ == "__main__":
	# Set environment
	batch_size = 16
	input_size = 128
	embedding_dim = 100
	input_vocab_size = 1000
	units = 512

	# Instantiate the layer
	layer_1 = GRUEncoder(
		input_vocab_size=input_vocab_size,
		embedding_dim=embedding_dim,
		units=units
	)

	# Build the layer and create weights
	y, states = layer_1(tf.ones(shape=(batch_size, input_size)))

	# Output shape should be (batch_size, input_size, units)
	if y.shape != (batch_size, input_size, units):
		raise AssertionError("Incorrect output shape.")

	# Output shape for the hidden state should be (batch_size, units)
	if states.shape != (batch_size, units):
		raise AssertionError("Incorrect hidden state shape.")
