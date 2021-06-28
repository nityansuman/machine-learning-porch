# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class GRUEncoder(tf.keras.layers.Layer):
	"""GRU based encoder layer.

	Reference:
		- Ts: Maximum input sequence length.
		- Te: Number of units in encoder layer.
	"""

	def __init__(self, input_vocab_size, embedding_dim, units=512, **kwargs):
		"""Constructor.

		Args:
			input_vocab_size (int): Vocabulary size of the input language.
			embedding_dim (int): Embedding dimension.
			units (int): Number of nodes. Defaults to 512.
		"""
		super(GRUEncoder, self).__init__(name="GRUEncoder", **kwargs)
		self.input_vocab_size = input_vocab_size
		self.embedding_dim = embedding_dim
		self.units = units

		# Embedding layer converts tokens to vectors
		self.embedding = layers.Embedding(
			self.input_vocab_size,
			embedding_dim)

		# GRU layer processes vectors sequentially
		self.gru_layer = layers.GRU(
			self.units,
			return_sequences=True,
			recurrent_initializer="glorot_uniform")

		self.gru_layer_final = layers.GRU(
			self.units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer="glorot_uniform")

	def call(self, tokens):
		"""Forward pass over the encoder layer.

		Args:
			tokens (tensor): Tokens tensor of shape [None, Ts].

		Returns:
			tensor, tensor: Encoded output tensor of shape [None, Ts, Te]
			and final state tensor of shape [None, Te].
		"""
		# Lookup embeddings
		vectors = self.embedding(tokens)

		# Process vectors sequentially with rnn
		output = self.gru_layer(vectors)

		# Process sequence using another rnn
		output, state = self.gru_layer_final(output)

		# Return encoded sequence and final state
		return output, state
