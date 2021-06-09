# Import packages
import tensorflow as tf


class GRUEncoder(tf.keras.layers.Layer):
	"""Encoder module
	1. Takes a list of token IDs
	2. Looks up an embedding vector for each token
	3. Processes the embeddings into a new sequence
	4. Returns:
	    + The processed sequence. This will be passed to the attention head.
	    + The internal state. This will be used to initialize the decoder
	"""

	def __init__(self, input_vocab_size, embedding_dim, enc_units=512):
		"""Class constructor.

		Args:
			input_vocab_size (int): Vocabulary size of input language.
			embedding_dim (int): Embedding dimension.
			enc_units (int): Number of nodes for GRU layer.
		"""
		super(GRUEncoder, self).__init__()
		self.input_vocab_size = input_vocab_size
		self.embedding_dim = embedding_dim
		self.enc_units = enc_units
		# Embedding layer converts tokens to vectors
		self.embedding = tf.keras.layers.Embedding(
			self.input_vocab_size,
			embedding_dim)
		# GRU layer processes vectors sequentially
		self.gru_layer1 = tf.keras.layers.GRU(
			self.enc_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer="glorot_uniform")
		self.gru_layer2 = tf.keras.layers.GRU(
			self.enc_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer="glorot_uniform")

	def call(self, tokens, state=None):
		"""Forward pass over the layer.

		Args:
			tokens (tensor): Tokens IDs.
			state (tensor, optional): State of the previous layer. Defaults to None.

		Returns:
			tensor, tensor: Encoded output tensor, hidden state tensor.
		"""
		vectors = self.embedding(tokens)
		output, state = self.gru_layer1(vectors, initial_state=state)
		output, state = self.gru_layer2(output, initial_state=state)
		return output, state
