# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class TextConvNet(tf.keras.Model):
	"""Example text classification using 1D convolution architecture implementation.

	Args:
		vocab_size (int): Size of the vocabulary.
		embedding_dim (int, optional): Embedding dimension. Defaults to 100.

	Input Shape:
		2D input tensor of shape `batch_size, input_size`.

	Output Shape:
		2D output tensor of shape `batch_size, 1`.
	"""

	def __init__(self, vocab_size, embedding_dim=100, **kwargs):
		super().__init__(**kwargs)
		self.embedding = layers.Embedding(vocab_size, embedding_dim)
		self.conv_1 = layers.Conv1D(
			filters=128,
			kernel_size=(7),
			padding="valid",
			activation="relu",
			strides=3
		)
		self.conv_2 = layers.Conv1D(
			filters=128,
			kernel_size=(7),
			padding="valid",
			activation="relu",
			strides=3
		)
		self.dropout = layers.Dropout(rate=0.4)
		self.global_pooling = layers.GlobalMaxPool1D()
		self.dense_1 = layers.Dense(512, activation="relu")
		self.dense_2 = layers.Dense(1, activation="sigmoid")

	def call(self, tokens):
		x = self.embedding(tokens)
		x = self.dropout(x)
		x = self.conv_1(x)
		x = self.conv_2(x)
		x = self.global_pooling(x)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.dense_2(x)
		return x


# Test
if __name__ == "__main__":
	# Set environment
	vocab_size = 10000
	embedding_dim = 100

	# Instantiate the model
	model = TextConvNet(vocab_size=vocab_size, embedding_dim=embedding_dim)
	print(model)
