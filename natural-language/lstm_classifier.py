# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class LstmClassifier(tf.keras.Model):
	"""Example text classification using bidirectional long short term memory implementation.

	Args:
		vocab_size (int): Size of the vocabulary.
		embedding_dim (int, optional): Embedding dimension. Defaults to 100.
		units (int, optional): Number of units in sequential proessing layer. Defaults to 128.

	Input Shape:
		2D input tensor of shape `batch_size, input_size`.

	Output Shape:
		2D output tensor of shape `batch_size, 1`.
	"""

	def __init__(self, vocab_size, embedding_dim=100, units=128, **kwargs):
		super().__init__(**kwargs)
		self.embedding = layers.Embedding(vocab_size, embedding_dim)
		self.lstm1 = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
		self.lstm2 = layers.Bidirectional(layers.LSTM(units))
		self.dense1 = layers.Dense(1, activation="sigmoid")

	def call(self, tokens):
		x = self.embedding(tokens)
		x = self.lstm1(x)
		x = self.lstm2(x)
		x = self.dense1(x)
		return tokens


# Test
if __name__ == "__main__":
	# Set environment
	vocab_size = 10000
	embedding_dim = 100

	# Instantiate the model
	model = LstmClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim)
	print(model)
