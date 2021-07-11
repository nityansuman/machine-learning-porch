# Import packages
import tensorflow as tf
from tensorflow.python.keras import activations


class FeedForwardNetworkLayer(tf.keras.layers.Layer):
	"""Feed forward network layer implementation from transformer architecture.

	Args:
		units (int, optional): Number of output units. Defaults to 2048.
		activation (str, optional): Activation function to use. Defaults to "relu".

	Input shape:
		N-D tensor of shape `batch_size, ..., input_dim`. An example of a 2D input of shape
		`batch_size, input_dim`.

	Output Shape:
		N-D tensor of shape `batch_size, ..., units`. An example of a 2D output of shape
		`batch_size, units`.
	"""

	def __init__(self, units=2048, activation="relu", **kwargs):
		super().__init__(trainable=True, **kwargs)
		self.units = units
		self.activation = tf.keras.layers.Activation(activation=activation)
		self.initializer =tf.keras.initializers.GlorotUniform()
		self.b_init = tf.keras.initializers.RandomNormal()

	def build(self, input_shape):
		self.w1 = self.add_weight(
			name="w1",
			shape=(input_shape[-1], self.units),
			initializer=self.initializer,
			trainable=True
		)
		self.b1 = self.add_weight(
			name="b1",
			shape=(self.units,),
			initializer=self.b_init,
			trainable=True
		)
		self.w2 = self.add_weight(
			name="w2",
			shape=[self.units, self.units],
			initializer=self.initializer,
			trainable=True
		)
		self.b2 = self.add_weight(
			name="b2",
			shape=(self.units,),
			initializer=self.b_init,
			trainable=True
		)

	def call(self, inputs):
		x = tf.matmul(inputs, self.w1) + self.b1
		x = self.activation(x)
		return tf.matmul(x, self.w2) + self.b2
