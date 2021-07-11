# Import packages
import tensorflow as tf


class FeedForwardNetworkLayer(tf.keras.layers.Layer):
	"""Feed forward network layer implementation from transformer architecture.

	Args:
		units (int, optional): Number of output units. Defaults to 2048.
		activation (str, optional): Activation function to use. Defaults to "relu".
		initializer (str, optional): Weight initializer to use. Defaults to "glorot_uniform".
		bias_initializer (str, optional): Bias initialier to use. Defaults to "zeros".

	Input Shape:
		N-D tensor of shape `batch_size, ..., input_dim`. An example of a 2D input of shape
		`batch_size, input_dim`.

	Output Shape:
		N-D tensor of shape `batch_size, ..., units`. An example of a 2D output of shape
		`batch_size, units`.
	"""

	def __init__(self, units=2048, activation="relu", initializer="glorot_uniform",
		bias_initializer="zeros", **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.activation = tf.keras.activations.get(activation)
		self.initializer = tf.keras.initializers.get(initializer)
		self.b_init = tf.keras.initializers.get(bias_initializer)

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
			shape=(self.units, self.units),
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


# Test
if __name__ == "__main__":
	# Set environment
	batch_size = 16
	input_size = 128
	units = 512

	# Instantiate the layer
	layer_1 = FeedForwardNetworkLayer(units=units)

	# Build the layer and create weights
	y = layer_1(tf.ones(shape=(batch_size, input_size)))

	# Output shape should be (16, 512)
	assert y.shape == (batch_size, units)

	# All weights should be four
	assert len(layer_1.weights) == 4

	# All weights in the layer are trainable
	# Therefor trainable weights should be four as well
	assert len(layer_1.trainable_weights) == 4
