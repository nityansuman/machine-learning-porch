# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class ConvNet(tf.keras.Model):
	"""ConvNet image classifier implementation.

	Args:
		tf.keras.Model (cls): Parent model class.
	"""

	def __init__(self, num_classes, output_activation="softmax", **kwargs):
		"""Constructor.

		Args:
			num_classes (int): Number of target classes.
			output_activation (str, optional): Activation for the output layer.
				Set this according to `num_classes`. Defaults to `softmax`.
		"""
		super(ConvNet, self).__init__(**kwargs)
		self.conv_2d_1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu")
		self.conv_2d_2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu")
		self.max_pool = layers.MaxPool2D()
		self.flatten = layers.Flatten()
		self.drop_out = layers.Dropout(0.5)
		self.final_dense = layers.Dense(units=num_classes, activation=output_activation)

	def call(self, x):
		"""Foreward pass over the model.

		Args:
			x (tensor): Input tensor shape of [None, 32, 32].

		Returns:
			tensor: Tensor of shape [None, `num_classes`].
		"""
		x = self.conv_2d_1(x)
		x = self.max_pool(x)
		x = self.conv_2d_2(x)
		x = self.max_pool(x)
		x = self.flatten(x)
		x = self.drop_out(x)
		x = self.final_dense(x)
		return x
