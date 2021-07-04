# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class LeNet(tf.keras.Model):
	"""LeNet image classifier implementation.

	Args:
		tf.keras.Model (cls): Parent model class.
	"""

	def __init__(self, num_classes, output_activation="softmax", **kwargs):
		"""Constructor.

		Args:
			num_classes (int): Number of target classes.
			output_activation (str, optional): Output activation. Defaults to `softmax`.
		"""
		super().__init__(**kwargs)
		self.conv_2d_1 = layers.Conv2D(filters=30, kernel_size=(5, 5), activation="relu")
		self.conv_2d_2 = layers.Conv2D(filters=15, kernel_size=(3, 3), activation="relu")
		self.max_pool = layers.MaxPool2D(pool_size=(2, 2))
		self.flatten = layers.Flatten()
		self.dropout = layers.Dropout(rate=0.5)
		self.dense_1 = layers.Dense(units=500, activation="relu")
		self.dense_2 = layers.Dense(units=num_classes, activation=output_activation)

	def call(self, x):
		"""Forward pass over the model.

		Args:
			x (tensor): Input tensor of shape [None, 28, 28, 1].

		Returns:
			tensor: Output tensor of shape [None, `num_classes`].
		"""
		x = self.conv_2d_1(x)
		x = self.max_pool(x)
		x = self.conv_2d_2(x)
		x = self.max_pool(x)
		x = self.flatten(x)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.dense_2(x)
		return x
