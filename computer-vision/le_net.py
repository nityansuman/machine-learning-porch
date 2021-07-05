# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class LeNetBase(tf.keras.layers.Layer):
	def __init__(self, filters, kernel_size, pool_size, trainable=True, **kwargs):
		"""Constructor.

		Args:
			filters (int): Number of filters.
			kernel_size (Union[int, tuple]): Kernel size for convolutions.
			pool_size (Union[int, tuple]): Pool kernel size.
			trainable (bool, optional): Boolean flag to make layer trainable.
				Defaults to True.
		"""
		super().__init__(trainable=trainable, **kwargs)
		self.conv_2d = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu")
		self.max_pool = layers.MaxPool2D(pool_size=pool_size, strides=(2, 2), padding="valid")

	def call(self, x):
		"""Forward pass over the layer.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x = self.conv_2d(x)
		x = self.max_pool(x)
		return x


class LeNet(tf.keras.Model):
	"""LeNet is a convolutional neural network structure proposed by
	Yann LeCun et al. in 1989.
	"""

	def __init__(self, num_classes, output_activation="softmax", **kwargs):
		"""Constructor.

		Args:
			num_classes (int): Number of target classes.
			output_activation (str, optional): Output activation.
				Defaults to `softmax`.
		"""
		super().__init__(**kwargs)
		self.layer_1 = LeNetBase(filters=30,
								 kernel_size=(5, 5),
								 pool_size=(2, 2))
		self.layer_2 = LeNetBase(filters=15,
								 kernel_size=(3, 3),
								 pool_size=(2, 2))
		self.flatten = layers.Flatten()
		self.dropout = layers.Dropout(rate=0.5)
		self.dense_1 = layers.Dense(units=500, activation="relu")
		self.logit = layers.Dense(units=num_classes,
								  activation=output_activation)

	def call(self, x):
		"""Forward pass over the model.

		Args:
			x (tensor): Input tensor of shape [None, 28, 28, 1].

		Returns:
			tensor: Output tensor of shape [None, `num_classes`].
		"""
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.flatten(x)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.logit(x)
		return x
