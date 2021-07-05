# Import packages
import tensorflow as tf
from tensorflow.keras import layers


class AlexNetBase(tf.keras.layers.Layer):
	def __init__(
		self, filters, kernal_size, strides, padding,
		pool_size, pool_stride, pool_padding, activation="relu",
		trainable=True, **kwargs):
		"""Constructor.

		Args:
			filters (int): Number of filters
			kernal_size (Union[int, tuple]): Kernel size for convolution.
			strides (Union[int, tuple]): Kernel stride size.
			padding (str): Padding strategy.
			pool_size (Union[int, tuple]): Pool kernel size.
			pool_stride (Union[int, tuple]): Pool stride size.
			pool_padding (str): Pool padding strategy.
			activation (str, optional): Layer activation. Defaults to "relu".
			trainable (bool, optional): Boolean to make layer trainable.
				Defaults to True.
		"""
		super().__init__(trainable=trainable, **kwargs)
		self.conv_2d = layers.Conv2D(
			filters=filters,
			kernel_size=kernal_size,
			strides=strides,
			padding=padding,
			activation=activation)
		self.max_pool = layers.MaxPool2D(
			pool_size=pool_size,
			strides=pool_stride,
			padding=pool_padding)
		self.batch_norm = layers.BatchNormalization()

	def call(self, x):
		"""Forward pass over the layer.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x = self.conv_2d(x)
		x = self.batch_norm(x)
		x = self.max_pool(x)
		return x


class DenseBlock(tf.keras.layers.Layer):
	def __init__(self,
				 units,
				 num_classes,
				 output_activation,
				 activation="relu",
				 dp_rate=0.4,
				 trainable=True,
				 **kwargs):
		"""Constructor.

		Args:
			units (int): Number of hidden units.
			num_classes (int): Number of target classes.
			output_activation (str): Output activation.
			activation (str, optional): Layer activation. Defaults to "relu".
			dp_rate (float, optional): Dropout rate. Defaults to 0.4.
			trainable (bool, optional): Boolean to make layer trainable.
				Defaults to True.
		"""
		super().__init__(trainable=trainable, **kwargs)
		self.flatten = layers.Flatten()
		self.dense_1 = layers.Dense(units=units, activation=activation)
		self.dense_2 = layers.Dense(units=units, activation=activation)
		self.dropout = layers.Dropout(rate=dp_rate)
		self.logit = layers.Dense(
			units=num_classes,
			activation=output_activation)

	def call(self, x):
		"""Forward pass over the layer.

		Args:
			x (tensor): Input tensor.

		Returns:
			tensor: Output tensor.
		"""
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.dense_2(x)
		x = self.logit(x)
		return x


class AlexNet(tf.keras.Model):
	"""AlexNet model architecture implementation.
	"""

	def __init__(self, num_classes, output_activation="softmax", **kwargs):
		"""Constructor.

		Args:
			num_classes (int): Number of target classes.
			output_activation (str, optional): Output layer activation.
				Defaults to "softmax".
		"""
		super().__init__(**kwargs)
		self.block_1 = AlexNetBase(
			filters=96,
			kernel_size=(11, 11),
			strides=(4, 4),
			padding="same",
			pool_size=(3, 3),
			pool_stride=(2, 2),
			pool_padding="valid",
			activation="relu")
		self.block_2 = AlexNetBase(
			filters=256,
			kernel_size=(5, 5),
			strides=(1, 1),
			padding="same",
			pool_size=(3, 3),
			pool_stride=(2, 2),
			pool_padding="valid",
			activation="relu")
		self.cnn_1 = layers.Conv2D(
			filters=384,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding="same",
			activation="relu")
		self.cnn_2 = layers.Conv2D(
			filters=384,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding="same",
			activation="relu")
		self.cnn_3 = layers.Conv2D(
			filters=256,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding="same",
			activation="relu")
		self.max_pool = layers.MaxPool2D(
			pool_size=(3, 3),
			strides=(2, 2),
			padding="valid")
		self.classifier = DenseBlock(
			units=4096,
			num_classes=num_classes,
			output_activation=output_activation,
			activation="relu",
			dp_rate=0.4)

	def call(self, x):
		"""Forward pass over the model.

		Args:
			x (tensor): Input tensor of shape [None, 227, 227, 3].

		Returns:
			tensor: Output tensor of shape [None, `num_classes`].
		"""
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.cnn_1(x)
		x = self.cnn_2(x)
		x = self.cnn_3(x)
		x = self.max_pool(x)
		x = self.classifier(x)
		return x
