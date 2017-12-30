from keras.applications import InceptionResNetV2
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Dropout, BatchNormalization, Flatten, Dense, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, Add

import os

class ImageModel:
	def __init__(self,regularization_factor=0.01,dropout_factor=0.5,pooling=None):
		self.regularization_factor = regularization_factor
		self.dropout_factor = dropout_factor
		self.pooling = pooling

	def defineTopModel(self,input_shape,display=False):
		x = Input(input_shape)
		z = AveragePooling2D(input_shape=input_shape,pool_size=(8,8), strides=(1,1), padding='valid')(x)
		z = Flatten()(z)
		z = Dense(64, activation='relu',kernel_regularizer=l2(self.regularization_factor),kernel_initializer='truncated_normal')(z)
		z = Dropout(self.dropout_factor)(z)
		z = BatchNormalization()(z)
		z = Dense(32, activation='relu',kernel_regularizer=l2(self.regularization_factor),kernel_initializer='truncated_normal')(z)
		z = Dropout(self.dropout_factor)(z)
		z = BatchNormalization()(z)
		y = Dense(1, activation='sigmoid',kernel_regularizer=l2(self.regularization_factor),kernel_initializer='truncated_normal')(z)

		self.top_model = Model(inputs=x,outputs=y)
		if display: 
			self.top_model.summary()

	def defineBaseModel(self,input_shape):
		self.base_model = InceptionResNetV2(weights='imagenet', include_top=False,input_shape=input_shape,pooling=self.pooling)

	def defineFullModel(self,input_shape,top_model_input_shape,model_name):
		self.defineTopModel(top_model_input_shape)

		# Load weights from latest checkpoint if one exists
		if model_name:
			checkpoint_path = 'checkpoints/weights.{}.hdf5'.format(model_name)
			if not os.path.exists('checkpoints/'):
				raise Exception('Checkpoint not found')
			if os.path.exists(checkpoint_path):
				print('Resuming training from checkpoint: {}'.format(checkpoint_path))
				self.top_model.load_weights(checkpoint_path)

		self.defineBaseModel(input_shape)
		out = self.top_model(self.base_model.output)

		self.model = Model(inputs=self.base_model.input,outputs=out)