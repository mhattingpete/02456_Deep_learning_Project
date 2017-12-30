from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, History, LambdaCallback
from argparse import ArgumentParser

import numpy as np
import os
import matplotlib.pyplot as plt

# local imports
from divideData import DivideData, readCategories
from model import ImageModel

parser = ArgumentParser()
parser.add_argument("--createBottlenecks", help="Create bottlenecks or use the already created bottlenecks. \
	If there is no bottlenecks found the script will create bottlenecks anyway",type=bool,default=False)
parser.add_argument("--model_name", help="Name to save model weihgts and logs as",type=str,default=None)
parser.add_argument("--batch_size", help="Batch size",type=int,default=32)
parser.add_argument("--epochs", help="Number of epochs",type=int,default=100)
parser.add_argument("--learningRate",help="Learning rate",type=float,default=1e-4)
parser.add_argument("--regularization_factor", help="Regularization factor, a number between 0 and 1",type=float,default=0.01)
parser.add_argument("--dropout_factor", help="Dropout factor, a number between 0 and 1",type=float,default=0.5)
parser.add_argument("--applyPooling",help="Which pooling to apply to base_model (either None, avg or max)",type=str,default=None)
args = parser.parse_args()

class ImageClassifier:
	def __init__(self,batch_size=32,createBottlenecks=False,learningRate=1e-4,regularization_factor=0.01,dropout_factor=0.5,pooling=''):
		# dimensions of our images.
		self.pooling = pooling
		self.img_width, self.img_height = 299, 299
		self.top_model_input_shape = None
		if self.img_width == 150 and self.img_height == 150:
			self.top_model_input_shape = (3,3,1536)
			self.bottleneckspath = './bottlenecks/no_pooling_150/'
		elif self.img_width == 299 and self.img_height == 299:
			self.top_model_input_shape = (8,8,1536)
			self.bottleneckspath = './bottlenecks/no_pooling_299/'
		if self.pooling == 'avg':
			self.top_model_input_shape = (1536,)
			self.bottleneckspath = './bottlenecks/avg_pooling_299/'

		self.originalDataPath = '../image_files/image_data'
		self.dividedDataPath = '../image_files/data_classify'

		self.batch_size = batch_size
		self.regularization_factor = regularization_factor
		self.dropout_factor = dropout_factor
		self.learningRate = learningRate

		self.imageModel = ImageModel(regularization_factor=self.regularization_factor,dropout_factor=self.dropout_factor,pooling=self.pooling)

	def defineCallbacks(self,epochs,model_name):
		history = History()
		callbacks = []
		plot_progress_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: self.plot_progress(epoch,history))
		callbacks.append(history)
		callbacks.append(plot_progress_cb)
		callbacks.append(EarlyStopping(monitor='val_loss',patience=int(max([epochs*0.2,6])),mode='auto',min_delta=0.01,verbose=2))
		callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(max([epochs*0.05,2])), verbose=2, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0))
		if model_name:
			checkpoint_path = 'checkpoints/weights.{}.hdf5'.format(model_name)
			callbacks.append(ModelCheckpoint(checkpoint_path,
				save_weights_only=True,
				save_best_only=True))
			log_path = './logs/{}'.format(model_name)
			callbacks.append(TensorBoard(log_dir=log_path, write_graph=False))
		return callbacks
		
	def retrain(self,epochs=50,model_name=None):
		if not os.path.isfile(self.bottleneckspath+'bottleneck_train.npy') or not os.path.isfile(self.bottleneckspath+'bottleneck_validation.npy'):
			createBottlenecks = True
		if createBottlenecks == True:
			self.saveBottleneck()

		resPaths = readCategories(self.dividedDataPath)
		self.nb_test_eczema, self.nb_validation_eczema, self.nb_train_eczema = resPaths['Eczema']
		self.nb_test_psoriasis, self.nb_validation_psoriasis, self.nb_train_psoriasis = resPaths['Psoriasis']

		train_data = np.load(self.bottleneckspath+'bottleneck_train.npy')
		train_labels = np.array([0] * (self.nb_train_eczema) + [1] * (train_data.shape[0]-self.nb_train_eczema))

		validation_data = np.load(self.bottleneckspath+'bottleneck_validation.npy')
		validation_labels = np.array([0] * (self.nb_validation_eczema) + [1] * (validation_data.shape[0]-self.nb_validation_eczema))

		test_data = np.load(self.bottleneckspath+'bottleneck_test.npy')
		test_labels = np.array([0] * (self.nb_test_eczema) + [1] * (test_data.shape[0]-self.nb_test_eczema))

		# build a classifier model to put on top of the convolutional model
		self.imageModel.defineTopModel(input_shape=self.top_model_input_shape,display=True)

		# Load weights from latest checkpoint if one exists
		if model_name:
			checkpoint_path = 'checkpoints/weights.{}.hdf5'.format(model_name)
			if not os.path.exists('checkpoints/'):
				os.makedirs('checkpoints/')
			if os.path.exists(checkpoint_path):
				print('Resuming training from checkpoint: {}'.format(checkpoint_path))
				self.imageModel.top_model.load_weights(checkpoint_path)

		# compile the model with a Adam optimizer
		# and a very slow learning rate.
		self.imageModel.top_model.compile(loss='binary_crossentropy',
		              optimizer=Adam(lr=self.learningRate),
		              metrics=['accuracy'])

		# Add callbacks for training
		callbacks = self.defineCallbacks(epochs,model_name)

		print('\nStart fitting the top layers')
		self.imageModel.top_model.fit(train_data,train_labels,
			epochs=epochs,batch_size=self.batch_size,
			validation_data=(validation_data,validation_labels),
			callbacks=callbacks,
			verbose=2)

		print('Ended fitting top layers')
		loss,accuracy = self.imageModel.top_model.evaluate(x=validation_data, y=validation_labels, batch_size=self.batch_size,verbose=0)
		print('Model got a validation loss of {} and accuracy {}'.format(*(round(loss,4),round(accuracy,4))))

	def retrainV2(self,epochs=50,model_name=None):
		validatePercent = 20
		testPercent = 20
		DivideData(self.originalDataPath, self.dividedDataPath, validatePercent, testPercent, self.batch_size)
		self.train_data_dir = self.dividedDataPath+'/train'
		self.validation_data_dir = self.dividedDataPath+'/validation'
		self.test_data_dir = self.dividedDataPath+'/test'

		resPaths = readCategories(self.dividedDataPath)
		self.nb_test_eczema, self.nb_validation_eczema, self.nb_train_eczema = resPaths['Eczema']
		self.nb_test_psoriasis, self.nb_validation_psoriasis, self.nb_train_psoriasis = resPaths['Psoriasis']

		self.nb_train_samples = self.nb_train_eczema+self.nb_train_psoriasis
		self.nb_validation_samples = self.nb_validation_eczema+self.nb_validation_psoriasis
		self.nb_test_samples = self.nb_test_eczema+self.nb_test_psoriasis

		self.imageModel.defineFullModel((self.img_height, self.img_width, 3),self.top_model_input_shape,model_name)

		# prepare data augmentation configuration
		datagen = ImageDataGenerator(
		    rescale=1. / 255,
		    shear_range=0.4,
		    zoom_range=0.6,
		    horizontal_flip=True)

		valid_datagen = ImageDataGenerator(rescale=1. / 255)

		train_generator = datagen.flow_from_directory(
		    self.train_data_dir,
		    target_size=(self.img_height, self.img_width),
		    batch_size=self.batch_size,
		    class_mode='binary',
		    shuffle=False)
		
		validation_generator = valid_datagen.flow_from_directory(
		    self.validation_data_dir,
		    target_size=(self.img_height, self.img_width),
		    batch_size=self.batch_size,
		    class_mode='binary',
		    shuffle=False)

		# compile the model with a Adam optimizer
		# and a very slow learning rate.
		for layer in self.imageModel.model.layers[:15]:
			layer.trainable = False

		self.imageModel.model.compile(loss='binary_crossentropy',
		              optimizer=Adam(lr=self.learningRate),
		              metrics=['accuracy'])

		# Add callbacks for training
		callbacks = self.defineCallbacks(epochs,model_name)

		print('\nStart fitting the model')
		self.imageModel.model.fit_generator(train_generator,
			steps_per_epoch=self.nb_train_samples/self.batch_size,
			epochs=epochs,
			validation_data=validation_generator,
			validation_steps=self.nb_validation_samples/self.batch_size,
			callbacks=callbacks,
			verbose=2)

		del train_generator
		del validation_generator

		print('Ended fitting model')

	def saveBottleneck(self):
		validatePercent = 20
		testPercent = 20
		DivideData(self.originalDataPath, self.dividedDataPath, validatePercent, testPercent, self.batch_size)
		self.train_data_dir = self.dividedDataPath+'/train'
		self.validation_data_dir = self.dividedDataPath+'/validation'
		self.test_data_dir = self.dividedDataPath+'/test'

		resPaths = readCategories(self.dividedDataPath)
		self.nb_test_eczema, self.nb_validation_eczema, self.nb_train_eczema = resPaths['Eczema']
		self.nb_test_psoriasis, self.nb_validation_psoriasis, self.nb_train_psoriasis = resPaths['Psoriasis']

		self.nb_train_samples = self.nb_train_eczema+self.nb_train_psoriasis
		self.nb_validation_samples = self.nb_validation_eczema+self.nb_validation_psoriasis
		self.nb_test_samples = self.nb_test_eczema+self.nb_test_psoriasis

		self.imageModel.defineBaseModel(input_shape=(self.img_height,self.img_width,3))

		# prepare data augmentation configuration
		datagen = ImageDataGenerator(
		    rescale=1. / 255,
		    shear_range=0.4,
		    zoom_range=0.6,
		    horizontal_flip=True)

		valid_datagen = ImageDataGenerator(rescale=1. / 255)

		train_generator = datagen.flow_from_directory(
		    self.train_data_dir,
		    target_size=(self.img_height, self.img_width),
		    batch_size=self.batch_size,
		    class_mode='binary',
		    shuffle=False)

		bottleneck_train = self.imageModel.base_model.predict_generator(train_generator, steps=self.nb_train_samples // self.batch_size, verbose=1)
		np.save(self.bottleneckspath+'bottleneck_train',bottleneck_train)

		print('Done creating bottlenecks for training set\n')
		del bottleneck_train
		del train_generator
		
		validation_generator = valid_datagen.flow_from_directory(
		    self.validation_data_dir,
		    target_size=(self.img_height, self.img_width),
		    batch_size=self.batch_size,
		    class_mode='binary',
		    shuffle=False)

		bottleneck_valid = self.imageModel.base_model.predict_generator(validation_generator, steps=self.nb_validation_samples // self.batch_size, verbose=1)
		np.save(self.bottleneckspath+'bottleneck_validation',bottleneck_valid)

		print('Done creating bottlenecks for validation set\n')
		del bottleneck_valid
		del validation_generator

		test_generator = valid_datagen.flow_from_directory(
		    self.test_data_dir,
		    target_size=(self.img_height, self.img_width),
		    batch_size=self.batch_size,
		    class_mode='binary',
		    shuffle=False)

		bottleneck_test = self.imageModel.base_model.predict_generator(test_generator, steps=self.nb_test_samples // self.batch_size, verbose=1)
		np.save(self.bottleneckspath+'bottleneck_test',bottleneck_test)

		print('Done creating bottlenecks for validation set\n')
		del bottleneck_test
		del test_generator

	def plot_progress(self,epoch,history):
		plt.figure()
		locpath=''
		plt.plot(range(epoch+1),history.history['loss'],'b',label='training loss')
		plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
		plt.axis([0, epoch+1, 0, 2])
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.title('training error')
		plt.legend(loc='best')
		plt.savefig(locpath+'training_error.png')
		plt.close('all')

if __name__ == '__main__':
	classifier = ImageClassifier(batch_size=args.batch_size,createBottlenecks=args.createBottlenecks,learningRate=args.learningRate,
		regularization_factor=args.regularization_factor,dropout_factor=args.dropout_factor,pooling=args.applyPooling)
	classifier.retrain(epochs=args.epochs,model_name=args.model_name)