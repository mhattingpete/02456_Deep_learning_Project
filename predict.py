from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from argparse import ArgumentParser

import numpy as np
import os
import matplotlib.pyplot as plt

# local imports
from divideData import doRandomRequest, readCategories
from model import ImageModel

parser = ArgumentParser()
parser.add_argument("--model_name", help="Name to save model weihgts and logs as",type=str,default=None)
parser.add_argument("--predict_path", help="Where to find images to predict for",type=str,default=None)
parser.add_argument("--batch_size", help="Batch size",type=int,default=32)
parser.add_argument("--applyPooling",help="Which pooling to apply to base_model (either None, avg or max)",type=str,default=None)
args = parser.parse_args()

class ImageClassifier:
	def __init__(self,batch_size=32,pooling=''):
		# dimensions of our images.
		self.img_width, self.img_height = 299, 299

		if self.img_width == 150 and self.img_height == 150:
			self.top_model_input_shape = (3,3,1536)
			self.bottleneckspath = './bottlenecks/no_pooling_150/'
		elif self.img_width == 299 and self.img_height == 299:
			self.top_model_input_shape = (8,8,1536)
			self.bottleneckspath = './bottlenecks/no_pooling_299/'
		elif self.pooling:
			self.top_model_input_shape = (1536,)
			self.bottleneckspath = './bottlenecks/avg_pooling_299/'

		self.originalDataPath = '../image_files/image_data'
		self.dividedDataPath = '../image_files/data_classify'

		self.batch_size = batch_size
		self.imageModel = ImageModel()

	def predict(self,path,model_name):
		predict_path = './predictedImages'
		doRandomRequest(self.dividedDataPath+'/test',path,predict_path) # finds 20 images from each of the test set and do a visual prediction on them
		labels = []
		images = []
		predictions = []
		classNames = ['Eczema','Psoriasis']
		num_images = len([f for f in os.listdir(path+'/'+classNames[0])]) + len([f for f in os.listdir(path+'/'+classNames[1])])

		valid_datagen = ImageDataGenerator(rescale=1. / 255)
		test_generator = valid_datagen.flow_from_directory(
		    path,
		    target_size=(self.img_height, self.img_width),
		    batch_size=num_images,
		    class_mode='binary',
		    shuffle=False)

		# build a classifier model to put on top of the convolutional model
		self.imageModel.defineFullModel((self.img_height, self.img_width, 3),self.top_model_input_shape,model_name)

		batches = 0
		for x_batch, y_batch in test_generator:
			p = self.imageModel.model.predict(x_batch)
			#for i in range(len(p)):
			predictions.append(p)
			images.append(x_batch)
			labels.append(y_batch)
			batches += 1
			if batches >= num_images / self.batch_size:
				break
		predictions = np.asarray(predictions).reshape((-1))
		labels = np.asarray(labels,dtype='int32').reshape((-1))
		images = np.asarray(images).reshape((-1,self.img_height, self.img_width, 3))

		self.convertPredToImages(images,predictions,labels,classNames,predict_path)
		print('\nDone predicting on images')

		resPaths = readCategories(self.dividedDataPath)
		self.nb_test_eczema, self.nb_validation_eczema, self.nb_train_eczema = resPaths['Eczema']
		self.nb_test_psoriasis, self.nb_validation_psoriasis, self.nb_train_psoriasis = resPaths['Psoriasis']

		test_data = np.load(self.bottleneckspath+'bottleneck_test.npy')
		test_labels = np.array([0] * (self.nb_test_eczema) + [1] * (test_data.shape[0]-self.nb_test_eczema))

		self.imageModel.top_model.compile(loss='binary_crossentropy',
		              optimizer=Adam(),
		              metrics=['accuracy'])

		loss,accuracy = self.imageModel.top_model.evaluate(x=test_data, y=test_labels, batch_size=self.batch_size,verbose=0)
		print('Model got a test loss of {} and accuracy {}'.format(*(round(loss,4),round(accuracy,4))))

	def convertPredToImages(self,images,predictions,labels,classNames,predict_path):
		for i,pred in enumerate(predictions):
			p = int(round(pred))
			predClass = 'Pred: '+classNames[p]+' True: '+classNames[labels[i]]
			filename = 't_'+classNames[labels[i]]+'_p_'+classNames[p]+'_'+str(i)
			self.plot_prediction(images[i],filename,predClass,predict_path)

	def plot_prediction(self,image,filename,title,predict_path):
		plt.figure()
		plt.imshow(image)
		plt.title(title)
		plt.savefig(predict_path+'/'+filename+'.png')
		plt.close('all')

if __name__ == '__main__':
	classifier = ImageClassifier(batch_size=args.batch_size,pooling=args.applyPooling)
	classifier.predict(path=args.predict_path,model_name=args.model_name)