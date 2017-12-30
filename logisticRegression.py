import numpy as np
from skimage.transform import resize
from imageio import imread
import os

from sklearn.linear_model import SGDClassifier
from divideData import readCategories

def train_and_predict():
	classifier = SGDClassifier() # same as logistic regression but uses stochastic gradient descent
	dataPath = '../image_files/data_classify/'
	trainDataPath = dataPath+'train/'
	validationDataPath = dataPath+'validation/'
	resPaths = readCategories(dataPath)
	_, nb_validation_eczema, nb_train_eczema = resPaths['Eczema']
	_, nb_validation_psoriasis, nb_train_psoriasis = resPaths['Psoriasis']

	nb_train_samples = nb_train_eczema+nb_train_psoriasis
	nb_validation_samples = nb_validation_eczema+nb_validation_psoriasis

	images,labels = [],[]
	first = True
	size = 299
	batch_size = 100
	i = 0
	for name in os.listdir(trainDataPath):
		dirName = os.path.join(trainDataPath,name)
		for f in os.listdir(dirName):
			f = os.path.join(dirName,f)
			if name == 'Eczema':
				labels.append(0)
			else:
				labels.append(1)
			img = imread(f)
			img = resize(img,(size,size))
			img = img.flatten()
			images.append(img.tolist())
			if len(labels) == batch_size:
				if first:
					classifier.partial_fit(np.asarray(images),np.asarray(labels),classes=np.array([0,1]))
					first = False
				else:
					classifier.partial_fit(np.asarray(images),np.asarray(labels))
				i += 1
				print('Processed {} batches'.format(str(i)))
				images,labels = [],[]

	del images
	del labels
	print('Done fitting the model')

	print('Start computing score for validation set')
	images,labels = [],[]
	scores = []
	size = 299
	batch_size = 100
	i = 0
	for name in os.listdir(validationDataPath):
		dirName = os.path.join(validationDataPath,name)
		for f in os.listdir(dirName):
			f = os.path.join(dirName,f)
			if name == 'Eczema':
				labels.append(0)
			else:
				labels.append(1)
			img = imread(f)
			img = resize(img,(size,size))
			img = img.flatten()
			images.append(img.tolist())
			if len(labels) == batch_size:
				scores.append(classifier.score(np.asarray(images),np.asarray(labels)))
				i += 1
				print('Processed {} batches'.format(str(i)))
				images,labels = [],[]

	scores = np.asarray(scores)
	print(scores,scores.shape)

	del images
	del labels
	print('Done with validation data')
	print('\nModel got a validation accuracy of {}'.format(np.mean(scores)))

if __name__ == '__main__':
	train_and_predict()