# 02456_Deep_learning_Project
The final project for the deep learning course.

## Usage

You can run either of the scripts train or predict to either train a new model or predict on some images using a already trained model.

### Training

The training script can be run by the following command

```
python3 retrain.py --batch_size=32 --epochs=100 --regularization_factor=0.03 --model_name="modelName" --createBottlenecks=True
```

### Prediction

To run the prediction script on some images use the following:

```
python3 predict.py --batch_size=32 --model_name="modelName" --predict_path='pathToImageFolders'
```
Note the script expects that the images are located in two folders one for each class where the folder name is the class. This is only used such that you can verify how well the model performs.

## Running the prediction on unlabeled images

To make the model predict on unlabeled images you should use the jupyter notebook that loads some images and then runs a prediction on them. You can give any folder you want to the script and it will run the prediction on the images in the folder and display them with the title corresponding to the prediction on the images.

## Built With

* [Keras](http://www.keras.io) - The neural network framework used

## Authors

* **Kim Rasmus Rylund** - [Rylund](https://github.com/Rylund)
* **Martin Hatting Petersen** - [mhattingpete](https://github.com/mhattingpete)

## Acknowledgments

* Inspiration of code: [Keras-transfer-learning](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* Another inspiration: [Tensorflow-transfer-learning](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)


