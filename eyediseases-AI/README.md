# eyediseases-AI
Image based deep learning to detect eye diseases. Transfer learning and feature extraction using keras, imagenet, and Inception v3.

There are two approaches covered here:
I. Transfer Learning Approach (94% accuracy, 100 hours of training duration, 500 epochs, 12 mins/epoch)
II. Feature Extraction & Bottleneck Approach (99.1% accuracy, 75 mins of training duration, 50 epochs, 90 sec/epoch)

### Highlights

* Extract features by feeding the images to an InceptionV3 model trained with imagenet dataset.
* Save training and validation features to h5 files.
* Create a small neural network model.
* Write generators to feed saved features to your model.
* Achieve 99.1% accuracy
* Training speed reduced to 1/10th. 12 mins/epoch to 1.5 min/epoch!


### Jupyter notebooks
* Features-Extract.ipynb - Extract features by feeding images through InceptionV3 pretrained model
* Features-Train.ipynb - Feed the features to train a small neural network with a classifier
* Features-Evaluate.ipynb - Evaluate the model along with occlusion maps

### Models
output/model.24-0.99.hdf5.zip:
* Model file created by Features-Train.ipynb.
* Unzip and load the model in Features-Evaluate.ipynb
