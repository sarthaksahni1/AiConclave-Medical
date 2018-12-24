from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model 

classifier = Sequential()
classifier.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(384, 256, 3)))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.1))
classifier.add(GlobalAveragePooling2D())
classifier.add(Dense(3, activation='softmax'))
classifier.summary()
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.datasets import load_files   
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(data_path):
    data = load_files(data_path)
    img_files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 3)
    return img_files, targets

train_files, train_targets = load_dataset('dataset/train')
valid_files, valid_targets = load_dataset('dataset/valid')
test_files, test_targets = load_dataset('dataset/test', shuffle=False)

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(384, 256))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(image_paths):
    return np.vstack([path_to_tensor(path) for path in image_paths])

train_tensors  = paths_to_tensor(tqdm(train_files))
train_tensors = train_tensors.astype('float32') / 255
valid_tensors = paths_to_tensor(tqdm(valid_files))
valid_tensors = valid_tensors.astype('float32') / 255
test_tensors = paths_to_tensor(tqdm(test_files))
test_tensors = test_tensors.astype('float32') / 255
print(train_tensors.shape)

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("model_without_transfer_learning.h5", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
classifier.fit_generator(train_tensors, train_targets, epochs = 100, validation_data=(valid_tensors, valid_targets), callbacks = [checkpoint], shuffle=False)