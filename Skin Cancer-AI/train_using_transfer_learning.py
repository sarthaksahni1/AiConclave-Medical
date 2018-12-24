from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 3)
    print(files, targets)
    return files, targets

train_files, train_targets = load_dataset('dataset/train')
valid_files, valid_targets = load_dataset('dataset/valid')
test_files, test_targets = load_dataset('dataset/test')
# test_files, test_targets = valid_files, valid_targets
# train_files, train_targets = valid_files, valid_targets

print('train_files size: {}'.format(len(train_files)))
print('train_files shape: {}'.format(train_files.shape))
print('target shape: {}'.format(train_targets.shape))

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(384, 256))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(image_paths):
    return np.vstack([path_to_tensor(path) for path in image_paths])

train_tensors = paths_to_tensor(tqdm(train_files))
valid_tensors = paths_to_tensor(tqdm(valid_files))
test_tensors = paths_to_tensor(tqdm(test_files))
print(train_tensors.shape)

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

apply_train_image_transform = False
if apply_train_image_transform:
    datagen_train = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
    datagen_train.fit(train_tensors)
    shape = (train_tensors.shape[0] * 2,) + train_tensors.shape[1:]
    generated = np.ndarray(shape=shape)
    for i, image in tqdm(enumerate(train_tensors)):
        generated[i] = datagen_train.random_transform(image)
    train_tensors = np.concatenate((train_tensors, generated))
    train_targets = train_targets.repeat(2, axis=0)
    
from keras.applications.inception_resnet_v2 import preprocess_input
train_imgs_preprocess = preprocess_input(train_tensors)
valid_imgs_preprocess = preprocess_input(valid_tensors)
test_imgs_preprocess = preprocess_input(test_tensors)
del train_tensors, valid_tensors, test_tensors

from keras.applications.inception_resnet_v2 import InceptionResNetV2
transfer_model = InceptionResNetV2(include_top=False)
train_data = transfer_model.predict(train_imgs_preprocess)
valid_data = transfer_model.predict(valid_imgs_preprocess)
test_data = transfer_model.predict(test_imgs_preprocess)
del train_imgs_preprocess, valid_imgs_preprocess, test_imgs_preprocess
print(train_data.shape)

from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential

my_model = Sequential()
my_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
my_model.add(Dropout(0.2))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.2))
my_model.add(Dense(3, activation='softmax'))
my_model.summary()

my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpoint_filepath = 'weights.best.my.hdf5'
my_checkpointer = ModelCheckpoint(filepath=checkpoint_filepath,verbose=1, save_best_only=True)
my_model.fit(train_data, train_targets, validation_data=(valid_data, valid_targets),epochs=60, batch_size=200, callbacks[my_checkpointer], verbose=1)