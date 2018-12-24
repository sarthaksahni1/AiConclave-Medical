from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    data = load_files(path)
    print('Done Loading files...')
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 3)
    return files, targets

train_files, train_targets = load_dataset('dataset/train')
valid_files, valid_targets = load_dataset('dataset/valid')
test_files, test_targets = load_dataset('dataset/test')

from keras.preprocessing import image
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(384, 256))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def paths_to_tensor(image_paths):
    return np.vstack([path_to_tensor(path) for path in image_paths])

test_tensors = paths_to_tensor(tqdm(test_files))
test_tensors = test_tensors.astype('float32') / 255

from keras.models import load_model
model = load_model('model_without_transfer_learning.h5')
my_predictions = [model.predict(np.expand_dims(feature, axis=0)) for feature in test_tensors]
test_accuracy = 100 * np.sum(np.array(my_predictions)==np.argmax(test_targets, axis=1)) / len(my_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
