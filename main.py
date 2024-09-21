# importing libraries
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# resnet model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# model architecture
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

# function for feature extraction from images
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocesed_input = preprocess_input(expanded_img_array)
    result = model.predict(preprocesed_input)
    normalized_result = result/norm(result)

    return normalized_result

#list for file names
file_names = []

# storing file names in list
for file in os.listdir('Dataset/images'):
    file_names.append(os.path.join('Dataset/images', file))

# feature list
feature_list = []

# creating features for images
for file in tqdm(file_names):
    feature_list.append(extract_features(file, model))

# saving feature list and file names
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(file_names, open('filenames.pkl', 'wb'))

# saving model
model.save('model.h5')