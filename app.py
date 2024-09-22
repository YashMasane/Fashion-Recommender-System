import streamlit as st
import pickle
import tensorflow 
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import os

import numpy as np
from numpy.linalg import norm


st.title('Fashion Recommender System')

embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb'))).reshape((44441, 2048))
embeddings = embeddings.reshape((44441, 2048))
file_names = pickle.load(open('filenames.pkl', 'rb'))
model = tensorflow.keras.models.load_model('model.h5')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocesed_input = preprocess_input(expanded_img_array)
    result = model.predict(preprocesed_input)
    normalized_result = result/norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors(features)

    return indices

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        display_image = display_image.resize((300, 300))
        st.image(display_image)
        # feature extract
        features = extract_features(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        print(features)
        indices = recommend(features,embeddings)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(file_names[indices[0][0]])
        with col2:
            st.image(file_names[indices[0][1]])
        with col3:
            st.image(file_names[indices[0][2]])
        with col4:
            st.image(file_names[indices[0][3]])
        with col5:
            st.image(file_names[indices[0][4]])
    else:
        st.header("Some error occured in file upload")


