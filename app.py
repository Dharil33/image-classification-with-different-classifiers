import numpy as np
import streamlit as st
import os
import pickle
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

dictionary = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'T-Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot'
}

st.title("Fashion MNIST Classification")

def prediction(file):
    rr2 = [[0] * 28 for i in range(28)]
    for i in range(0, 28):
        for j in range(0, 28):
            coordinate =(i, j)
            rr2[j][i] = file.getpixel(coordinate)
    arr1 = list(np.concatenate(rr2).flat)
    column_name=['pixel'+str(i) for i in range(1,785)]
    X_test1 = pd.DataFrame(np.array(arr1).reshape(-1,len(arr1)),columns = column_name)
    prediction = loaded_model.predict(X_test1)
    for key, value in dictionary.items():
        if key == prediction:
            return value

uploaded_file = st.file_uploader("Choose an Image file", type=['jpg','png','jpeg'], accept_multiple_files=False)

if uploaded_file:
    im = Image.open(uploaded_file)
    st.image(im)

if st.button("Predict"):
    result = prediction(im)
    st.success(result)
   

