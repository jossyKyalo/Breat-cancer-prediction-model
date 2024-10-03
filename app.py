import streamlit as st
import joblib
import numpy as np

# Loading  the trained model
model = joblib.load('breast_cancer_model.pkl')

# Setting up Streamlit title and input form
st.title('Breast Cancer Prediction App')

st.write('Enter tumor characteristics to predict if the tumor is benign or malignant:')

# Collect user inputs
mean_radius = st.number_input('Mean Radius')
mean_texture = st.number_input('Mean Texture')
mean_perimeter = st.number_input('Mean Perimeter')
mean_area = st.number_input('Mean Area')
mean_smoothness = st.number_input('Mean Smoothness')

# Once the user clicks a button, predict
if st.button('Predict'):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    prediction = model.predict(input_data)
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    st.write(f'The model predicts: **{result}**')




