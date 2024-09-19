# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')


st.title('Heart Disease Prediction')

st.sidebar.header('Options')
upload_data = st.sidebar.file_uploader("Upload your own dataset", type=["csv"])
show_feature_importance = st.sidebar.checkbox("Show Feature Importance")


st.markdown("""
    This app predicts the likelihood of heart disease based on user inputs. 
    You can use the provided form to enter your medical data or upload your own dataset. 
    The model will then predict whether heart disease is likely.
""")


def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
    chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    restecg = st.selectbox('Resting ECG results', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia', [0, 1, 2, 3])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


if upload_data is not None:
    user_data = pd.read_csv(upload_data)
    st.write("Uploaded Dataset")
    st.write(user_data)
    input_df = user_data
else:
    input_df = user_input_features()


input_scaled = scaler.transform(input_df)


threshold = 0.4  


if st.button('Predict'):
    prediction_proba = model.predict_proba(input_scaled)
    prediction = (prediction_proba[:, 1] >= threshold).astype(int)
    
    st.subheader('Prediction')
    if upload_data is not None:
        result = pd.DataFrame({
            'Prediction': ['Yes' if pred == 1 else 'No' for pred in prediction],
            'Probability of Heart Disease': [f"{proba[1] * 100:.2f}%" for proba in prediction_proba]
        })
        st.write(result)
    else:
        heart_disease_status = 'Yes' if prediction[0] == 1 else 'No'
        st.write(f"Does the person have heart disease? **{heart_disease_status}**")
        st.write(f"Probability of heart disease: **{prediction_proba[0][1] * 100:.2f}%**")

    
    st.subheader('Raw Prediction Probabilities')
    st.write(f"Probability of No Heart Disease (0): {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease (1): {prediction_proba[0][1]:.2f}")


if show_feature_importance:
    st.subheader('Feature Importance')
    st.write("The plot below shows the importance of each feature in the model's predictions.")
    st.image('models/feature_importance.png')
