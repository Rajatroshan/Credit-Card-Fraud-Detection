import streamlit as st
import requests


st.title("Credit Card Fraud Prediction")



features = []
for i in range(30):  
    value = st.number_input(f"Feature {i + 1}", value=0.0)
    features.append(value)

if st.button("Predict"):
 
    input_data = {'features': features}

   
    response = requests.post('http://127.0.0.1:5000/predict', json=input_data)

    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f'Prediction: {"Fraud" if prediction == 1 else "Legitimate"}')
    else:
        st.error('Error in prediction. Please check the server.')
