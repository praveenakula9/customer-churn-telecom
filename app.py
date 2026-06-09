import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_churn_model.pkl')

st.title('Customer Churn Predictor')
st.write('Enter customer details to predict churn probability.')

# Input fields
tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges ($)', 18.0, 120.0, 65.0)
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
paperless = st.selectbox('Paperless Billing', ['Yes', 'No'])

if st.button('Predict Churn'):
    st.warning('Connect input fields to model features matching your encoding.')
    st.info('See notebook for full feature list — map inputs to get_dummies columns.')