import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Student Marks Prediction App")

# Sample data
data = pd.DataFrame({
    'Hours': [1, 2, 3, 4.5, 5.5, 6.1, 7.4, 8, 9, 10],
    'Scores': [10, 20, 30, 45, 55, 61, 74, 80, 90, 95]
})

# Train model
X = data[['Hours']]
y = data['Scores']
model = LinearRegression()
model.fit(X, y)

# Input
hours = st.slider("Enter study hours", 0.0, 12.0, 1.0)

# Prediction
if st.button("Predict"):
    pred = model.predict([[hours]])
    st.success(f"Predicted Score: {pred[0]:.2f}")