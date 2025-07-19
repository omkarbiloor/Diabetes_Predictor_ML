from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model_path = os.path.join('model', 'model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')
diabetes_predictor = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get form data
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            Bloodpressure = float(request.form['Bloodpressure'])
            Skinthickness = float(request.form['Skinthickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            # Create DataFrame
            new_data = pd.DataFrame([[Pregnancies, Glucose, Bloodpressure, Skinthickness,
                                      Insulin, BMI, DiabetesPedigreeFunction, Age]],
                                    columns=['Pregnancies', 'Glucose', 'Bloodpressure', 'Skinthickness',
                                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

            # Scale data
            new_data_scaled = scaler.transform(new_data)

            # Predict
            prediction = diabetes_predictor.predict(new_data_scaled)

            # Result based on prediction
            if prediction[0] == 1:
                result = "Positive: The person is likely diabetic."
            else:
                result = "Negative: The person is likely not diabetic."

            # Show result in result.html
            return render_template('result.html', results=result)
        
        except Exception as e:
            return f"Something went wrong: {e}"
