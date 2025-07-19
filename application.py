from flask import Flask, jsonify, request, send_from_directory, render_template
import pandas as pd
import numpy as np
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

##Import diabetes model and scaler

diabetes_predictor = load('models/diabetes_model.pkl')
standard_scaler=load('models/scaler.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST']) 
def predict_datapoint():
    if request.method=="POST":
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        Bloodpressure = float(request.form['Bloodpressure'])
        Skinthickness = float(request.form['Skinthickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])  
        Age = float(request.form['Age'])
        
        new_data_scaled=standard_scaler.transform([[Pregnancies, Glucose, Bloodpressure, Skinthickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction=diabetes_predictor.predict(new_data_scaled)
        if prediction[0] == 1:
            result = "Positive: The person is likely diabetic."
        else:
            result = "Negative: The person is likely not diabetic."
            
        return render_template('home.html',results=result)
        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
