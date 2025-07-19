from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

application = Flask(__name__)
app = application


diabetes_predictor = joblib.load('models/diabetes_model.pkl')
standard_scaler = joblib.load('models/scaler.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        Bloodpressure = float(request.form['Bloodpressure'])
        Skinthickness = float(request.form['Skinthickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        new_data_scaled = standard_scaler.transform([[
            Pregnancies, Glucose, Bloodpressure, Skinthickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])


        prediction = diabetes_predictor.predict(new_data_scaled)[0]
        result = "Positive" if prediction == 1 else "Negative"

        return render_template("result.html", result=result)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
