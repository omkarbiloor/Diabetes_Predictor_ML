# 🩺 Diabetes Prediction Web Application

A simple machine learning web application that predicts whether a female patient is likely diabetic based on medical inputs. Built using **Flask**, trained in **Jupyter Notebook**, and deployed using **Render**.

🌐 **Live Demo**: [Diabetes Predictor App](https://diabetes-predictor-ml-q5hy.onrender.com/)

---

## 🚀 Features

- Beautiful animated UI with HTML & CSS
- Predicts diabetes using medical input values
- Real-time result displayed on the same page
- Trained with a logistic regression/classification model
- Deployed using Render with minimal backend load

---

## 🛠️ Tech Stack

- Python
- Flask
- HTML5 + CSS3
- Jupyter Notebook
- scikit-learn (for ML model)
- joblib (for model serialization)

---

## 📁 Project Structure

## .
## ├── diabetes.ipynb # Jupyter notebook for training and saving the model
## ├── application.py # Flask backend
## ├── models/
## │ ├── diabetes_model.pkl # Trained ML model
## │ └── scaler.pkl # Scaler used for normalization
## ├── templates/
## │ ├── index.html # Welcome page
## │ └── home.html # Prediction form and results displayed here


## 📌 How It Works

1. User provides medical details:
   - Pregnancies
   - Glucose
   - Blood Pressure
   - Skin Thickness
   - Insulin
   - BMI
   - Diabetes Pedigree Function
   - Age

2. Input data is scaled using the `scaler.pkl`.

3. The scaled input is passed to the trained model (`diabetes_model.pkl`) to predict the result.

4. The result is shown on the same page with animated styling.

---

## 🎯 Output Example

- ✅ **Negative**: The person is likely not diabetic (green result box).
- ❌ **Positive**: The person is likely diabetic (red result box).

---

## 📎 Notes

- This model is for educational/demo purposes only and **not for medical use**.
- It uses a basic machine learning pipeline and pre-cleaned dataset (likely PIMA Indian Diabetes Dataset).

---
