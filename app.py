# app.py
# This is the main Flask application. It loads the trained model and
# scaler, shows a form to the user, and displays a prediction result.

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler when the app starts
with open("model/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    # Just shows the empty form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the values the user typed into the form
    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    pedigree = float(request.form["pedigree"])
    age = float(request.form["age"])

    # Put them in the same order and with the same column names the
    # model was trained on
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    # Scale the input the same way the training data was scaled
    input_scaled = scaler.transform(input_data)

    # Get the prediction (0 = non-diabetic, 1 = diabetic)
    prediction = model.predict(input_scaled)[0]

    # Get the probability (confidence) of being diabetic
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        result = "Diabetic Risk"
    else:
        result = "Low Risk"

    return render_template("result.html", result=result, probability=round(probability, 1))


if __name__ == "__main__":
    app.run(debug=True)