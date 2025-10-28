from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load("covid_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    Fever = request.form['Fever']
    Dry_Cough = request.form['Dry_Cough']
    Fatigue = request.form['Fatigue']
    Breathing_Problem = request.form['Breathing_Problem']

    encoder = LabelEncoder()
    features = np.array([[Fever, Dry_Cough, Fatigue, Breathing_Problem]])
    features_encoded = np.where(features == 'Yes', 1, 0)

    prediction = model.predict(features_encoded)[0]
    result = "Positive" if prediction == 1 else "Negative"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
