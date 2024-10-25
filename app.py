from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load(r"C:\Users\Naveena Reddy\Downloads\Student-Marks-Prediction\model\student_mark_predictor.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    input_data = pd.DataFrame({'study_hours': [study_hours]})
    prediction = model.predict(input_data)[0][0]
    return render_template('index.html', predicted_marks=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
