from flask import Flask, render_template, request
import pandas as pd
import os
from dep import load_data, clean_data, predict_churn, generate_insights

app = Flask(__name__)

# Load and process data once, using a portable path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = load_data(DATA_PATH)
insights, plots = generate_insights(df)

@app.route('/')
def index():
    return render_template('index.html', insights=insights, plots=plots)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect all original columns from the user
        user_data = {
            'gender': request.form['gender'],  # "Male" or "Female"
            'SeniorCitizen': int(request.form['SeniorCitizen']),  # 0 or 1
            'Partner': request.form['Partner'],  # "Yes" or "No"
            'Dependents': request.form['Dependents'],  # "Yes" or "No"
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],  # "Yes" or "No"
            'MultipleLines': request.form['MultipleLines'],  # "Yes", "No", "No phone service"
            'InternetService': request.form['InternetService'],  # "DSL", "Fiber optic", "No"
            'OnlineSecurity': request.form['OnlineSecurity'],  # "Yes", "No", "No internet service"
            'OnlineBackup': request.form['OnlineBackup'],  # "Yes", "No", "No internet service"
            'DeviceProtection': request.form['DeviceProtection'],  # "Yes", "No", "No internet service"
            'TechSupport': request.form['TechSupport'],  # "Yes", "No", "No internet service"
            'StreamingTV': request.form['StreamingTV'],  # "Yes", "No", "No internet service"
            'StreamingMovies': request.form['StreamingMovies'],  # "Yes", "No", "No internet service"
            'Contract': request.form['Contract'],  # "Month-to-month", "One year", "Two year"
            'PaperlessBilling': request.form['PaperlessBilling'],  # "Yes" or "No"
            'PaymentMethod': request.form['PaymentMethod'],  # "Electronic check", etc.
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }
        user_df = pd.DataFrame([user_data])
        pred = predict_churn(user_df)
        print("Prediction:", pred)
        result = "Churn" if pred[0] == 1 else "No Churn"
        return render_template('predict.html', prediction=result)
    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)