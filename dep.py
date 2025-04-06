import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model1 = joblib.load(os.path.join(BASE_DIR, 'xgb_model.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))

def load_data(file):
    return pd.read_csv(file)

def clean_data(df, for_prediction=False):

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(['customerID', 'gender', 'MultipleLines', 'PhoneService', 'StreamingTV', 'StreamingMovies'], axis=1, errors='ignore')
    label_encoder = LabelEncoder()
    if 'Churn' in df.columns:
        df['Churn_Encoded'] = label_encoder.fit_transform(df['Churn'])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    

    if for_prediction and encoders is not None:  # Prediction mode
        for col in categorical_cols:
            if col in encoders:
                encoded_data = encoders[col].transform(df[[col]])
                encoded_cols = [f'{col}_{cat}' for cat in encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(col, axis=1)
    
    valid_indices = ~np.isnan(df['TotalCharges'])
    f1 = df.iloc[valid_indices.to_numpy(), :].drop(['Churn', 'Churn_Encoded'], axis=1, errors='ignore')
    if not for_prediction:
        t1 = df.iloc[valid_indices.to_numpy(), df.columns.get_loc('Churn_Encoded')].values
        over = SMOTE(sampling_strategy=1)
        f1, t1 = over.fit_resample(f1.values, t1)
        return f1, t1
    return f1.values

def predict_churn(df):
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))
    encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
    if feature_cols is None or encoders is None:
        raise ValueError("Feature columns or encoders not loaded. Run training first.")
    cleaned_values = clean_data(df.copy(), for_prediction=True)
    cleaned_df = pd.DataFrame(cleaned_values, columns=feature_cols)
    xgb_proba = model1.predict_proba(cleaned_df)[:, 1]
    pred = (xgb_proba > 0.4).astype(int)
    return pred

def generate_insights(df):
    df['Churn_Encoded'] = df['Churn'].map({'No': 0, 'Yes': 1})
    churn_rate = df['Churn_Encoded'].mean() * 100
    insights = f"Churn Rate: {churn_rate:.2f}%"
    plots = []

    churn_counts = df['Churn'].value_counts().reset_index()
    fig1 = px.bar(churn_counts, x='Churn', y='count', title='Churn vs Non-Churn Customers')
    plots.append(fig1.to_json())

    contract_counts = df['Contract'].value_counts().reset_index()
    fig2 = px.bar(contract_counts, x='Contract', y='count', title='Distribution of Contract Types')
    plots.append(fig2.to_json())

    lines_counts = df['MultipleLines'].value_counts().reset_index()
    fig3 = px.bar(lines_counts, x='MultipleLines', y='count', title='Distribution of Line Types')
    plots.append(fig3.to_json())

    payment_counts = df['PaymentMethod'].value_counts().reset_index()
    fig4 = px.bar(payment_counts, x='PaymentMethod', y='count', title='Distribution of Payment Types')
    plots.append(fig4.to_json())

    internet_counts = df['InternetService'].value_counts().reset_index()
    fig5 = px.bar(internet_counts, x='InternetService', y='count', title='Distribution of Internet Service Types')
    plots.append(fig5.to_json())

    fig6 = px.histogram(df, x='tenure', nbins=80, title='Tenure Distribution (Months)', histnorm='probability density')
    plots.append(fig6.to_json())

    fig7 = px.histogram(df, x='MonthlyCharges', nbins=80, title='Monthly Charges Distribution', histnorm='probability density')
    plots.append(fig7.to_json())

    fig8 = px.histogram(df, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
    plots.append(fig8.to_json())

    fig9 = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges for Churned vs Retained Customers')
    plots.append(fig9.to_json())

    fig10 = px.histogram(df, x='PaymentMethod', color='Churn', barmode='group', title='Churn by Payment Type')
    plots.append(fig10.to_json())

    fig11 = px.box(df, x='Churn_Encoded', y='tenure', title='Tenure of Churned vs Retained Customers')
    plots.append(fig11.to_json())

    fig12 = px.histogram(df, x='InternetService', color='Churn', barmode='group', title='Churn by Internet Service Type')
    plots.append(fig12.to_json())

    fig13 = px.histogram(df, x='MultipleLines', color='Churn', barmode='group', title='Churn by Line Type')
    plots.append(fig13.to_json())

    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72], 
                               labels=['0-6', '6-12', '12-18', '18-24', '24-30', '30-36', '36-42', '42-48', '48-54', '54-60', '60-66', '66-72'])
    fig14 = px.histogram(df, x='TenureGroup', color='Churn', barmode='group', title='Churn by Tenure Group')
    plots.append(fig14.to_json())

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    corr = df[numeric_cols].corr()
    fig15 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', 
                                      text=corr.round(2).values, texttemplate="%{text}"))
    fig15.update_layout(title='Correlation Matrix')
    plots.append(fig15.to_json())

    fig16 = px.scatter(df, x='tenure', y='TotalCharges', color='Churn', title='Tenure vs Total Charges (Colored by Churn)')
    plots.append(fig16.to_json())

    fig17 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Tenure vs Monthly Charges (Colored by Churn)')
    plots.append(fig17.to_json())

    segment = df.groupby(['Contract', 'InternetService'])['Churn_Encoded'].mean().reset_index()
    fig18 = px.bar(segment, x='Contract', y='Churn_Encoded', color='InternetService', barmode='group', 
                   title='Churn Rate by Contract and Internet Service')
    plots.append(fig18.to_json())

    return insights, plots