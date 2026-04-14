import pandas as pd
import numpy as np
import shap
import pickle

def get_risk(prob):
    """Categorizes churn probability into risk buckets."""
    if prob < 0.3:
        return "Low Risk 🟢"
    elif prob < 0.7:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

def build_input_vector(senior, tenure, monthly, total, gender, partner, dependents,
                       phone_service, multiple_lines, internet, online_security,
                       online_backup, device_protect, tech_support, streaming_tv,
                       streaming_movies, contract, paperless, payment):
    """
    Takes the raw Streamlit sidebar inputs, maps them to a single-row DataFrame, 
    and applies dummy encoding to match the XGBoost training schema exactly.
    """
    raw_dict = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless else "No",
        "PaymentMethod": payment
    }
    
    raw_df = pd.DataFrame([raw_dict])
    encoded_df = pd.get_dummies(raw_df)
    
    # Load the exact columns the model was trained on
    with open("columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    
    # Map the encoded inputs to the exact model schema, filling missing with 0
    input_vec = {}
    for col in model_columns:
        if col in encoded_df.columns:
            # We use bool/int conversion depending on pandas version, so float is safe
            input_vec[col] = float(encoded_df.iloc[0][col])
        else:
            input_vec[col] = 0.0
            
    return input_vec

def get_feature_contributions(model, model_columns, input_df, df_full):
    """
    Uses SHAP (SHapley Additive exPlanations) to calculate exactly how much 
    each feature contributed to the final probability score for the Glass Box UI.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # XGBoost shap output shapes can vary slightly by version; handle both 1D and 2D
    if isinstance(shap_values, list):
        contributions = shap_values[1][0] 
    elif len(shap_values.shape) > 1:
        contributions = shap_values[0]
    else:
        contributions = shap_values
        
    return pd.Series(contributions, index=model_columns)

def preprocess_uploaded_df(raw_df, model_columns):
    """
    Takes a raw bulk CSV upload, cleans it, applies dummy encoding, 
    and forces the shape to perfectly match the XGBoost model expected input.
    """
    if 'customerID' in raw_df.columns:
        customer_ids = raw_df['customerID']
        df = raw_df.drop('customerID', axis=1)
    else:
        # Generate dummy IDs if the user didn't upload them
        customer_ids = pd.Series([f"CUST-{i:04d}" for i in range(len(raw_df))])
        df = raw_df.copy()
        
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
        
    # Standardize TotalCharges (often comes in as string with blank spaces in this dataset)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
    df_encoded = pd.get_dummies(df)
    
    # Align the new encoded columns perfectly with the trained model columns
    final_df = pd.DataFrame()
    for col in model_columns:
        if col in df_encoded.columns:
            final_df[col] = df_encoded[col]
        else:
            final_df[col] = 0
            
    return final_df, customer_ids

def generate_sample_template():
    """Generates the downloadable CSV template for Tab 2."""
    data = {
        "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
        "gender": ["Female", "Male", "Male"],
        "SeniorCitizen": [0, 0, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "No"],
        "tenure": [1, 34, 2],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No"],
        "InternetService": ["DSL", "DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [29.85, 1889.5, 108.15]
    }
    return pd.DataFrame(data)