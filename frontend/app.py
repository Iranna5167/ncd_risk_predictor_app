import streamlit as st
import joblib
import pandas as pd
import numpy as np # Used for np.number for feature identification

# --- Configuration and Page Setup ---
# Sets up the browser tab title, favicon, page layout
st.set_page_config(
    page_title="NCD Risk Predictor for India",
    page_icon="❤️",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="collapsed" # 'auto', 'expanded', or 'collapsed'
)

# --- Load Saved ML Models and Feature Lists ---
# Paths are relative to frontend/app.py. They point to your 'models/' folder.
try:
    diabetes_model = joblib.load('models/diabetes_risk_model_pipeline.pkl')
    hypertension_model = joblib.load('models/hypertension_risk_model_pipeline.pkl')

    # Load feature lists (these are CRITICAL for ensuring inputs match model's training order/names)
    diabetes_numerical_features = joblib.load('models/numerical_features.pkl')
    diabetes_categorical_features = joblib.load('models/categorical_features.pkl')

    hypertension_numerical_features = joblib.load('models/numerical_features_h.pkl')
    hypertension_categorical_features = joblib.load('models/categorical_features_h.pkl')

    # Reconstruct the full ordered list of features for each model from the loaded lists
    # This order is CRUCIAL for the model's ColumnTransformer to work correctly during prediction
    all_diabetes_features = diabetes_numerical_features + diabetes_categorical_features
    all_hypertension_features = hypertension_numerical_features + hypertension_categorical_features

    # Display a success message in a sidebar (optional, useful for debugging)
    st.sidebar.success("ML Models and features loaded successfully!")
except Exception as e:
    # If models fail to load, stop the app and show an error
    st.error(f"Error loading ML models or features. Make sure they are in the 'models/' folder relative to this app. Error: {e}")
    st.stop() # Stop the app execution

# --- Helper Function to Determine Risk Level ---
def get_risk_level(probability):
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.3:
        return "Moderate Risk"
    else:
        return "Low Risk"

# --- Helper Function to Generate Personalized Advisories ---
# This logic combines static messages with dynamic probability and factors
def generate_advisory_text(risk_level, condition_name, probability, top_factors_list):
    advisory_messages = {
        "Low Risk": {
            "title": f"🎉 Low Risk for {condition_name}!",
            "icon": "✅",
            "color": "green",
            "message": f"Your current profile indicates a low risk ({probability*100:.1f}%) for {condition_name}. Keep up the good work! Maintain a balanced diet, regular physical activity (e.g., yoga, brisk walking), and regular health check-ups. Focus on traditional Indian diet practices like consuming more millets, lentils, and fresh seasonal vegetables."
        },
        "Moderate Risk": {
            "title": f"⚠️ Moderate Risk for {condition_name}.",
            "icon": "🟠",
            "color": "orange",
            "message": f"Your current profile suggests a moderate risk ({probability*100:.1f}%) for {condition_name}. This is a good time for proactive steps. We recommend consulting a healthcare professional for a check-up. Focus on reducing processed foods, sugary drinks, and high-salt/high-fat snacks. Increase intake of whole grains and fresh fruits. Regular physical activity like walking for 30-45 minutes daily can make a big difference."
        },
        "High Risk": {
            "title": f"🚨 High Risk for {condition_name}!",
            "icon": "🔴",
            "color": "red",
            "message": f"Your current profile indicates a high risk ({probability*100:.1f}%) for {condition_name}. We strongly advise consulting a doctor or specialist for a detailed evaluation as soon as possible. Focus on strict dietary control (e.g., portion control, low GI foods for diabetes, low sodium for hypertension), and incorporate daily structured exercise. Stress management techniques like meditation or pranayama can also be beneficial."
        }
    }

    # Add specifics from top contributing factors (based on your ML analysis)
    if top_factors_list:
        factors_str = ", ".join(top_factors_list)
        advisory_messages[risk_level]["message"] += f"\n\n**Key factors contributing to your risk:** {factors_str}."

        # Example of more specific advice based on factors (you can expand this!)
        if "age" in factors_str.lower() and risk_level != "Low Risk":
             advisory_messages[risk_level]["message"] += " While age is a factor, lifestyle changes remain impactful."
        if "bmi" in factors_str.lower() and risk_level != "Low Risk":
             advisory_messages[risk_level]["message"] += " Focusing on weight management can significantly reduce your risk."
        if "HbA1c_level" in factors_str or "blood_glucose_level" in factors_str:
            advisory_messages[risk_level]["message"] += " Monitoring blood sugar is vital."
        if "smoking_history" in factors_str or "heart_disease" in factors_str:
            advisory_messages[risk_level]["message"] += " Quitting smoking and managing heart health are paramount."

    return advisory_messages[risk_level]

# --- Helper Function to Extract Top Contributing Factors (simplified for app) ---
# This directly uses the saved feature importance data.
def get_top_factors_for_display(model_pipeline, numerical_feats, categorical_feats, user_input_values, n=3):
    try:
        # Reconstruct feature names after OneHotEncoding for interpretability
        ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        # Note: get_feature_names_out creates new names like 'gender_Female'
        ohe_feature_names = list(ohe.get_feature_names_out(categorical_feats))

        # Combine original numerical names with new one-hot encoded names
        all_model_features_for_importance = numerical_feats + ohe_feature_names

        classifier = model_pipeline.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importances = pd.Series(classifier.feature_importances_, index=all_model_features_for_importance)
            # Get the top N most important features overall based on the model
            top_features_overall = importances.nlargest(n).index.tolist()

            # Further refine for display: map back to original categorical names
            display_factors = []
            for factor_name in top_features_overall:
                # Check if it's an original numerical feature
                if factor_name in numerical_feats:
                    display_factors.append(f"{factor_name.replace('_', ' ').title()}")
                # Check if it's an one-hot encoded feature, then get its original category name
                else:
                    original_cat_feature = next((col for col in categorical_feats if col in factor_name), None)
                    if original_cat_feature:
                        display_factors.append(f"{original_cat_feature.replace('_', ' ').title()} ({user_input_values[original_cat_feature]})")
                    else:
                        display_factors.append(factor_name.replace('_', ' ').title()) # Fallback
            return display_factors
        return []
    except Exception as e:
        # st.warning(f"Could not get contributing factors: {e}") # For debugging purposes in app
        return [] # Return empty list if error occurs

# --- Application Title and Description ---
st.title("🇮🇳 NCD Risk Predictor & Health Advisory for India")
st.markdown("""
Welcome! This interactive tool uses Artificial Intelligence to estimate your personalized risk for
**Diabetes** and **Hypertension** based on your health profile.
It then provides actionable, culturally relevant health advisories.
""")

st.warning("🚨 **Disclaimer:** This tool provides a risk assessment for informational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.")

st.markdown("---")

# --- User Input Form ---
st.header("👤 Enter Your Health Profile")

# Use columns to organize input fields for a cleaner layout
col1, col2, col3 = st.columns(3) # Can use 2 or 3 columns

with col1:
    st.subheader("Demographics")
    gender = st.selectbox(
        "Gender",
        ('Female', 'Male', 'Other'),
        index=0 # Default to Female
    )
    age = st.number_input(
        "Age (Years)",
        min_value=0, max_value=120, value=30, step=1
    )
    bmi = st.number_input(
        "BMI (Body Mass Index kg/m²)",
        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
        help="Calculate your BMI: weight (kg) / [height (m)]²"
    )

with col2:
    st.subheader("Medical History")
    hypertension_input = st.radio( # This is for the diabetes model's input feature
        "Do you have a history of Hypertension?",
        ('No', 'Yes'),
        index=0
    )
    heart_disease_input = st.radio( # Note: Renamed from 'heart_disease' to avoid confusion if used as a feature
        "Do you have a history of Heart Disease?",
        ('No', 'Yes'),
        index=0
    )
    smoking_history = st.selectbox(
        "Smoking History",
        ('never', 'No Info', 'current', 'former', 'ever', 'sometimes'),
        index=0 # Default to 'never'
    )

with col3:
    st.subheader("Blood Biomarkers")
    HbA1c_level = st.number_input(
        "HbA1c Level (%)",
        min_value=3.0, max_value=15.0, value=5.5, step=0.1,
        help="Glycated Hemoglobin. Normal: <5.7%, Pre-diabetes: 5.7-6.4%, Diabetes: >=6.5%"
    )
    blood_glucose_level = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=50, max_value=300, value=100, step=1,
        help="Normal: <100 mg/dL, Pre-diabetes: 100-125 mg/dL, Diabetes: >=126 mg/dL (Fasting)"
    )

st.markdown("---")

# --- Prediction Button ---
# Button to trigger the prediction logic
if st.button("Calculate My Risk", help="Click to get your personalized risk assessment"):

    # --- Prepare User Input for Models ---
    # Create a dictionary from user inputs, matching the exact keys from your ML training
    # Convert 'Yes'/'No' radio button inputs to 1/0 integers as the models expect
    user_inputs_raw = {
        'gender': gender,
        'age': float(age), # Ensure float type for numerical features
        'hypertension': 1 if hypertension_input == 'Yes' else 0,
        'heart_disease': 1 if heart_disease_input == 'Yes' else 0,
        'smoking_history': smoking_history,
        'bmi': float(bmi),
        'HbA1c_level': float(HbA1c_level),
        'blood_glucose_level': float(blood_glucose_level)
    }

    # --- Make Predictions ---
    # 1. Prepare input for Diabetes model: Ensure column order matches 'all_diabetes_features'
    # The Streamlit input names should ideally match the feature names used in training.
    input_df_diabetes = pd.DataFrame([user_inputs_raw], columns=all_diabetes_features)

    # 2. Prepare input for Hypertension model: Ensure column order matches 'all_hypertension_features'
    # Note: 'diabetes' column is NOT in hypertension model's features, so ensure it's not passed.
    # This is handled by defining all_hypertension_features from the loaded .pkl
    input_df_hypertension = pd.DataFrame([user_inputs_raw], columns=all_hypertension_features)

    # Diabetes Prediction
    diabetes_pred_proba = diabetes_model.predict_proba(input_df_diabetes)[0, 1] # Probability of Class 1 (Diabetes)
    diabetes_prediction_class = diabetes_model.predict(input_df_diabetes)[0] # Predicted class (0 or 1)

    # Hypertension Prediction
    hypertension_pred_proba = hypertension_model.predict_proba(input_df_hypertension)[0, 1] # Probability of Class 1 (Hypertension)
    hypertension_prediction_class = hypertension_model.predict(input_df_hypertension)[0] # Predicted class (0 or 1)

    # --- Generate Risk Levels and Advisories ---
    diabetes_risk_level = get_risk_level(diabetes_pred_proba)
    hypertension_risk_level = get_risk_level(hypertension_pred_proba)

    # Get top contributing factors for display
    diabetes_top_factors = get_top_factors_for_display(diabetes_model, diabetes_numerical_features, diabetes_categorical_features, user_inputs_raw)
    hypertension_top_factors = get_top_factors_for_display(hypertension_model, hypertension_numerical_features, hypertension_categorical_features, user_inputs_raw)


    # --- Display Results ---
    st.markdown("---")
    st.header("📈 Your Personalized Risk Assessment")

    # --- Diabetes Results Section ---
    st.subheader("Diabetes Risk")
    diabetes_advisory_info = generate_advisory_text(diabetes_risk_level, 'Diabetes', diabetes_pred_proba, diabetes_top_factors)

    # Use Streamlit containers for visual grouping
    with st.container(border=True): # New in Streamlit 1.30, adds a nice border
        st.markdown(f"**Overall Risk Level:** <span style='font-weight:bold; color:{diabetes_advisory_info['color']};'>{diabetes_risk_level}</span>", unsafe_allow_html=True)
        st.markdown(f"**Probability:** <span style='font-weight:bold;'>{(diabetes_pred_proba * 100):.1f}%</span>", unsafe_allow_html=True)
        st.info(diabetes_advisory_info['message'])

    st.markdown("---")

    # --- Hypertension Results Section ---
    st.subheader("Hypertension Risk")
    hypertension_advisory_info = generate_advisory_text(hypertension_risk_level, 'Hypertension', hypertension_pred_proba, hypertension_top_factors)

    with st.container(border=True): # New in Streamlit 1.30, adds a nice border
        st.markdown(f"**Overall Risk Level:** <span style='font-weight:bold; color:{hypertension_advisory_info['color']};'>{hypertension_risk_level}</span>", unsafe_allow_html=True)
        st.markdown(f"**Probability:** <span style='font-weight:bold;'>{(hypertension_pred_proba * 100):.1f}%</span>", unsafe_allow_html=True)
        st.info(hypertension_advisory_info['message'])

st.markdown("---")
st.info("Remember: This is a risk assessment tool, not a diagnostic one. Always consult a medical professional.")