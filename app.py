import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import catboost
# Load the model
model = joblib.load('catboost_model.pkl')




# Define feature names
feature_names = [
'Gestational_age','Maternal_age','Gravidity','Parturition','Prior_C-sections_number',
'Uterine_surgery_number','Cervical_canal_length','Uterine_anteroposterior_diameter_ratio', 
'Placental_abnormal_vasculature_diameter','Placental_abnormal_vasculature_area',
'Intraplacental_dark_T2_band_area'
]
# features = np.array([[40,274,7,1,1,0,4.3,1.21,2.3,2.783,4.0746]])
# print(model.predict_proba(input)[0][1])

# Streamlit user interface
st.title("PAS Risk Model")

# age: numerical input
Gestational_age = st.number_input("Gestational Age (year):", min_value=0, max_value=100, value=25)

# sex: categorical selection
Maternal_age = st.number_input("Maternal Age (day):",  min_value=0, max_value=500, value=205)

# cp: categorical selection
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=50, value=3)

# trestbps: numerical input
Parturition = st.number_input("Parturition:", min_value=0, max_value=50, value=1)

# trestbps: numerical input
Prior_C_sections_number = st.number_input("Prior C-Sections Number:", min_value=0, max_value=50, value=0)

# chol: numerical input
Uterine_surgery_number = st.number_input("Uterine Surgery Number:", min_value=0, max_value=50, value=1)

# fbs: categorical selection
Cervical_canal_length = st.number_input("Cervical Canal Length (mm):", min_value=0, max_value=200, value=22)

# restecg: categorical selection
Uterine_anteroposterior_diameter_ratio = st.number_input("Uterine Anteroposterior Diameter Ratio:", min_value=0.0, max_value=50.0, value=1.5)


# thalach: numerical input
Placental_abnormal_vasculature_diameter = st.number_input("Placental Abnormal Vasculature Diameter (mm):", min_value=0., max_value=50.0, value=3.4)

# thalach: numerical input
Placental_abnormal_vasculature_area = st.number_input("Placental Abnormal Vasculature Area (mm2):",  min_value=0.0, max_value=50.0, value=9.6)

# exang: categorical selection
Intraplacental_dark_T2_band_area = st.number_input("Intraplacental Dark T2WI Band Area (mm2):", min_value=0.0, max_value=50.0, value=7.5)

# Process inputs and make predictions
feature_values = [Gestational_age,Maternal_age,Gravidity,Parturition,Prior_C_sections_number,Uterine_surgery_number,Cervical_canal_length,Uterine_anteroposterior_diameter_ratio,Placental_abnormal_vasculature_diameter,Placental_abnormal_vasculature_area,Intraplacental_dark_T2_band_area]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_proba = model.predict_proba(features)[0][1]

    # Display prediction results
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    
    advice = f"The PAS Risk Model predicts a {predicted_proba * 100}% probability of intraoperative bleeding and hysterectomy during surgery."
    st.write(advice)




    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(features, columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(features, columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
