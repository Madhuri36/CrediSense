import streamlit as st
import pandas as pd
import joblib

model = joblib.load("extra_trees_credit_model.pkl")
encoder={col:joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing","Saving accounts","Checking account"]}

st.title("Credit Risk Prediction Application")
st.write("Enter the following Applicant information to predict if the credit risk is good or bad.")

age = st.number_input("Age", min_value=18, max_value=80, value=30) 
sex = st.selectbox("Sex", options=["male", "female"])
job= st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing= st.selectbox("Housing", options=["own", "rent", "free"])
saving_accounts= st.selectbox("Saving accounts", options=["little", "moderate", "quite rich", "rich"])
checking_account= st.selectbox("Checking account", options=["little", "moderate", "rich"])
credit_amount= st.number_input("Credit Amount", min_value=0, value=1000)
duration=st.number_input("Duration", min_value=1, value=12)

input_df=pd.DataFrame({
    "Age":[age],
    "Sex": [encoder["Sex"].transform([sex])[0]],
    "Job":[job],
    "Housing":[encoder["Housing"].transform([housing])[0]],
    "Saving accounts":[encoder["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account":[encoder["Checking account"].transform([checking_account])[0]],
    "Credit amount":[credit_amount],
    "Duration":[duration]
})
proba = model.predict_proba(input_df)[0]
if st.button("Predict Credit Risk"):
    pred = model.predict(input_df)[0]
    
    if pred == 1:
        st.success("The Predicted Credit risk is: **GOOD**")
        st.write(f"The model is **{proba[1]*100:.2f}% confident** that the applicant has GOOD creditworthiness based on the provided features.")

    else:
        st.error("The Predicted Credit risk is: **BAD**")
        st.write(f"The model is **{proba[1]*100:.2f}% confident** that the applicant has BAD creditworthiness based on the provided features.")
