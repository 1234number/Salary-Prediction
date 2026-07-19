import streamlit as st
import joblib
import numpy as np

# -------------------- Load Model --------------------

model = joblib.load("Annual Salary Prediction.pkl")

# -------------------- Page Settings --------------------

st.set_page_config(
    page_title="Annual Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Annual Salary Prediction")
st.write("Predict an employee's annual salary using Machine Learning.")

st.markdown("---")

# -------------------- Input Fields --------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

education = st.selectbox(
    "Education",
    ["Bachelor", "Diploma", "Master", "PhD"]
)

experience = st.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=40,
    value=2
)

department = st.selectbox(
    "Department",
    ["HR", "IT", "Marketing", "Operations", "Sales"]
)

job_level = st.selectbox(
    "Job Level",
    ["Junior", "Lead", "Manager", "Mid", "Senior"]
)

city = st.selectbox(
    "City",
    ["Chennai", "Delhi", "Hyderabad", "Mumbai"]
)

company_tenure = st.number_input(
    "Company Tenure",
    min_value=0,
    max_value=30,
    value=2
)

projects_completed = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=100,
    value=5
)

skill_score = st.slider(
    "Skill Score",
    min_value=0,
    max_value=100,
    value=50
)

# -------------------- Encoding --------------------

education_map = {
    "Bachelor": 0,
    "Diploma": 1,
    "Master": 2,
    "PhD": 3
}

department_map = {
    "HR": 0,
    "IT": 1,
    "Marketing": 2,
    "Operations": 3,
    "Sales": 4
}

job_level_map = {
    "Junior": 0,
    "Lead": 1,
    "Manager": 2,
    "Mid": 3,
    "Senior": 4
}

city_map = {
    "Chennai": 0,
    "Delhi": 1,
    "Hyderabad": 2,
    "Mumbai": 3
}

# -------------------- Prediction --------------------

if st.button("Predict Salary"):

    features = np.array([[
        age,
        education_map[education],
        experience,
        department_map[department],
        job_level_map[job_level],
        city_map[city],
        company_tenure,
        projects_completed,
        skill_score
    ]])

    prediction = model.predict(features)

    salary = prediction[0]

    st.success(f"💰 Predicted Annual Salary : {salary:.2f} LPA")

    st.info(f"Approximate Annual Salary : ₹ {salary*100000:,.0f}")

st.markdown("---")
st.caption("Developed using Streamlit & Scikit-Learn")