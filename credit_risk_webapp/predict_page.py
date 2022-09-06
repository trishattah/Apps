import streamlit as st
import pickle
import numpy as np
import _pickle as cPickle
import gzip

def load_model():
    with gzip.open('saved_steps_zp.pkl.gz', 'rb') as f:
        data = cPickle.load(f)
        return data
    # with open('saved_steps.pkl','rb') as file:
    #     data = pickle.load(file)
    # return data

data = load_model()

rf = data['model']
le_loan_intent = data['le_loan_intent']
le_home_ownership = data['le_home_ownership']

def show_predict_page():
    st.title("Credit Loan Default Probability")
    st.write("""### We need some information to predict the loan default probability""")

    loan_intent = (
        'DEBTCONSOLIDATION',
        'EDUCATION',
        'HOMEIMPROVEMENT',
        'MEDICAL',
        'PERSONAL',
        'VENTURE'
    )

    home_ownership = (
        'MORTGAGE',
        'OTHER',
        'OWN',
        'RENT'
    )

    age = st.number_input('Applicant Age')

    income = st.number_input('Applicant Income')

    home_ownership = st.selectbox('Home Ownership Status', home_ownership)

    credit_history = st.number_input('Credit History Length')

    loan_intent = st.selectbox('Loan Intent', loan_intent)

    loan_amount = st.number_input('Loan Amount')

    loan_int_rate = st.number_input('Loan Intrest Rate')


    ok = st.button('Predict Loan Default')

    if ok:
        X = np.array([[age, income, home_ownership, credit_history, loan_intent, loan_amount, loan_int_rate]])
        X[:, 2] = le_home_ownership.transform(X[:, 2])
        X[:, 4] = le_loan_intent.transform(X[:, 4])
        X = X.astype(float)

        loan_default = rf.predict_proba(X)
        st.subheader(f"Estimated probability of default is {loan_default[0][1]:.2f}")
