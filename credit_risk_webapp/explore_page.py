import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



@st.cache
def load_data():
    df = pd.read_csv('credit_risk_dataset.csv')
    df = df[['person_age', 'person_income', 'person_home_ownership','cb_person_cred_hist_length',
    'loan_intent','loan_amnt','loan_int_rate','loan_status']]

    df= df[(df['person_income']<150000) &(df['person_age']<85)]
    df = df.dropna()
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Data with aggregated values")
    data = df

    st.write("""#### Mean values for loan intrest rate, income and loan amount by Homeownership type""")

    cross = df.groupby(['person_home_ownership','loan_status'])[["loan_int_rate",'person_income','loan_amnt']].mean()
    st.dataframe(cross.style.highlight_max(axis=0))

    st.write("""#### Mean values for loan intrest rate, income and loan amount by Loan Intent type""")

    gb = df.groupby(['loan_intent','loan_status'])[["loan_int_rate",'person_income','loan_amnt']].mean()
    st.dataframe(gb.style.highlight_max(axis=0))

    st.write("""#### Loan Probablilty Defaults Vs Income with Loan Intent and Home Ownwership Status """)

    for i in set(data['loan_intent']):
        aa= data[data['loan_intent'].isin([i])]
        g = sns.catplot(x='person_home_ownership', y="person_income",data=aa,
                       saturation=1, kind="box", col = 'loan_status', row = 'loan_intent',
                       ci=None, aspect=1, linewidth=1)
        st.pyplot(g)
