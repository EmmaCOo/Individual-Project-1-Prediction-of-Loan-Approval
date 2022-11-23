#!/usr/bin/env python
# coding: utf-8

# In[1]:


# #app.py

# import streamlit as st
# import pandas as pd
# import joblib

# # Title
# st.header("LOAN APPROVAL PREDICTION APP")

# # Input bar 1
# height = st.number_input("Enter Height")

# # Input bar 2
# weight = st.number_input("Enter Weight")

# # Dropdown input
# eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

# # If button is pressed
# if st.button("Submit"):
    
#     # Unpickle classifier
#     clf = joblib.load("clf.pkl")
    
#     # Store inputs into dataframe
#     X = pd.DataFrame([[height, weight, eyes]], 
#                      columns = ["Height", "Weight", "Eyes"])
#     X = X.replace(["Brown", "Blue"], [1, 0])
    
#     # Get prediction
#     prediction = clf.predict(X)[0]
    
#     # Output prediction
#     st.text(f"This instance is a {prediction}")
    


# In[2]:


#loan_appy
import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing



#TITLE

st.header('LOAN APPROVAL PREDICTION APP')

income = st.number_input('Enter Applicant Income')

co_income = st.number_input('Enter Co-Applicant Income')

loan_amount = st.number_input('Enter Desired Loan Amount')

loan_term = st.number_input('Enter Loan Term in Months')

# Dropdown input
gender = st.selectbox("Select Your Gender", ('Male', 'Female'))

marital_status = st.selectbox('Select Your Marital Status', ('Yes','No'))

dependents = st.selectbox('Select No of Dependents',('0','1','2','3'))

education = st.selectbox('Enter Your Education', ('Graduate','Not Graduate'))

self_employed = st.selectbox('Self_Employed', ('Yes','No'))

credit_history = st.selectbox('Credit History', ('Yes', 'No'))

property_area = st.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))
                              

#if button is pressed
    
# If button is pressed
if st.button("Submit"):
    
    clf = joblib.load('loan_model.pkl')
    
    #Store inputs into dataframe
    X = pd.DataFrame([[income, co_income, loan_amount, loan_term, gender, marital_status,
                      dependents, education, self_employed, credit_history, property_area]],
                    columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term',
                               'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                               'Credit_History','Property_Area'])
    #Yes: 1, No: 0 
    X = X.replace(['Male','Female'],[1, 0])    #label encoding of gender
    X = X.replace(['Yes','No'],[1, 0])         # encoding of marital
    X = X.replace(['Graduate','Not Graduate'],[0, 1])  #encoding of Education
    X = X.replace(['Urban','Rural','Semiurban'], [2, 0, 1])  # semiurban 1, urban =  2,  rural = 0
    
    scaler = preprocessing.Normalizer()
    X[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]=scaler.fit_transform(X[['ApplicantIncome',
                                                                                               'CoapplicantIncome',
                                                                                               'LoanAmount',
                                                                                               'Loan_Amount_Term']])
    
        
    
    # Get prediction
    prediction = clf.predict(X)[0]
    prediction_prob = clf.predict_proba(X)
    
    # Output prediction
    if prediction == 1:
        st.subheader('Your loan application will be APPROVED with a probability of {}%'.format(round(prediction_prob[0][1]*100 , 3)))
    else:
        st.subheader('Your loan application will be REJECTED with a probability of {}%'.format(round(prediction_prob[0][0]*100 , 3)))
    


# sex = 0 if sex == 'Male' else 1
# f_class , s_class , t_class = 0,0,0
# if p_class == 'First Class':
# 	f_class = 1
# elif p_class == 'Second Class':
# 	s_class = 1
# else:
# 	t_class = 1
# input_data = scaler.transform([[sex , age, f_class , s_class, t_class]])
# prediction = model.predict(input_data)
# predict_probability = model.predict_proba(input_data)

    
    


# In[ ]:




