import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack
import os

import warnings
warnings.filterwarnings('ignore')

# changing page main title and main icon(logo)
PAGE_CONFIG = {"page_title":"Titanic Survival Prediction", "page_icon":":ship:", "layout":"centered"}
st.set_page_config(**PAGE_CONFIG)   

st.sidebar.text("Created on Sat, Feb 4 2021")
st.sidebar.markdown("**@author:Sumit Kumar** :monkey_face:")
st.sidebar.markdown("[My Github](https://github.com/IMsumitkumar) :penguin:")
st.sidebar.markdown("[findingdata.ml](https://www.findingdata.ml/) :spider_web:")
st.sidebar.markdown("coded with :heart:")

# sidebar header
st.sidebar.subheader("Titanic Survival Prediction")

st.subheader("Titanic Survival Prediction")
st.title("")

model = pickle.load(open('models/decision_tree_model.sav', 'rb')) 

st.sidebar.image("https://i.imgur.com/zwJmDjo.jpg", width=300)

pclass = ('1st class', '2nd class', '3rd class')
sex = ('Male', 'Female')

five, six, seven = st.beta_columns(3)
sibsp = five.number_input('Number of siblings / spouses ?', value=1, min_value=1)
age = six.number_input("Age ?", value=1, min_value=1)
parch = seven.number_input("NUmber of parents / children?", value=1, min_value=1)

one, two = st.beta_columns(2)
pclass = one.selectbox(
"Pclass", pclass)

sex = two.selectbox(
"SEX ?", sex)

fare = st.number_input("Fare ?", min_value=0.0)

if pclass == '1st class':
    pclass = 1
elif pclass == '2nd class':
    pclass = 2
elif pclass == '3rd class':
    pclass = 3

if sex == 'Male':
    sex = 0
elif sex == "Female":
    sex = 1

fare = np.log(int(fare)+0.1)

query_vector = [[pclass, sex, int(age), int(sibsp), int(parch), fare]]

if st.button("Predict"):
    predicted_cls = model.predict(query_vector)
    st.markdown("Predicted Class")
    st.success(predicted_cls[0])
    predicted_probas = np.round(model.predict_proba(query_vector), 3)
    st.markdown("Predicted class probabilities : Not survived   :"+ str(predicted_probas[0][0]))
    st.markdown("Predicted class probabilities : Survived   :"+ str(predicted_probas[0][1]))