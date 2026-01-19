import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load('..\model\obesity_model_pipeline.joblib')

st.title("Obesity Level Prediction")
st.write("Enter your details to predict your obesity level.")

with st.form("Questionare"):
    gender = st.selectbox("Gênero",['Male','Female'])
    age = st.number_input("Idade", min_value=1, max_value=120,value=25)

    family_history= st.selectbox("Histórico Familiar de sobrepeso",["yes","no"])
    frequent_fast_food= st.selectbox("Consumo frequente de fast food",["yes","no"])
    frequent_vegetables= st.selectbox("Consumo Frequente de Vegetais",["yes","no"])
    number_of_meals = st.number_input("Número de refeições por dia (1-4)", min_value=1, max_value=4,step=0.1)
    food_between_meals=st.selectbox("Consumo de alimentos entre as refeições",["yes","no"])
    smokes=st.selectbox("Fuma",["yes","no"])
    water_intake=st.number_input("Consumo de água diário"



    submit_button=st.form_submit_button(label="Calcular nível de obesidade")

ajusta_nomes={"FAVC": "Consumo de alimentos com alto teor calórico",
  "FCVC": "Frequência de consumo de vegetais",
  "NCP": "Número de refeições por dia",
  "CAEC": "Consumo de alimentos entre as refeições",
  "SMOKE": "Fuma",
  "SCC": "Consumo de bebidas alcoólicas",
  "FAF": "Frequência de atividade física",
  "TUE": "Tempo gasto em atividades físicas",
  "CALC": "Tempo gasto em atividades sedentárias",
  "MTRANS": "Meio de transporte utilizado",
  "CH2O": "Consumo de água diário"
}


Gender	Age	family_history	FAVC	FCVC	NCP	CAEC	SMOKE	CH2O	SCC	FAF	TUE	CALC	MTRANS	Obesity