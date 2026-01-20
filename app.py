import streamlit as st
import pandas as pd
import joblib
import sklearn



pipeline = joblib.load('obesity_model_pipeline.joblib')

st.title("Obesity Level Prediction")
st.write("Enter your details to predict your obesity level.")

with st.form("Questionare"):
    gender = st.selectbox("Gênero",['Male','Female'])
    age = st.number_input("Idade", min_value=1, max_value=120,value=25)

    family_history= st.selectbox("Histórico Familiar de sobrepeso",["yes","no"])
    frequent_fast_food= st.selectbox("Consumo frequente de fast food",["yes","no"])
    frequent_vegetables= st.number_input("Consumo Frequente de Vegetais(1-3)", min_value=1, max_value=3,step=1)
    number_of_meals = st.number_input("Número de refeições por dia (1-3)", min_value=1, max_value=3,step=1)
    food_between_meals=st.selectbox("Consumo de alimentos entre as refeições",["Sometimes","Frequently","Always","no"])
    smokes=st.selectbox("Fuma",["yes","no"])
    water_intake=st.number_input("Consumo de água diário (1-3)", min_value=1, max_value=3,step=1)
    alcohol_consumption=st.selectbox("Consumo de bebidas alcoólicas",["yes","no"])
    physical_activity_frequency=st.number_input("Frequência de atividade física (0-3)", min_value=0, max_value=3,step=1)
    time_spent_exercising=st.number_input("Tempo gasto em atividades físicas (0-3)", min_value=0, max_value=3,step=1)
    time_spent_sitting=st.selectbox("Frequencia em atividades sedentárias (1-3)",['Sometimes', 'Frequently', 'Always', 'no'])
    transportation_mode=st.selectbox("Meio de transporte utilizado",["Automobile","Motorbike","Bike","Public_Transportation","Walking"])





    submit_button=st.form_submit_button("Calcular nível de obesidade")



if submit_button:
    input_data= pd.DataFrame({
        'Gender':[gender],
        'Age':[age],
        'family_history':[family_history],
        'Consumo de alimentos com alto teor calórico':[frequent_fast_food],
        'Frequência de consumo de vegetais':[frequent_vegetables],
        'Número de refeições por dia':[number_of_meals],
        'Consumo de alimentos entre as refeições':[food_between_meals],
        'Fuma':[smokes],
        'Consumo de água diário':[water_intake],
        'Consumo de bebidas alcoólicas':[alcohol_consumption],
        'Frequência de atividade física':[physical_activity_frequency],
        'Tempo gasto em atividades físicas':[time_spent_exercising],
        'Tempo gasto em atividades sedentárias':[time_spent_sitting],
        'Meio de transporte utilizado':[transportation_mode]
    })


    st.write("Processando dados")

    try:
        prediction= pipeline.predict(input_data)[0]

        st.success(f"Resultado da análise:{prediction}")

        if prediction in (0,1,2):
            st.info("Nível de obesidade baixo. Mantenha um estilo de vida saudável!")
        elif prediction ==3:
            st.info("Nível de obesidade moderado. Considere adotar hábitos mais saudáveis.")
        elif prediction in (4,5,6):
            st.warning("Recomenda-se consultar um profissional de saúde para orientação adequada.")

 
    except Exception as e:
        
        
        st.error(f"Ocorreu um erro durante a predição: {e,prediction}")
        st.warning("Por favor, verifique os dados e tente novamente.")