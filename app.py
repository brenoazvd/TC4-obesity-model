from altair import Data
import streamlit as st
import pandas as pd
import joblib
import sklearn
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import pyplot as plt

import seaborn as sns

@st.cache_data
def load_dataset():
    data = pd.read_csv('Obesity.csv')
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

                    
    data.rename(columns=ajusta_nomes, inplace=True)
    X = data.drop(columns=['Obesity', 'Weight', 'Height'], errors='ignore')
    
    return data,X


pipeline = joblib.load('obesity_model_pipeline.joblib')

st.title("Obesity Level Prediction")
st.write("Enter your details to predict your obesity level.")


# O comando retorna dois objetos que guardamos em 'tab_simulador' e 'tab_dashboard'
tab_simulador, tab_dashboard = st.tabs(["🧬 Simulador", "📊 Dashboard"])



with tab_simulador:
    st.header("Simulador de Nível de Obesidade")
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



# ==============================================================================
# ABA 2: DASHBOARD DINÂMICO
# ==============================================================================
with tab_dashboard:


    st.header("Dashboard de Análise de Obesidade")
    st.write("Análise Explicativa do Modelo")
    # Acessar os passos do pipeline
    step_model = pipeline.named_steps['model']
    step_preprocessor = pipeline.named_steps['scaling'] # Ou 'preprocessor', confira seu código
    
    feature_names = step_preprocessor.get_feature_names_out()
    # Gráfico 1: Importância das Features
    st.subheader("1. O que mais impacta o risco?")
    importances = step_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

    

    # Plotar
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis',hue='Feature', dodge=False)
    ax1.set_title("Top 10 Fatores de Risco")
    st.pyplot(fig1)
    st.caption("Fatores que mais influenciam a classificação de Obesidade Tipo III.")
    st.markdown("---")

    st.subheader("2. Distribuição dos Níveis de Obesidade na Base de Dados")
    data,X = load_dataset()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    

    # 1. Configurar o que queremos ver
    # Vamos ver o impacto da "Atividade Física" (FAF) e "Vegetais" (FCVC)
    features_para_ver = ['num__FAF', 'num__FCVC'] 

    # Precisamos pegar os nomes corretos que saíram do transformer
    # Se der erro de nome, imprima 'nomes_features' para conferir
    nomes_features = step_preprocessor.get_feature_names_out()

    print("Gerando gráfico de Dependência Parcial (Causa e Efeito)...")

    # 2. Plotar
    fig, ax = plt.subplots(figsize=(12, 20))

    # A classe 6 é a Obesidade Tipo III (o caso grave)
    # Se o seu modelo for binário ou diferente, ajuste o target.
    display = PartialDependenceDisplay.from_estimator(
        step_model,                # Sua Random Forest
        step_preprocessor.transform(X), # Seus dados transformados
        features=nomes_features, # Vamos varrer pelos índices
        feature_names=nomes_features, # Nomes das colunas
        target=6, # Focando na Classe 6 (Obesidade Grave)
        ax=ax
    )


    st.pyplot(fig)
    