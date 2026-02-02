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


    st.header("Dashboard de Analise de Obesidade")
    st.write("Analise explicativa do modelo + insights para equipe de saude")
    # Acessar os passos do pipeline
    step_model = pipeline.named_steps['model']
    step_preprocessor = pipeline.named_steps['scaling'] # Ou 'preprocessor', confira seu codigo
    
    feature_names = step_preprocessor.get_feature_names_out()
    feature_label_map = {
        "scaler__Age": "Idade",
        "scaler__Frequência de consumo de vegetais": "Frequencia de vegetais",
        "scaler__Número de refeições por dia": "Numero de refeicoes/dia",
        "scaler__Consumo de água diário": "Consumo de agua diario",
        "scaler__Frequência de atividade física": "Frequencia de atividade fisica",
        "scaler__Tempo gasto em atividades físicas": "Tempo em atividade fisica",
        "categorical__Gender": "Genero",
        "categorical__family_history": "Historico familiar",
        "categorical__Consumo de alimentos com alto teor calórico": "Fast food frequente",
        "categorical__Fuma": "Fuma",
        "categorical__Consumo de bebidas alcoólicas": "Consumo de alcool",
        "categorical__Meio de transporte utilizado": "Meio de transporte",
        "categorical_order__Consumo de alimentos entre as refeições": "Comer entre refeicoes",
        "categorical_order__Tempo gasto em atividades sedentárias": "Tempo sedentario",
    }
    # Grafico 1: Importancia das features
    st.subheader("1. O que mais impacta o risco?")
    importances = step_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp["Feature_label"] = df_imp["Feature"].map(feature_label_map).fillna(df_imp["Feature"])
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

    # Plotar
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature_label', palette='viridis',hue='Feature_label', dodge=False)
    ax1.set_title("Top 10 Fatores de Risco")
    ax1.set_xlabel("Importancia (quanto maior, mais influencia)")
    ax1.set_ylabel("Fator")
    st.pyplot(fig1)
    st.caption("Fatores com maior influencia no resultado (nao indica causalidade).")
    top3 = df_imp['Feature_label'].head(3).tolist()
    st.markdown(
        "Insights (equipe medica):\n"
        f"- Principais variaveis do modelo: {', '.join(top3)}.\n"
        "- Use como triagem: avaliar comportamento alimentar e atividade fisica.\n"
        "- Importancia do modelo nao e causalidade; validar clinicamente."
    )
    st.markdown(
        "Interpretacao curta:\n"
        "- Quanto maior a barra, maior a influencia no resultado.\n"
        "- Fatores do topo merecem atencao em protocolos de triagem."
    )

    st.subheader("2. Distribuicao dos niveis de obesidade na base")
    data,X = load_dataset()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    dist = data['Obesity'].value_counts().sort_index()
    sns.barplot(x=dist.index.astype(str), y=dist.values, ax=ax2, palette='viridis')
    ax2.set_xlabel("Classe de Obesidade")
    ax2.set_ylabel("Quantidade")
    ax2.set_title("Distribuicao das classes na base")
    st.pyplot(fig2)
    st.caption("Classes mais frequentes podem influenciar a predicao do modelo.")

    # 1. Configurar o que queremos ver
    # Usar indices para evitar erro de nome nas features
    nomes_features = step_preprocessor.get_feature_names_out()
    nomes_lower = [n.lower() for n in nomes_features]
    idx_vegetais = next((i for i, n in enumerate(nomes_lower) if "vegetais" in n), None)
    idx_atividade = next((i for i, n in enumerate(nomes_lower) if "atividade" in n and "sedent" not in n), None)
    pdp_features = [i for i in [idx_atividade, idx_vegetais] if i is not None]
    if len(pdp_features) < 2:
        pdp_features = list(range(min(2, len(nomes_features))))

    # 2. Plotar
    fig, ax = plt.subplots(figsize=(10, 6))
    X_sample = X.sample(n=min(500, len(X)), random_state=42)

    # A classe 6 e a Obesidade Tipo III (o caso grave)
    # Se o seu modelo for binario ou diferente, ajuste o target.
    try:
        PartialDependenceDisplay.from_estimator(
            step_model,                # Sua Random Forest
            step_preprocessor.transform(X_sample), # Dados transformados (amostra)
            features=pdp_features, # Apenas duas features para nao travar
            feature_names=nomes_features, # Nomes das colunas
            target=6, # Focando na Classe 6 (Obesidade Grave)
            ax=ax
        )
        st.pyplot(fig)
        st.caption("Grafico de dependencia parcial para duas variaveis chave.")
    except Exception as e:
        st.warning(f"Nao foi possivel gerar o grafico de dependencia parcial: {e}")
