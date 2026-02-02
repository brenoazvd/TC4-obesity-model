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
    st.caption("Este grafico mostra o que mais pesa no resultado (nao significa causa).")
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
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    class_map = {
        "Insufficient_Weight": "Abaixo do peso",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso I",
        "Overweight_Level_II": "Sobrepeso II",
        "Obesity_Type_I": "Obesidade I",
        "Obesity_Type_II": "Obesidade II",
        "Obesity_Type_III": "Obesidade III",
    }
    dist = data['Obesity'].value_counts().sort_index()
    labels = [class_map.get(str(x), str(x)) for x in dist.index]
    sns.barplot(x=labels, y=dist.values, ax=ax2, palette='viridis')
    ax2.set_xlabel("Nivel de obesidade")
    ax2.set_ylabel("Quantidade")
    ax2.set_title("Distribuicao das classes na base")
    ax2.tick_params(axis="x", labelrotation=20)
    st.pyplot(fig2)
    st.caption("Se uma categoria aparece muito, o resultado tende a puxar mais para ela.")

    # 3. Relacao entre habitos e nivel de obesidade (insights clinicos)
    st.subheader("3. Relacoes entre habitos e nivel de obesidade")
    st.caption("Graficos simples + interpretacao direta para equipe de saude.")

    class_order = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]
    class_map_pt = {
        "Insufficient_Weight": "Abaixo do peso",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso I",
        "Overweight_Level_II": "Sobrepeso II",
        "Obesity_Type_I": "Obesidade I",
        "Obesity_Type_II": "Obesidade II",
        "Obesity_Type_III": "Obesidade III",
    }

    # localizar colunas
    col_faf = next((c for c in data.columns if "atividade" in c.lower() and "frequ" in c.lower()), None)
    col_ch2o = next((c for c in data.columns if "agua" in c.lower()), None)
    col_tue = next((c for c in data.columns if "tempo gasto em atividades fisicas" in c.lower()), None)
    col_fh = "family_history" if "family_history" in data.columns else None

    df_plot = data.copy()
    df_plot["Obesity_pt"] = df_plot["Obesity"].map(class_map_pt).fillna(df_plot["Obesity"].astype(str))
    order_pt = [class_map_pt[c] for c in class_order]

    # 2) Obesidade x atividade fisica (FAF) - boxplot
    if col_faf:
        fig_faf, ax_faf = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_plot, x="Obesity_pt", y=col_faf, order=order_pt, ax=ax_faf)
        ax_faf.set_xlabel("Nivel de obesidade")
        ax_faf.set_ylabel("Frequencia de atividade fisica")
        ax_faf.set_title("Obesidade x atividade fisica")
        ax_faf.tick_params(axis="x", labelrotation=20)
        st.pyplot(fig_faf)
        st.caption(
            "Interpretacao: niveis mais altos tendem a mostrar menor atividade fisica."
        )

    # 3) Obesidade x consumo de agua (CH2O) - barras empilhadas
    if col_ch2o:
        df_plot["ch2o_bin"] = df_plot[col_ch2o].round().clip(1, 3).astype(int)
        tab_ch2o = pd.crosstab(df_plot["Obesity_pt"], df_plot["ch2o_bin"], normalize="index") * 100
        tab_ch2o = tab_ch2o.reindex(index=order_pt, fill_value=0)
        fig_ch2o, ax_ch2o = plt.subplots(figsize=(10, 4))
        tab_ch2o.plot(kind="bar", stacked=True, ax=ax_ch2o, colormap="Blues", legend=True)
        ax_ch2o.set_xlabel("Nivel de obesidade")
        ax_ch2o.set_ylabel("% das pessoas")
        ax_ch2o.set_title("Obesidade x consumo de agua")
        ax_ch2o.tick_params(axis="x", labelrotation=20)
        ax_ch2o.legend(title="Consumo de agua (1-3)", loc="upper right")
        st.pyplot(fig_ch2o)
        st.caption(
            "Interpretacao: baixo consumo hidrico aparece mais em niveis elevados."
        )

    # 4) Obesidade x historico familiar - barras agrupadas
    if col_fh:
        tab_fh = pd.crosstab(df_plot["Obesity_pt"], df_plot[col_fh], normalize="index") * 100
        tab_fh = tab_fh.reindex(index=order_pt, fill_value=0)
        tab_fh = tab_fh.rename(columns={"yes": "Sim", "no": "Nao"})
        fig_fh, ax_fh = plt.subplots(figsize=(10, 4))
        tab_fh.plot(kind="bar", ax=ax_fh, colormap="Greens", legend=True)
        ax_fh.set_xlabel("Nivel de obesidade")
        ax_fh.set_ylabel("% das pessoas")
        ax_fh.set_title("Obesidade x historico familiar")
        ax_fh.tick_params(axis="x", labelrotation=20)
        ax_fh.legend(title="Historico familiar", loc="upper right")
        st.pyplot(fig_fh)
        st.caption(
            "Interpretacao: historico familiar aparece com maior frequencia em casos mais graves."
        )

    # 5) Obesidade x uso de tecnologia (TUE) - boxplot
    if col_tue:
        fig_tue, ax_tue = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_plot, x="Obesity_pt", y=col_tue, order=order_pt, ax=ax_tue)
        ax_tue.set_xlabel("Nivel de obesidade")
        ax_tue.set_ylabel("Tempo em telas (TUE)")
        ax_tue.set_title("Obesidade x uso de tecnologia (TUE)")
        ax_tue.tick_params(axis="x", labelrotation=20)
        st.pyplot(fig_tue)
        st.caption(
            "Interpretacao: mais tempo em telas tende a acompanhar niveis mais altos."
        )
