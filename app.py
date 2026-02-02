import streamlit as st
import pandas as pd
import joblib
import sklearn
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import pyplot as plt

import seaborn as sns

COL_FAVC = "Consumo de alimentos com alto teor cal\u00f3rico"
COL_FCVC = "Frequ\u00eancia de consumo de vegetais"
COL_NCP = "N\u00famero de refei\u00e7\u00f5es por dia"
COL_CAEC = "Consumo de alimentos entre as refei\u00e7\u00f5es"
COL_SMOKE = "Fuma"
COL_SCC = "Consumo de bebidas alco\u00f3licas"
COL_FAF = "Frequ\u00eancia de atividade f\u00edsica"
COL_TUE = "Tempo gasto em atividades f\u00edsicas"
COL_CALC = "Tempo gasto em atividades sedent\u00e1rias"
COL_MTRANS = "Meio de transporte utilizado"
COL_CH2O = "Consumo de \u00e1gua di\u00e1rio"

@st.cache_data
def load_dataset():
    data = pd.read_csv('Obesity.csv')
    ajusta_nomes={"FAVC": COL_FAVC,
                  "FCVC": COL_FCVC,
                  "NCP": COL_NCP,
                  "CAEC": COL_CAEC,
                  "SMOKE": COL_SMOKE,
                  "SCC": COL_SCC,
                  "FAF": COL_FAF,
                  "TUE": COL_TUE,
                  "CALC": COL_CALC,
                  "MTRANS": COL_MTRANS,
                  "CH2O": COL_CH2O
                  }

                    
    data.rename(columns=ajusta_nomes, inplace=True)
    X = data.drop(columns=['Obesity', 'Weight', 'Height'], errors='ignore')
    
    return data,X


pipeline = joblib.load('obesity_model_pipeline.joblib')

st.title("Predicao de Nivel de Obesidade")
st.write("Preencha os dados para prever o nivel de obesidade.")


# O comando retorna dois objetos que guardamos em 'tab_simulador' e 'tab_dashboard'
tab_simulador, tab_dashboard = st.tabs(["Simulador", "Dashboard"])



with tab_simulador:
    st.header("Simulador de Nivel de Obesidade")
    with st.form("Questionario"):
        gender_map = {"Masculino": "Male", "Feminino": "Female"}
        gender_label = st.selectbox("Genero", list(gender_map.keys()))
        gender = gender_map[gender_label]
        age = st.number_input("Idade", min_value=1, max_value=120,value=25)

        yes_no_map = {"Sim": "yes", "Nao": "no"}
        family_history_label = st.selectbox("Historico familiar de sobrepeso", list(yes_no_map.keys()))
        family_history = yes_no_map[family_history_label]
        frequent_fast_food_label = st.selectbox("Consumo frequente de fast food", list(yes_no_map.keys()))
        frequent_fast_food = yes_no_map[frequent_fast_food_label]
        escala_1_3 = {"Baixo": 1, "Medio": 2, "Alto": 3}
        escala_0_3 = {"Nenhum": 0, "Baixo": 1, "Medio": 2, "Alto": 3}
        escala_0_2 = {"Nenhum": 0, "Baixo": 1, "Medio": 2}
        frequent_vegetables_label = st.selectbox("Consumo frequente de vegetais", list(escala_1_3.keys()))
        frequent_vegetables = escala_1_3[frequent_vegetables_label]
        number_of_meals_label = st.selectbox("Numero de refeicoes por dia", list(escala_1_3.keys()))
        number_of_meals = escala_1_3[number_of_meals_label]
        between_meals_map = {
            "Nunca": "no",
            "As vezes": "Sometimes",
            "Frequente": "Frequently",
            "Sempre": "Always",
        }
        food_between_meals_label = st.selectbox("Consumo de alimentos entre as refeicoes", list(between_meals_map.keys()))
        food_between_meals = between_meals_map[food_between_meals_label]
        smokes_label = st.selectbox("Fuma", list(yes_no_map.keys()))
        smokes = yes_no_map[smokes_label]
        water_intake_label = st.selectbox("Consumo de agua diario", list(escala_1_3.keys()))
        water_intake = escala_1_3[water_intake_label]
        alcohol_consumption_label = st.selectbox("Consumo de bebidas alcoolicas", list(yes_no_map.keys()))
        alcohol_consumption = yes_no_map[alcohol_consumption_label]
        physical_activity_frequency_label = st.selectbox("Frequencia de atividade fisica", list(escala_0_3.keys()))
        physical_activity_frequency = escala_0_3[physical_activity_frequency_label]
        time_spent_exercising_label = st.selectbox("Tempo em telas (0-2)", list(escala_0_2.keys()))
        time_spent_exercising = escala_0_2[time_spent_exercising_label]
        time_spent_sitting_label = st.selectbox(
            "Frequencia em atividades sedentarias",
            list(between_meals_map.keys()),
        )
        time_spent_sitting = between_meals_map[time_spent_sitting_label]
        transport_map = {
            "Automovel": "Automobile",
            "Moto": "Motorbike",
            "Bicicleta": "Bike",
            "Transporte publico": "Public_Transportation",
            "Caminhando": "Walking",
        }
        transport_label = st.selectbox("Meio de transporte utilizado", list(transport_map.keys()))
        transportation_mode = transport_map[transport_label]





        submit_button=st.form_submit_button("Calcular nivel de obesidade")



    if submit_button:
        input_data= pd.DataFrame({
            'Gender':[gender],
            'Age':[age],
            'family_history':[family_history],
            COL_FAVC:[frequent_fast_food],
            COL_FCVC:[frequent_vegetables],
            COL_NCP:[number_of_meals],
            COL_CAEC:[food_between_meals],
            'Fuma':[smokes],
            COL_CH2O:[water_intake],
            COL_SCC:[alcohol_consumption],
            COL_FAF:[physical_activity_frequency],
            COL_TUE:[time_spent_exercising],
            COL_CALC:[time_spent_sitting],
            COL_MTRANS:[transportation_mode]
        })


        st.write("Processando dados")

        try:
            prediction= pipeline.predict(input_data)[0]
            class_map_pt = {
                0: "Abaixo do peso",
                1: "Peso normal",
                2: "Sobrepeso I",
                3: "Sobrepeso II",
                4: "Obesidade I",
                5: "Obesidade II",
                6: "Obesidade III",
            }
            pred_label = class_map_pt.get(int(prediction), str(prediction))
            st.success(f"Resultado da analise: {pred_label}")

            if prediction in (0,1,2):
                st.info("Nivel de obesidade baixo. Mantenha um estilo de vida saudavel!")
            elif prediction ==3:
                st.info("Nivel de obesidade moderado. Considere adotar habitos mais saudaveis.")
            elif prediction in (4,5,6):
                st.warning("Recomenda-se consultar um profissional de saude para orientacao adequada.")

    
        except Exception as e:
            

            st.error(f"Ocorreu um erro durante a predicao: {e,prediction}")
            st.warning("Por favor, verifique os dados e tente novamente.")



# ==============================================================================
# ABA 2: DASHBOARD DINAMICO
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
        f"scaler__{COL_FCVC}": "Frequencia de vegetais",
        f"scaler__{COL_NCP}": "Numero de refeicoes/dia",
        f"scaler__{COL_CH2O}": "Consumo de agua diario",
        f"scaler__{COL_FAF}": "Frequencia de atividade fisica",
        f"scaler__{COL_TUE}": "Tempo em atividade fisica",
        "categorical__Gender": "Genero",
        "categorical__family_history": "Historico familiar",
        f"categorical__{COL_FAVC}": "Fast food frequente",
        "categorical__Fuma": "Fuma",
        f"categorical__{COL_SCC}": "Consumo de alcool",
        f"categorical__{COL_MTRANS}": "Meio de transporte",
        f"categorical_order__{COL_CAEC}": "Comer entre refeicoes",
        f"categorical_order__{COL_CALC}": "Tempo sedentario",
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
    col_faf = COL_FAF if COL_FAF in data.columns else None
    col_ch2o = COL_CH2O if COL_CH2O in data.columns else None
    col_tue = COL_TUE if COL_TUE in data.columns else None
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
        ch2o_colors = ["#e0f3f8", "#abd9e9", "#74add1"]
        tab_ch2o = tab_ch2o.rename(columns={1: "Baixo", 2: "Medio", 3: "Alto"})
        tab_ch2o.plot(kind="bar", stacked=True, ax=ax_ch2o, color=ch2o_colors, legend=True)
        ax_ch2o.set_xlabel("Nivel de obesidade")
        ax_ch2o.set_ylabel("% das pessoas")
        ax_ch2o.set_title("Obesidade x consumo de agua")
        ax_ch2o.tick_params(axis="x", labelrotation=20)
        ax_ch2o.legend(title="Consumo de agua", loc="upper right")
        st.pyplot(fig_ch2o)
        st.caption(
            "Interpretacao: baixo consumo hidrico aparece mais em niveis elevados."
        )

    # 4) Obesidade x historico familiar - % com historico familiar
    if col_fh:
        tab_fh = pd.crosstab(df_plot["Obesity_pt"], df_plot[col_fh], normalize="index") * 100
        tab_fh = tab_fh.reindex(index=order_pt, fill_value=0)
        tab_fh = tab_fh.rename(columns={"yes": "Sim", "no": "Nao"})
        perc_sim = tab_fh["Sim"] if "Sim" in tab_fh.columns else None
        if perc_sim is not None:
            fig_fh, ax_fh = plt.subplots(figsize=(10, 4))
            sns.barplot(x=perc_sim.index, y=perc_sim.values, ax=ax_fh, palette="Greens")
            ax_fh.set_xlabel("Nivel de obesidade")
            ax_fh.set_ylabel("% com historico familiar")
            ax_fh.set_title("Historico familiar por nivel de obesidade")
            ax_fh.tick_params(axis="x", labelrotation=20)
            st.pyplot(fig_fh)
            st.caption(
                "Interpretacao: maior % com historico familiar nos niveis mais graves."
            )
        else:
            st.warning("Nao foi possivel calcular o historico familiar.")

    # 5) Obesidade x uso de tecnologia (TUE) - boxplot
    if col_tue:
        fig_tue, ax_tue = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_plot, x="Obesity_pt", y=col_tue, order=order_pt, ax=ax_tue)
        ax_tue.set_xlabel("Nivel de obesidade")
        ax_tue.set_ylabel("Tempo em telas (0-2)")
        ax_tue.set_title("Obesidade x uso de tecnologia (TUE)")
        ax_tue.tick_params(axis="x", labelrotation=20)
        st.pyplot(fig_tue)
        st.caption(
            "Interpretacao: mais tempo em telas tende a acompanhar niveis mais altos."
        )

    st.subheader("4. Insights praticos para equipe medica")
    st.markdown(
        "- Triagem precoce: pacientes com historico familiar + baixa atividade fisica merecem acompanhamento mais proximo.\n"
        "- Prevencao simples: aumento de agua e reducao do tempo em telas sao acoes de baixo custo.\n"
        "- Educacao em saude: reforcar habitos saudaveis pode reduzir risco em grupos vulneraveis."
    )
