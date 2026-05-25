import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── 1. CONFIGURAÇÃO DA PÁGINA E INTERFACE ──────────────────────────────────
st.set_page_config(
    page_title="Sistema de Diagnóstico - Hospital",
    page_icon="🏥",
    layout="centered"
)

st.title("Sistema Inteligente de Diagnóstico de Obesidade")
st.markdown("""
Este sistema utiliza um modelo de **Machine Learning (Random Forest)** de alta precisão 
para auxiliar a equipe médica na triagem e diagnóstico precoce de pacientes.
""")
st.write("---")

# ── 2. CARREGAMENTO DOS ARTEFATOS DO MODELO ───────────────────────────────
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_obesidade.pkl')
    precomputador = joblib.load('precomputador_obesidade.pkl')
    mapeador = joblib.load('mapeador_alvo_obesidade.pkl')
    return modelo, precomputador, mapeador

try:
    modelo, precomputador, mapeador = carregar_modelo()
except FileNotFoundError:
    st.error("Erro: Arquivos do modelo não encontrados! Certifique-se de rodar o script de modelagem primeiro.")
    st.stop()

# ── 3. FORMULÁRIO DE ENTRADA DE DADOS (VISÃO DE NEGÓCIO) ──────────────────
st.subheader("Ficha Clínica do Paciente")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Sexo Biológico", ["Male", "Female"])
    age = st.number_input("Idade (anos)", min_value=14, max_value=100, value=25)
    height = st.number_input("Altura (metros)", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)

with col2:
    family_history = st.selectbox("Histórico Familiar de Excesso de Peso?", ["yes", "no"])
    favc = st.selectbox("Consumo Frequente de Alimentos Hipercalóricos?", ["yes", "no"])
    fcvc = st.slider("Frequência de Consumo de Vegetais (1=Raro, 3=Sempre)", 1, 3, 2)
    ncp = st.slider("Número de Refeições Principais por Dia", 1, 4, 3)

st.markdown("#### Estilo de Vida e Hábitos")
col3, col4 = st.columns(2)

with col3:
    caec = st.selectbox("Come entre as refeições (Beliscar)?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Hábito de Fumar?", ["yes", "no"])
    ch2o = st.slider("Consumo Diário de Água (1=<1L, 2=1-2L, 3=>2L)", 1, 3, 2)

with col4:
    scc = st.selectbox("Monitora a Ingestão Calórica Diária?", ["yes", "no"])
    faf = st.slider("Atividade Física Semanal (0=Nenhuma, 3=Frequente)", 0, 3, 1)
    tue = st.slider("Tempo Diário de Uso de Telas (0=0-2h, 2=>5h)", 0, 2, 1)

calc = st.selectbox("Consumo de Bebida Alcoólica?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Meio de Transporte Habitual", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

# ── 4. PROCESSAMENTO E PREDIÇÃO EM TEMPO REAL ────────────────────────────
st.write("---")

if st.button("Calcular Diagnóstico Clínico", type="primary"):
    
    # 1. Monta o dicionário com as respostas exatamente nos nomes das colunas originais
    dados_paciente = {
        'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
        'family_history': [family_history], 'FAVC': [favc], 'FCVC': [fcvc], 'NCP': [ncp],
        'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf],
        'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
    }
    
    # Transformando em DataFrame
    df_paciente = pd.DataFrame(dados_paciente)
    
    # 2. Passa os dados pelo mesmo precomputador do treino
    dados_processados = precomputador.transform(df_paciente)
    
    # 3. Realiza a predição numérica
    predicao_num = modelo.predict(dados_processados)[0]
    
    # 4. Converte o número de volta para o texto oficial do diagnóstico
    diagnostico_final = mapeador.inverse_transform([predicao_num])[0]
    
    # Substitui os underscores por espaços para ficar estético na tela do médico
    diagnostico_formatado = diagnostico_final.replace('_', ' ')
    
    # 5. Exibe o resultado com formatação visual baseada no risco
    st.subheader("Resultado da Avaliação:")
    
    if "Obesity" in diagnostico_final:
        st.error(f"Alerta Clínico: O paciente apresenta critérios para **{diagnostico_formatado}**.")
    elif "Overweight" in diagnostico_final:
        st.warning(f"Atenção: O paciente encontra-se na faixa de **{diagnostico_formatado}**.")
    else:
        st.success(f"Diagnóstico Estável: O paciente apresenta classificação de **{diagnostico_formatado}**.")