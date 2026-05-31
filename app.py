import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── 1. CONFIGURAÇÃO DA PÁGINA (Deve ser a primeira linha do Streamlit) ──────
st.set_page_config(
    page_title="Sistema de Diagnóstico - Hospital",
    page_icon="🏥",
    layout="wide" # Alterado para wide para o Dashboard ficar bonito de lado a lado
)

# Estilização CSS para os Cards do Dashboard
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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
    st.error("Erro: Arquivos do modelo não encontrados! Certifique-se de ter os arquivos .pkl na mesma pasta.")
    st.stop()

# ── 3. CARREGAMENTO E TRADUÇÃO DOS DADOS PARA O DASHBOARD ─────────────────
@st.cache_data
def carregar_dados_dashboard():
    df = pd.read_csv("Obesity.csv")
    # Dicionário de tradução para os gráficos ficarem profissionais na banca
    traducao = {
        'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
        'CH2O': 'Consumo de Água (Litros)', 'FAF': 'Atividade Física (Dias/Semana)',
        'family_history': 'Histórico Familiar', 'FAVC': 'Alimentos Calóricos',
        'Obesity': 'Nível de Obesidade'
    }
    df.rename(columns=traducao, inplace=True)
    mapa_sim_nao = {'yes': 'Sim', 'no': 'Não'}
    if 'Histórico Familiar' in df.columns:
        df['Histórico Familiar'] = df['Histórico Familiar'].map(mapa_sim_nao).fillna(df['Histórico Familiar'])
    if 'Alimentos Calóricos' in df.columns:
        df['Alimentos Calóricos'] = df['Alimentos Calóricos'].map(mapa_sim_nao).fillna(df['Alimentos Calóricos'])
    return df

try:
    df_clinico = carregar_dados_dashboard()
except Exception as e:
    st.error(f"Erro ao carregar Obesity.csv para o dashboard: {e}")
    df_clinico = pd.DataFrame()

# ── 4. CRIAÇÃO DAS ABAS DE NAVEGAÇÃO ──────────────────────────────────────
aba_dashboard, aba_predicao = st.tabs([
    "📊 Dashboard de Insights Clínicos", 
    "🩺 Simulador de Triagem Preditiva"
])

# =========================================================================
# ABA 1: DASHBOARD ANALÍTICO (Visão de Negócio / Gestão Hospitalar)
# =========================================================================
with aba_dashboard:
    st.title("📊 Painel de Análise Epidemiológica e Fatores de Risco")
    st.markdown("""
    Este painel estratégico auxilia a **diretoria hospitalar e as equipes médicas** a mapear os principais 
    gatilhos comportamentais e genéticos da base de pacientes, permitindo desenhar políticas de **medicina preventiva**.
    """)
    st.write("---")
    
    if not df_clinico.empty:
        # --- Cards de Métricas Principais (KPIs) ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><p style="margin:0; font-size:14px; color:#6c757d; font-weight:bold;">PACIENTES ANALISADOS</p><h2 style="margin:0; color:#17a2b8;">{len(df_clinico):,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card" style="border-left-color: #28a745;"><p style="margin:0; font-size:14px; color:#6c757d; font-weight:bold;">MÉDIA DE IDADE</p><h2 style="margin:0; color:#28a745;">{df_clinico["Idade"].mean():.1f} anos</h2></div>', unsafe_allow_html=True)
        with col3:
            pct_genetica = (df_clinico['Histórico Familiar'] == 'Sim').sum() / len(df_clinico) * 100
            st.markdown(f'<div class="metric-card" style="border-left-color: #dc3545;"><p style="margin:0; font-size:14px; color:#6c757d; font-weight:bold;">PROPENSÃO GENÉTICA</p><h2 style="margin:0; color:#dc3545;">{pct_genetica:.1f}%</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card" style="border-left-color: #ffc107;"><p style="margin:0; font-size:14px; color:#6c757d; font-weight:bold;">MÉDIA CONSUMO DE ÁGUA</p><h2 style="margin:0; color:#ffc107;">{df_clinico["Consumo de Água (Litros)"].mean():.1f} L/dia</h2></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Gráficos do Dashboard ---
        col_esq, col_dir = st.columns(2)
        with col_esq:
            st.subheader("🧬 O Peso da Genética no Desenvolvimento Clínico")
            ordem_obesidade = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
            fig_genetica = px.histogram(
                df_clinico, x="Nível de Obesidade", color="Histórico Familiar",
                barmode="group", category_orders={"Nível de Obesidade": ordem_obesidade},
                color_discrete_map={'Sim': '#dc3545', 'Não': '#6c757d'}
            )
            fig_genetica.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_genetica, use_container_width=True)
            st.info("**Insight Médico:** Nos níveis mais severos (Obesity Type II e III), a presença de histórico familiar positivo beira os 100%.")

        with col_dir:
            st.subheader("🏃‍♂️ Estilo de Vida: Hidratação vs Idade")
            fig_scatter = px.scatter(
                df_clinico, x="Idade", y="Consumo de Água (Litros)",
                color="Nível de Obesidade", size="Peso", opacity=0.7,
                category_orders={"Nível de Obesidade": ordem_obesidade}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("**Insight de Negócio:** Há uma grande concentração de pacientes jovens (18 a 30 anos) já em níveis de obesidade severa com baixo consumo de água.")

        st.markdown("---")
        st.subheader("🥦 Perfil de Atividade Física e Hábitos Alimentares")
        fig_box = px.box(
            df_clinico, x="Nível de Obesidade", y="Atividade Física (Dias/Semana)",
            color="Alimentos Calóricos", category_orders={"Nível de Obesidade": ordem_obesidade},
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#4ecdc4'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Insira o arquivo 'Obesity.csv' na pasta para visualizar os gráficos.")

# =========================================================================
# ABA 2: SIMULADOR DE TRIAGEM PREDITIVA (O seu código original de ML)
# =========================================================================
with aba_predicao:
    st.title("Sistema Inteligente de Diagnóstico de Obesidade")
    st.markdown("""
    Este sistema utiliza um modelo de **Machine Learning (Random Forest)** de alta precisão 
    para auxiliar a equipe médica na triagem e diagnóstico precoce de pacientes.
    """)
    st.write("---")

    st.subheader("Preencha os Dados do Paciente para Triagem:")

    # Organizando os campos do seu formulário original em colunas
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        gender = st.selectbox("Gênero:", ["Masculino", "Feminino"])
        age = st.number_input("Idade:", min_value=1, max_value=100, value=25)
        height = st.number_input("Altura (em metros):", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Peso (em kg):", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        family_history = st.selectbox("Histórico familiar de sobrepeso?", ["Sim", "Não"])
        favc = st.selectbox("Consome alimentos calóricos com frequência? (FAVC)", ["Sim", "Não"])
        fcvc = st.slider("Frequência de consumo de vegetais (FCVC) [1 a 3]:", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Número de refeições principais por dia (NCP) [1 a 4]:", 1.0, 4.0, 3.0, step=0.1)

    with col_f2:
        caec = st.selectbox("Consumo de alimentos entre as refeições (CAEC):", ["Às vezes", "Frequentemente", "Sempre", "Nunca"])
        smoke = st.selectbox("É fumante? (SMOKE)", ["Sim", "Não"])
        ch2o = st.slider("Consumo diário de água em litros (CH2O) [1 a 3]:", 1.0, 3.0, 2.0, step=0.1)
        scc = st.selectbox("Monitora o consumo de calorias diárias? (SCC)", ["Sim", "Não"])
        faf = st.slider("Frequência de atividade física por semana (FAF) [0 a 3]:", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Tempo de uso de dispositivos tecnológicos (TUE) [0 a 2]:", 0.0, 2.0, 1.0, step=0.1)
        calc = st.selectbox("Consumo de álcool (CALC):", ["Nunca", "Às vezes", "Frequentemente", "Sempre"])
        mtrans = st.selectbox("Meio de transporte principal (MTRANS):", ["Transporte Público", "Automóvel", "Andando", "Motocicleta", "Bicicleta"])

    st.write("---")

    if st.button("Executar Diagnóstico", type="primary"):
        # Monta o dicionário exatamente como o seu modelo original espera receber
        dados_paciente = {
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history': [family_history], 'FAVC': [favc], 'FCVC': [fcvc], 'NCP': [ncp],
            'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf],
            'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
        }
        
        df_paciente = pd.DataFrame(dados_paciente)
        
        # Predição usando seus artefatos .pkl salvos
        dados_processados = precomputador.transform(df_paciente)
        predicao_num = modelo.predict(dados_processados)[0]
        diagnostico_final = mapeador.inverse_transform([predicao_num])[0]
        diagnostico_formatado = diagnostico_final.replace('_', ' ')
        
        # Exibição do Resultado Clínico
        st.subheader("Resultado da Avaliação:")
        if "Obesity" in diagnostico_final:
            st.error(f"⚠️ Alerta Crítico: O paciente apresenta padrões compatíveis com **{diagnostico_formatado}**.")
        elif "Overweight" in diagnostico_final:
            st.warning(f"🟡 Atenção: O paciente foi classificado na faixa de **{diagnostico_formatado}**.")
        else:
            st.success(f"💚 Diagnóstico Saudável: O paciente apresenta estabilidade — **{diagnostico_formatado}**.")