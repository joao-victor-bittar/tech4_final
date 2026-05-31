import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── 1. CONFIGURAÇÃO DA PÁGINA ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Diagnóstico - Hospital",
    page_icon="🏥",
    layout="wide" # Mantido wide para o Dashboard ocupar a tela inteira de lado a lado
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
# ABA 1: DASHBOARD ANALÍTICO
# =========================================================================
with aba_dashboard:
    st.title("📊 Painel de Análise Epidemiológica e Fatores de Risco")
    st.markdown("""
    Este painel estratégico auxilia a **diretoria hospitalar e as equipes médicas** a mapear os principais 
    gatilhos comportamentais e genéticos da base de pacientes, permitindo desenhar políticas de **medicina preventiva**.
    """)
    st.write("---")
    
    if not df_clinico.empty:
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
        st.subheader("💥 Correlação entre Atividade Física e Consumo de Alimentos Calóricos")
        fig_box = px.box(
            df_clinico, x="Nível de Obesidade", y="Atividade Física (Dias/Semana)",
            color="Alimentos Calóricos", category_orders={"Nível de Obesidade": ordem_obesidade},
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#4ecdc4'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🌡️ Matriz de Correlação das Variáveis Clínicas e Comportamentais")
        num_cols = ['Idade', 'Altura', 'Peso', 'FCVC', 'NCP', 'Consumo de Água (Litros)', 'Atividade Física (Dias/Semana)', 'TUE']
        cols_presentes = [c for c in num_cols if c in df_clinico.columns]
        
        if cols_presentes:
            corr_matrix = df_clinico[cols_presentes].corr()
            fig_heatmap = px.imshow(
                corr_matrix, text_auto='.2f', aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu, zmin=-1, zmax=1
            )
            fig_heatmap.update_layout(margin=dict(l=20, r=20, t=20, b=20), coloraxis_colorbar=dict(title="Correlação"))
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.info("**Insight Clínico:** O mapa destaca como variáveis biométricas se correlacionam com os hábitos diários hospitalares.")
    else:
        st.warning("Insira o arquivo 'Obesity.csv' na pasta para visualizar os gráficos.")

# =========================================================================
# ABA 2: SIMULADOR DE TRIAGEM PREDITIVA (Código original do App2 restaurado)
# =========================================================================
with aba_predicao:
    st.title("Sistema de Diagnóstico de Obesidade")
    st.markdown("""
    Este sistema utiliza um modelo de **Machine Learning (Random Forest)** de alta precisão 
    para auxiliar a equipe médica na triagem e diagnóstico precoce de pacientes.
    """)
    st.write("---")

    st.subheader("Ficha Clínica do Paciente")

    col1, col2 = st.columns(2)

    with col1:
        gender_pt = st.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
        age = st.number_input("Idade (anos)", min_value=14, max_value=100, value=25)
        height = st.number_input("Altura (metros)", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)

    with col2:
        family_history_pt = st.selectbox("Histórico Familiar de Excesso de Peso?", ["Sim", "Não"])
        favc_pt = st.selectbox("Consumo Frequente de Alimentos Hipercalóricos?", ["Sim", "Não"])
        fcvc = st.slider("Frequência de Consumo de Vegetais (1=Raro, 3=Sempre)", 1, 3, 2)
        ncp = st.slider("Número de Refeições Principais por Dia", 1, 4, 3)

    st.markdown("#### Estilo de Vida e Hábitos")
    col3, col4 = st.columns(2)

    with col3:
        caec_pt = st.selectbox("Come entre as refeições (Beliscar)?", ["Não", "Às Vezes", "Frequentemente", "Sempre"])
        smoke_pt = st.selectbox("Hábito de Fumar?", ["Sim", "Não"])
        ch2o = st.slider("Consumo Diário de Água (1=<1L, 2=1-2L, 3=>2L)", 1, 3, 2)

    with col4:
        scc_pt = st.selectbox("Monitora a Ingestão Calórica Diária?", ["Sim", "Não"])
        faf = st.slider("Atividade Física Semanal (0=Nenhuma, 3=Frequente)", 0, 3, 1)
        tue = st.slider("Tempo Diário de Uso de Telas (0=0-2h, 2=>5h)", 0, 2, 1)

    calc_pt = st.selectbox("Consumo de Bebida Alcoólica?", ["Não", "Às Vezes", "Frequentemente", "Sempre"])
    mtrans_pt = st.selectbox("Meio de Transporte Habitual", ["Carro", "Moto", "Bicicleta", "Transporte Público", "Andando"])

    st.write("---")

    if st.button("Calcular Diagnóstico Clínico", type="primary"):
        
        # Mapeamentos reversos exatos para enviar os dados idênticos ao que o modelo do app2 espera
        mapa_genero = {"Masculino": "Male", "Feminino": "Female"}
        mapa_sim_nao = {"Sim": "yes", "Não": "no"}
        mapa_caec_calc = {"Não": "no", "Às Vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
        mapa_mtrans = {
            "Carro": "Automobile", "Moto": "Motorbike", "Bicicleta": "Bike",
            "Transporte Público": "Public_Transportation", "Andando": "Walking"
        }

        # Conversão das variáveis para o formato de treino original
        gender = mapa_genero[gender_pt]
        family_history = mapa_sim_nao[family_history_pt]
        favc = mapa_sim_nao[favc_pt]
        caec = mapa_caec_calc[caec_pt]
        smoke = mapa_sim_nao[smoke_pt]
        scc = mapa_sim_nao[scc_pt]
        calc = mapa_caec_calc[calc_pt]
        mtrans = mapa_mtrans[mtrans_pt]

        # Monta exatamente o dicionário com as colunas originais do app2
        dados_paciente = {
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history': [family_history], 'FAVC': [favc], 'FCVC': [fcvc], 'NCP': [ncp],
            'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf],
            'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
        }
        
        df_paciente = pd.DataFrame(dados_paciente)
        
        # Processamento e predição idênticos ao app2
        dados_processados = precomputador.transform(df_paciente)
        predicao_num = modelo.predict(dados_processados)[0]
        diagnostico_final = mapeador.inverse_transform([predicao_num])[0]
        diagnostico_formatado = diagnostico_final.replace('_', ' ')
        
        st.subheader("Resultado da Avaliação:")
        
        if "Obesity" in diagnostico_final:
            st.error(f"Alerta Clínico: O paciente apresenta critérios para **{diagnostico_formatado}**.")
        elif "Overweight" in diagnostico_final:
            st.warning(f"Atenção: O paciente encontra-se na faixa de **{diagnostico_formatado}**.")
        else:
            st.success(f"Diagnóstico Estável: O paciente apresenta classificação de **{diagnostico_formatado}**.")