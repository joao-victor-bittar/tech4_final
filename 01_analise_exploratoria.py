# =============================================================
# ETAPA 1 — ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# Tech Challenge Fase 4 — POSTECH Data Analytics
# =============================================================

# ── O que é EDA? ──────────────────────────────────────────────
# Antes de treinar qualquer modelo, precisamos ENTENDER os dados.
# EDA (Exploratory Data Analysis) nos ajuda a responder:
#   - Quantas linhas e colunas temos?
#   - Existem valores faltando?
#   - Como as variáveis se distribuem?
#   - Existe algum padrão entre as variáveis e a obesidade?
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. CARREGANDO OS DADOS ────────────────────────────────────
# pd.read_csv lê o arquivo .csv e transforma em um DataFrame,
# que é basicamente uma tabela (como o Excel) dentro do Python.
df = pd.read_csv('Obesity.csv')


# ── 2. VISÃO GERAL DO DATASET ─────────────────────────────────
print("=" * 55)
print("2. VISÃO GERAL DO DATASET")
print("=" * 55)

# .shape retorna (número de linhas, número de colunas)
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")

# .dtypes mostra o tipo de cada coluna:
#   - float64 → número decimal  (ex: altura, peso)
#   - str/object → texto        (ex: gênero, sim/não)
print("\nTipos de cada coluna:")
print(df.dtypes)


# ── 3. VERIFICANDO VALORES NULOS ──────────────────────────────
print("\n" + "=" * 55)
print("3. VALORES NULOS POR COLUNA")
print("=" * 55)

# .isnull() marca como True cada célula vazia.
# .sum() conta quantos True existem por coluna.
# Se tudo for zero: ótimo! Não precisamos tratar nulos.
nulos = df.isnull().sum()
print(nulos)
print(f"\nTotal de nulos no dataset: {nulos.sum()}")


# ── 4. PRIMEIRAS LINHAS ───────────────────────────────────────
print("\n" + "=" * 55)
print("4. PRIMEIRAS 5 LINHAS")
print("=" * 55)

# .head(5) mostra as 5 primeiras linhas — útil para ver
# como os dados estão formatados de verdade.
print(df.head(5).to_string())


# ── 5. ESTATÍSTICAS DAS COLUNAS NUMÉRICAS ────────────────────
print("\n" + "=" * 55)
print("5. ESTATÍSTICAS DESCRITIVAS (COLUNAS NUMÉRICAS)")
print("=" * 55)

# .describe() calcula automaticamente para cada coluna numérica:
#   count → quantos valores existem
#   mean  → média
#   std   → desvio padrão (o quanto os valores variam)
#   min   → menor valor
#   25%   → 1º quartil (25% dos dados estão abaixo desse valor)
#   50%   → mediana (valor do meio)
#   75%   → 3º quartil
#   max   → maior valor
print(df.describe().round(2).to_string())


# ── 6. DISTRIBUIÇÃO DA VARIÁVEL ALVO ─────────────────────────
print("\n" + "=" * 55)
print("6. DISTRIBUIÇÃO DOS NÍVEIS DE OBESIDADE (COLUNA ALVO)")
print("=" * 55)

# A coluna 'Obesity' é o que queremos prever (variável alvo).
# .value_counts() conta quantas vezes cada classe aparece.
# normalize=True transforma em percentual.
contagem = df['Obesity'].value_counts()
percentual = df['Obesity'].value_counts(normalize=True).mul(100).round(1)

resumo_alvo = pd.DataFrame({'Quantidade': contagem, 'Percentual (%)': percentual})
print(resumo_alvo)


# ── 7. DISTRIBUIÇÃO DAS COLUNAS CATEGÓRICAS ──────────────────
print("\n" + "=" * 55)
print("7. DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS")
print("=" * 55)

# Colunas categóricas = colunas de texto (sim/não, masculino/feminino...)
# Vamos ver quantas vezes cada valor aparece em cada coluna.
cat_cols = ['Gender', 'family_history', 'FAVC', 'CAEC',
            'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())


# ══════════════════════════════════════════════════════════════
# GERANDO GRÁFICOS
# ══════════════════════════════════════════════════════════════

# Estilo visual dos gráficos
sns.set_theme(style="whitegrid", palette="muted")

# Ordem das classes de obesidade (do mais leve ao mais grave)
ordem_obesidade = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I', 'Overweight_Level_II',
    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]


# ── GRÁFICO 1: Distribuição da variável alvo ─────────────────
# Um gráfico de barras mostrando quantas pessoas há em cada
# nível de obesidade. Importante para ver se as classes estão
# balanceadas (quantidade parecida entre si).
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, x='Obesity', order=ordem_obesidade,
              hue='Obesity', palette='RdYlGn_r', legend=False, ax=ax)
ax.set_title('Distribuição dos Níveis de Obesidade', fontsize=14, fontweight='bold')
ax.set_xlabel('Nível de Obesidade')
ax.set_ylabel('Quantidade de Pessoas')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('grafico1_distribuicao_obesidade.png', dpi=150)
plt.close()
print("\n[✓] Gráfico 1 salvo: distribuição da variável alvo")


# ── GRÁFICO 2: Peso e Altura por nível de obesidade ──────────
# Boxplot: mostra a distribuição de um valor numérico por grupo.
# A "caixa" representa 50% dos dados; a linha do meio é a mediana.
# Pontos fora dos "bigodes" são outliers (valores atípicos).
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, x='Obesity', y='Weight', order=ordem_obesidade,
            hue='Obesity', palette='RdYlGn_r', legend=False, ax=axes[0])
axes[0].set_title('Peso por Nível de Obesidade', fontweight='bold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x='Obesity', y='Height', order=ordem_obesidade,
            hue='Obesity', palette='RdYlGn_r', legend=False, ax=axes[1])
axes[1].set_title('Altura por Nível de Obesidade', fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Peso e Altura por Nível de Obesidade', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico2_peso_altura.png', dpi=150)
plt.close()
print("[✓] Gráfico 2 salvo: peso e altura por nível de obesidade")


# ── GRÁFICO 3: Histórico familiar x Obesidade ────────────────
# Mostra se ter família com sobrepeso influencia o nível de obesidade.
# Usamos stacked bar (barras empilhadas) para comparar proporções.
tabela = pd.crosstab(df['family_history'], df['Obesity'])[ordem_obesidade]
tabela_pct = tabela.div(tabela.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
tabela_pct.plot(kind='bar', stacked=True, colormap='RdYlGn_r', ax=ax)
ax.set_title('Histórico Familiar x Nível de Obesidade (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Histórico familiar de sobrepeso')
ax.set_ylabel('Proporção (%)')
ax.tick_params(axis='x', rotation=0)
ax.legend(title='Nível de Obesidade', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('grafico3_historico_familiar.png', dpi=150)
plt.close()
print("[✓] Gráfico 3 salvo: histórico familiar x obesidade")


# ── GRÁFICO 4: Atividade física x Obesidade ──────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Obesity', y='FAF', order=ordem_obesidade,
            hue='Obesity', palette='RdYlGn_r', legend=False, ax=ax)
ax.set_title('Frequência de Atividade Física por Nível de Obesidade',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Nível de Obesidade')
ax.set_ylabel('Frequência de Atividade Física (0–3)')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('grafico4_atividade_fisica.png', dpi=150)
plt.close()
print("[✓] Gráfico 4 salvo: atividade física x obesidade")


# ── GRÁFICO 5: Correlação entre variáveis numéricas ──────────
# Um heatmap de correlação mostra o quanto duas variáveis
# numéricas "andam juntas".
#   +1 → correlação positiva perfeita (uma sobe, a outra sobe)
#   -1 → correlação negativa perfeita (uma sobe, a outra desce)
#    0 → sem relação
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
correlacao = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax)
ax.set_title('Correlação entre Variáveis Numéricas', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico5_correlacao.png', dpi=150)
plt.close()
print("[✓] Gráfico 5 salvo: heatmap de correlação")

print("\n✅ Análise exploratória concluída!")
