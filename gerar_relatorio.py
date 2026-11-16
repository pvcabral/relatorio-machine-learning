"""
Perfis de Estudantes e Desempenho Acadêmico — Agrupamento com SOM (Self-Organizing Map)
========================================================================================

Gera todas as análises e gráficos usados no relatório (RELATORIO.md)

Base de dados:
    Higher Education Students Performance Evaluation
    Kaggle: https://www.kaggle.com/datasets/joebeachcapital/students-performance

Como executar:
    pip install -r requirements.txt
    python gerar_relatorio.py

Saídas:
    - Relatórios impressos no terminal
    - Gráficos em ./figuras/*.png
    - alunos_com_clusters.csv (base original + cluster de cada aluno)
"""

import os
import sys
import warnings

# Garante acentos e símbolos (ρ, …) no terminal do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # salva figuras em arquivo, sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minisom import MiniSom
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# Configurações
PASTA = os.path.dirname(os.path.abspath(__file__))
KAGGLE_DATASET = "joebeachcapital/students-performance"   # base pública no Kaggle
ARQUIVO_CSV = "StudentsPerformance_with_headers.csv"
PASTA_FIG = os.path.join(PASTA, "figuras")
ARQUIVO_SAIDA = os.path.join(PASTA, "alunos_com_clusters.csv")

SEED = 42
SIGMA = 1.0                   # raio de vizinhança do SOM
ITERACOES = 5000              # iterações de treino do SOM
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # cor fixa por cluster

np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
os.makedirs(PASTA_FIG, exist_ok=True)

NOMES_COLUNAS = [
    "id", "idade", "sexo", "escola_tipo", "bolsa", "trabalha", "arte_esporte", "parceiro",
    "salario", "transporte", "moradia", "edu_mae", "edu_pai", "irmaos", "status_pais",
    "ocup_mae", "ocup_pai", "horas_estudo", "leitura_nao_cient", "leitura_cient",
    "seminarios", "impacto_projetos", "presenca", "prep_prova_1", "prep_prova_2",
    "anotacoes", "escuta", "discussao", "flip", "gpa_ultimo", "gpa_esperado", "curso", "nota",
]

ROTULOS = {
    "idade": {1: "18-21", 2: "22-25", 3: "26+"},
    "sexo": {1: "Feminino", 2: "Masculino"},
    "escola_tipo": {1: "Privada", 2: "Pública", 3: "Outra"},
    "bolsa": {1: "Nenhuma", 2: "25%", 3: "50%", 4: "75%", 5: "Integral"},
    "trabalha": {1: "Sim", 2: "Não"},
    "arte_esporte": {1: "Sim", 2: "Não"},
    "parceiro": {1: "Sim", 2: "Não"},
    "salario": {1: "135-200", 2: "201-270", 3: "271-340", 4: "341-410", 5: ">410"},
    "transporte": {1: "Ônibus", 2: "Carro/táxi", 3: "Bicicleta", 4: "Outro"},
    "moradia": {1: "Aluguel", 2: "Dormitório", 3: "Com família", 4: "Outro"},
    "edu_mae": {1: "Fundamental", 2: "Fund. II", 3: "Médio", 4: "Superior", 5: "Mestrado", 6: "Doutorado"},
    "edu_pai": {1: "Fundamental", 2: "Fund. II", 3: "Médio", 4: "Superior", 5: "Mestrado", 6: "Doutorado"},
    "status_pais": {1: "Casados", 2: "Divorciados", 3: "Falecido(s)"},
    "horas_estudo": {1: "Nenhuma", 2: "<5h", 3: "6-10h", 4: "11-20h", 5: ">20h"},
    "seminarios": {1: "Sim", 2: "Não"},
    "presenca": {1: "Sempre", 2: "Às vezes", 3: "Nunca"},
    "gpa_ultimo": {1: "<2.00", 2: "2.00-2.49", 3: "2.50-2.99", 4: "3.00-3.49", 5: ">3.49"},
    "gpa_esperado": {1: "<2.00", 2: "2.00-2.49", 3: "2.50-2.99", 4: "3.00-3.49", 5: ">3.49"},
}
NOTA_ROT = {0: "Fail", 1: "DD", 2: "DC", 3: "CC", 4: "CB", 5: "BB", 6: "BA", 7: "AA"}

# Variáveis usadas no treino do SOM (bloco de hábitos de estudo / acadêmico)
FEATURES = ["horas_estudo", "leitura_nao_cient", "leitura_cient", "seminarios", "presenca",
            "anotacoes", "escuta", "discussao", "gpa_ultimo", "gpa_esperado"]


def titulo(texto):
    print("\n" + "=" * 80 + f"\n{texto}\n" + "=" * 80)


def salvar(nome):
    caminho = os.path.join(PASTA_FIG, nome)
    plt.tight_layout()
    plt.savefig(caminho, bbox_inches="tight")
    plt.close("all")
    print(f"[figura salva] {caminho}")


# 1. Carga e preparação
def baixar_base():
    """Baixa a base do Kaggle (ou usa o cache local do kagglehub) e retorna o caminho do CSV."""
    import kagglehub
    pasta = kagglehub.dataset_download(KAGGLE_DATASET)
    caminho = os.path.join(pasta, ARQUIVO_CSV)
    print(f"Base: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print(f"Arquivo: {caminho}")
    return caminho


def carregar_dados():
    titulo("1. CARGA E PREPARAÇÃO DOS DADOS")
    df = pd.read_csv(baixar_base())
    # Renomeia pela posição: os nomes originais têm encoding quebrado e duas colunas iguais
    df.columns = NOMES_COLUNAS
    df["curso_nome"] = "Curso " + df["curso"].astype(str)   # curso anonimizado
    df["nota_alta"] = (df["nota"] >= 5).astype(int)          # BB, BA ou AA
    df["nota_baixa"] = (df["nota"] <= 1).astype(int)         # Fail ou DD
    print(f"Formato: {df.shape} | valores ausentes: {df.isna().sum().sum()}")
    return df


# 2. Análise exploratória
def analise_exploratoria(df):
    titulo("2. ANÁLISE EXPLORATÓRIA")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    cont = df["nota"].value_counts().sort_index()
    ax[0].bar([NOTA_ROT[i] for i in cont.index], cont.values, color=CORES[0], width=0.7)
    for i, v in enumerate(cont.values):
        ax[0].text(i, v + 0.5, v, ha="center", fontsize=9)
    ax[0].set_title("Distribuição da nota final")
    ax[0].set_ylabel("Nº de estudantes")

    resumo_curso = (df.groupby("curso_nome")
                      .agg(n=("nota", "size"), nota_media=("nota", "mean"))
                      .sort_values("nota_media"))
    ax[1].barh(resumo_curso.index, resumo_curso["nota_media"], color=CORES[0], height=0.6)
    for i, (n, m) in enumerate(zip(resumo_curso["n"], resumo_curso["nota_media"])):
        ax[1].text(m + 0.05, i, f"{m:.1f}  (n={n})", va="center", fontsize=9)
    ax[1].set_title("Nota média por curso (0=Fail … 7=AA)")
    ax[1].set_xlim(0, 7.5)
    salvar("01_notas_e_cursos.png")

    print("\nNota média por curso:")
    print(resumo_curso.round(2))

    # Correlação de Spearman das variáveis ordinais com a nota
    cols_ord = ["idade", "bolsa", "salario", "edu_mae", "edu_pai", "irmaos", "horas_estudo",
                "leitura_nao_cient", "leitura_cient", "presenca", "anotacoes", "escuta",
                "discussao", "gpa_ultimo", "gpa_esperado"]
    corr = df[cols_ord + ["nota"]].corr(method="spearman")["nota"].drop("nota").sort_values()
    plt.figure(figsize=(7, 5))
    plt.barh(corr.index, corr.values, height=0.6,
             color=[CORES[1] if v < 0 else CORES[0] for v in corr.values])
    plt.axvline(0, color="gray", lw=1)
    plt.title("Correlação de Spearman com a nota final")
    plt.xlabel("ρ")
    salvar("02_correlacao_nota.png")
    print("\nCorrelação de Spearman com a nota:")
    print(corr.round(2).to_string())

    # Variáveis nominais: diferença máxima de nota média entre categorias
    nominais = ["sexo", "escola_tipo", "trabalha", "arte_esporte", "parceiro", "transporte", "moradia",
                "status_pais", "seminarios", "impacto_projetos", "prep_prova_1", "prep_prova_2", "flip"]
    amplitude = {c: df.groupby(c)["nota"].mean().max() - df.groupby(c)["nota"].mean().min()
                 for c in nominais}
    print("\nDiferença máxima de nota média entre categorias (variáveis nominais):")
    print(pd.Series(amplitude).sort_values(ascending=False).round(2).to_string())
    print("(atenção: categorias com poucos alunos, ex.: moradia 'Outro' com n=1, inflam esse valor)")


# Matriz com todas as variáveis (usada no Random Forest)
NOMINAIS_MODELO = ["curso", "escola_tipo", "transporte", "moradia", "status_pais", "ocup_mae", "ocup_pai",
                   "impacto_projetos", "prep_prova_1", "prep_prova_2", "flip"]


def matriz_modelo(df):
    """Todas as 31 variáveis do questionário + curso: ordinais como números, nominais em one-hot."""
    cols = [c for c in NOMES_COLUNAS if c not in ("id", "nota")]
    return pd.get_dummies(df[cols], columns=NOMINAIS_MODELO, dtype=int)


# 3. Treinamento do SOM
def preparar_features(df):
    X = df[FEATURES].copy()
    # Inverte escalas onde 1 = "melhor", para que valor maior = mais engajamento
    X["seminarios"] = 3 - X["seminarios"]   # 1=Sim -> 2 ; 2=Não -> 1
    X["presenca"] = 4 - X["presenca"]       # 1=Sempre -> 3
    return MinMaxScaler().fit_transform(X)


def novo_som(n_neuronios, Xs):
    """SOM em uma linha (1 x n). Cada neurônio representa um cluster."""
    som = MiniSom(1, n_neuronios, Xs.shape[1], sigma=SIGMA, learning_rate=0.5,
                  neighborhood_function="gaussian", random_seed=SEED)
    som.pca_weights_init(Xs)
    som.train(Xs, ITERACOES, random_order=True)
    rotulos = np.array([som.winner(x)[1] for x in Xs])   # neurônio vencedor de cada aluno
    return som, rotulos


def treinar_som(df, Xs):
    titulo("3. TREINAMENTO DO SOM (cada neurônio = um cluster)")

    # Testa SOMs com 2 a 6 neurônios e escolhe pelo índice de silhueta
    silhuetas = {}
    for n in range(2, 7):
        _, rot = novo_som(n, Xs)
        silhuetas[n] = silhouette_score(Xs, rot)
    print("Silhueta por número de neurônios (maior = grupos mais bem separados):")
    print(pd.Series(silhuetas).round(3).to_string())

    plt.figure(figsize=(6, 3.5))
    plt.plot(list(silhuetas), list(silhuetas.values()), "o-", color=CORES[0], lw=2, ms=8)
    plt.title("Escolha do número de neurônios (clusters)")
    plt.xlabel("nº de neurônios do SOM"); plt.ylabel("silhueta")
    plt.xticks(list(silhuetas))
    salvar("03_escolha_k.png")

    K = max(silhuetas, key=silhuetas.get)
    som, rot = novo_som(K, Xs)
    print(f"\nEscolhido: SOM 1x{K} | erro de quantização: {som.quantization_error(Xs):.3f}")

    # Renumera os clusters em ordem crescente de nota média (C1 = pior desempenho)
    ordem = pd.Series(df["nota"].values).groupby(rot).mean().sort_values().index
    mapa = {old: new for new, old in enumerate(ordem)}
    df["cluster"] = [mapa[r] for r in rot]
    nomes_cl = [f"C{i + 1}" for i in range(K)]
    df["cluster_nome"] = df["cluster"].map(dict(enumerate(nomes_cl)))

    # Protótipos (pesos) de cada neurônio = "aluno típico" de cada cluster
    pesos = som.get_weights()[0]                          # (K, n_features)
    prototipos = pd.DataFrame(pesos[list(ordem)], index=nomes_cl, columns=FEATURES)
    plt.figure(figsize=(11, 0.7 * K + 1.5))
    sns.heatmap(prototipos, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                cbar_kws={"label": "valor normalizado (0-1)"})
    plt.title("Protótipo de cada neurônio do SOM (maior = mais engajamento)")
    plt.xticks(rotation=35, ha="right")
    salvar("04_prototipos_som.png")

    print("\nAlunos por cluster:")
    print(df["cluster_nome"].value_counts().sort_index().to_string())
    return K, nomes_cl


# 4. Perfil dos clusters
def perfil_clusters(df, K, nomes_cl):
    titulo("4. PERFIL DOS CLUSTERS")
    perfil = df.groupby("cluster_nome").agg(
        n=("nota", "size"),
        **{c: (c, "mean") for c in FEATURES},
        nota_media=("nota", "mean"),
        pct_nota_alta=("nota_alta", "mean"),
        pct_nota_baixa=("nota_baixa", "mean"),
    )
    perfil[["pct_nota_alta", "pct_nota_baixa"]] *= 100
    print("Médias dos hábitos (escala ORIGINAL do questionário) e desfecho:")
    print(perfil.round(2))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    sns.boxplot(data=df, x="cluster_nome", y="nota", order=nomes_cl, palette=CORES[:K], ax=ax[0], width=0.5)
    ax[0].set_yticks(range(8)); ax[0].set_yticklabels([NOTA_ROT[i] for i in range(8)])
    ax[0].set_title("Nota final por cluster"); ax[0].set_xlabel("")
    dist = pd.crosstab(df["cluster_nome"], df["nota"].map(NOTA_ROT), normalize="index")
    dist = dist.reindex(columns=list(NOTA_ROT.values()), fill_value=0) * 100
    dist.plot(kind="barh", stacked=True, ax=ax[1], colormap="Blues", edgecolor="white", linewidth=1.5, width=0.6)
    ax[1].set_title("Distribuição das notas por cluster (%)"); ax[1].set_xlabel("%"); ax[1].set_ylabel("")
    ax[1].legend(title="nota", bbox_to_anchor=(1.01, 1), loc="upper left"); ax[1].invert_yaxis()
    salvar("05_notas_por_cluster.png")

    stat, p = kruskal(*[g["nota"].values for _, g in df.groupby("cluster_nome")])
    print(f"\nKruskal-Wallis (notas diferem entre clusters?): H={stat:.2f}, p-valor={p:.4f}")

    tab_curso = pd.crosstab(df["curso_nome"], df["cluster_nome"])
    tab_curso["n"] = tab_curso.sum(axis=1)
    tab_curso["nota_media_curso"] = df.groupby("curso_nome")["nota"].mean().round(2)
    print("\nComposição de cada curso (anonimizado) por cluster:")
    print(tab_curso.sort_values("nota_media_curso"))

    print("\nAssociação cluster x variáveis pessoais/familiares (qui-quadrado; com n=145, indicativo):")
    linhas = []
    for c in ["idade", "sexo", "escola_tipo", "bolsa", "trabalha", "arte_esporte", "parceiro", "salario",
              "transporte", "moradia", "edu_mae", "edu_pai", "status_pais"]:
        _, p, *_ = chi2_contingency(pd.crosstab(df["cluster"], df[c]))
        linhas.append({"variável": c, "p_valor_qui2": round(p, 4)})
    print(pd.DataFrame(linhas).sort_values("p_valor_qui2").to_string(index=False))

    for c in ["sexo", "trabalha", "bolsa", "edu_mae", "edu_pai"]:
        t = pd.crosstab(df["cluster_nome"], df[c].map(ROTULOS[c]), normalize="index") * 100
        print(f"\n--- {c} (% dentro de cada cluster) ---")
        print(t.round(0))

    print("\nProporção de homens por curso (ajuda a ver o confundimento sexo x curso):")
    print(df.groupby("curso_nome")["sexo"].apply(lambda s: (s == 2).mean()).round(2).to_string())


# 5. Comparação com K-Means
def comparar_kmeans(df, Xs, K, nomes_cl):
    titulo("5. COMPARAÇÃO COM K-MEANS + VISUALIZAÇÃO (PCA)")
    km = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Xs)
    df["kmeans"] = km.labels_
    print(f"Silhueta K-Means: {silhouette_score(Xs, km.labels_):.3f} | "
          f"Silhueta SOM: {silhouette_score(Xs, df['cluster']):.3f}")
    print(f"ARI (SOM vs K-Means): {adjusted_rand_score(df['cluster'], df['kmeans']):.3f}  (1 = partições idênticas)")
    print(pd.crosstab(df["cluster_nome"], df["kmeans"], rownames=["SOM"], colnames=["K-Means"]))

    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(Xs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in range(K):
        m = (df["cluster"] == c).values
        ax[0].scatter(P[m, 0], P[m, 1], s=45, color=CORES[c], label=nomes_cl[c], edgecolor="white", linewidth=1)
    ax[0].set_title("Clusters do SOM no plano PCA"); ax[0].legend()
    sc = ax[1].scatter(P[:, 0], P[:, 1], c=df["nota"], cmap="Blues", s=45, edgecolor="gray",
                       linewidth=0.5, vmin=0, vmax=7)
    plt.colorbar(sc, ax=ax[1], label="nota (0-7)"); ax[1].set_title("Nota final no plano PCA")
    for a in ax:
        a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    salvar("06_pca_clusters.png")

    print("\nCargas da PCA (o que cada componente representa):")
    print(pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"]).round(2))


# 6. Modelo supervisionado: Random Forest
def random_forest(df):
    titulo("6. MODELO SUPERVISIONADO: RANDOM FOREST (nota alta = BB, BA ou AA)")
    Xm = matriz_modelo(df)
    y = df["nota_alta"]
    print(f"Proporção de nota alta: {y.mean():.1%}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                random_state=SEED)
    for nome, modelo in [("Baseline (classe majoritária)", DummyClassifier(strategy="most_frequent")),
                         ("Random Forest", rf)]:
        bacc = cross_val_score(modelo, Xm, y, cv=cv, scoring="balanced_accuracy")
        auc = cross_val_score(modelo, Xm, y, cv=cv, scoring="roc_auc")
        print(f"{nome:32s} | acurácia balanceada {bacc.mean():.3f} ± {bacc.std():.3f} | "
              f"ROC-AUC {auc.mean():.3f} ± {auc.std():.3f}")

    rf.fit(Xm, y)
    imp = pd.Series(rf.feature_importances_, index=Xm.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 5))
    plt.barh(imp.index[::-1], imp.values[::-1], color=CORES[0], height=0.6)
    plt.title("Top 15 variáveis — Random Forest (importância MDI)")
    plt.xlabel("redução média de impureza")
    salvar("07_importancia_rf.png")
    print("\nTop 15 variáveis mais importantes:")
    print(imp.round(3).to_string())


# 7. Resumo final + exportação
def resumo_final(df):
    titulo("7. TABELA-RESUMO DOS CLUSTERS")
    resumo = df.groupby("cluster_nome").agg(
        alunos=("nota", "size"),
        nota_media=("nota", "mean"),
        pct_nota_alta=("nota_alta", lambda s: 100 * s.mean()),
        pct_fail_ou_DD=("nota_baixa", lambda s: 100 * s.mean()),
        horas_estudo=("horas_estudo", "mean"),
        gpa_ultimo=("gpa_ultimo", "mean"),
        gpa_esperado=("gpa_esperado", "mean"),
        pct_masculino=("sexo", lambda s: 100 * (s == 2).mean()),
        curso_mais_comum=("curso_nome", lambda s: s.mode()[0]),
    ).round(1)
    print(resumo)
    df.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"\n[CSV salvo] {ARQUIVO_SAIDA}")


def main():
    df = carregar_dados()
    analise_exploratoria(df)
    Xs = preparar_features(df)
    K, nomes_cl = treinar_som(df, Xs)
    perfil_clusters(df, K, nomes_cl)
    comparar_kmeans(df, Xs, K, nomes_cl)
    random_forest(df)
    resumo_final(df)
    print("\nConcluído. Gráficos em:", PASTA_FIG)


if __name__ == "__main__":
    main()
