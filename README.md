# Perfis de Estudantes e Desempenho Acadêmico com SOM

Trabalho da disciplina de **Machine Learning**. Usamos um **Mapa Auto-Organizável (SOM)** para encontrar perfis de estudantes a partir dos hábitos de estudo e analisamos como esses perfis se relacionam com a nota final, o curso e as características pessoais e familiares.

📄 **Relatório completo:** [RELATORIO.md](RELATORIO.md)

## Base de dados

**Higher Education Students Performance Evaluation**: 145 estudantes e 31 atributos (perfil pessoal, contexto familiar e hábitos de estudo), mais o curso e a nota final (de Fail a AA).

* **Fonte:** Kaggle: https://www.kaggle.com/datasets/joebeachcapital/students-performance

O script baixa a base automaticamente do Kaggle (via [`kagglehub`](https://github.com/Kaggle/kagglehub)), então ela não está no repositório.

## Como executar

Requer **Python 3.10+**.

```bash
# 1. (opcional) criar um ambiente virtual
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate

# 2. instalar as dependências
pip install -r requirements.txt

# 3. rodar a análise completa
python gerar_relatorio.py
```

O script:
1. baixa a base do Kaggle;
2. faz a análise exploratória;
3. treina o SOM (cada neurônio = um cluster; nº de clusters escolhido pela silhueta);
4. gera o perfil dos clusters (nota, curso, variáveis pessoais e familiares);
5. compara com o K-Means e visualiza com PCA;
6. treina um Random Forest complementar para prever "nota alta".

**Saídas** (não versionadas, recriadas a cada execução):
* relatórios no terminal;
* gráficos em `figuras/`;
* `alunos_com_clusters.csv`, a base original com o cluster de cada aluno.

## Estrutura do repositório

```
├── gerar_relatorio.py   # código completo da análise
├── RELATORIO.md         # relatório do trabalho
├── requirements.txt     # dependências
└── README.md
```

## Principais resultados

| Cluster | Alunos | Nota média (0–7) | % nota ≥ BB | % Fail/DD |
|---|---|---|---|---|
| C3: Engajados institucionais | 88 | 3,67 | 40% | 22% |
| C2: Pouco engajados | 33 | 2,64 | 24% | 42% |
| C1: Esforçados sem seminários | 24 | 2,42 | 17% | 42% |

* As notas diferem entre os perfis (Kruskal-Wallis, p ≈ 0,007), mesmo sem a nota ter sido usada no treino.
* O K-Means encontra praticamente os mesmos grupos (ARI ≈ 0,99).
* O **curso** é o fator que mais pesa no desempenho, seguido do **GPA do semestre anterior**.
