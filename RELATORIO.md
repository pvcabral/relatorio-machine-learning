# Perfis de Estudantes e Desempenho Acadêmico: Agrupamento com Mapas Auto-Organizáveis (SOM)

## Introdução

### Identificação

| Nome
|--- 
| Paulo Victor Medeiros
| Elke Rebouças

* **Vídeo da apresentação:** _[link do vídeo]_
* **Código:** _[[link do repositório](https://github.com/pvcabral/relatorio-machine-learning)]_ (script `gerar_relatorio.py`)

### Informações Gerais

**Problema.** O desempenho de um estudante ao final do semestre depende de muitos fatores: características pessoais, contexto familiar, hábitos de estudo e o próprio curso. Com uma base pequena e cheia de variáveis categóricas, é difícil enxergar essas relações só com estatística descritiva. A pergunta central do trabalho é: **existem perfis de estudantes, definidos pelos hábitos de estudo, que se associam a desempenhos diferentes?**

**Algoritmos usados.**
* **SOM (Self-Organizing Map):** rede neural não supervisionada formada por neurônios que "competem" pelos dados. Cada aluno é atribuído ao neurônio mais parecido com ele. Neste trabalho, **cada neurônio representa um cluster** (perfil de aluno). É o algoritmo principal do trabalho.
* **K-Means:** algoritmo de agrupamento clássico, usado para verificar se os clusters do SOM são estáveis.
* **PCA:** redução para 2 dimensões, só para visualização.
* **Random Forest:** classificador supervisionado complementar, para medir quanto do desempenho pode ser previsto e quais variáveis mais pesam.

**Objetivos.**
1. Treinar um SOM para encontrar perfis (clusters) de estudantes a partir dos hábitos de estudo.
2. Caracterizar cada perfil quanto à nota final, ao curso e às variáveis pessoais e familiares.
3. Verificar se os clusters são robustos, comparando o SOM com o K-Means.
4. Gerar insights e relatórios sobre o desempenho dos estudantes.

**Base de dados.** *Higher Education Students Performance Evaluation*, obtida no [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/students-performance). Tem **145 estudantes** e **33 colunas**, sem valores ausentes:
* **ID** do estudante;
* **Questões 1–10:** perfil pessoal (idade, sexo, tipo de escola, bolsa, trabalho, salário, transporte, moradia etc.);
* **Questões 11–16:** contexto familiar (escolaridade e ocupação dos pais, número de irmãos, situação dos pais);
* **Questões 17–30:** hábitos de estudo e histórico acadêmico (horas de estudo, leitura, presença, anotações, GPA do semestre anterior, GPA esperado etc.);
* **COURSE ID:** curso (9 cursos, tratados de forma anônima como "Curso 1" a "Curso 9");
* **GRADE:** nota final, de 0 a 7 (0 = Fail, 1 = DD, 2 = DC, 3 = CC, 4 = CB, 5 = BB, 6 = BA, 7 = AA).

Todas as variáveis são categóricas codificadas em números inteiros. Algumas são ordinais (ex.: horas de estudo de 1 a 5) e outras são nominais (ex.: meio de transporte).

![Distribuição das notas e nota média por curso](figuras/01_notas_e_cursos.png)

A nota mais frequente é DD (35 alunos). Há uma diferença grande entre cursos: os cursos 3, 5, 6 e 7 têm média acima de BB, enquanto os cursos 1, 8 e 9 ficam entre DD e DC.

## Metodologia

### Visão geral do pipeline

```
CSV  →  renomeação e rotulação  →  análise exploratória
     →  seleção de atributos (10 hábitos)  →  ajuste de sentido das escalas  →  normalização Min-Max
     →  treino do SOM com k neurônios (k = 2 a 6)  →  escolha de k pela silhueta
     →  cada aluno pertence ao cluster do seu neurônio vencedor
     →  perfil dos clusters (nota, curso, variáveis pessoais)  +  comparação com K-Means
     →  Random Forest (complementar)
```

### Técnicas de ML aplicadas

**1. Self-Organizing Map (SOM).** O SOM é um conjunto de neurônios organizados em uma grade, e cada neurônio tem um vetor de pesos (protótipo) com a mesma dimensão dos dados. Em cada iteração:
1. um aluno *x* é sorteado;
2. encontra-se o neurônio vencedor (*Best Matching Unit*, BMU), aquele cujo protótipo está mais próximo de *x* pela distância euclidiana;


Neste trabalho usamos um **SOM em linha (grade 1 × *k*)**: com *k* neurônios, obtemos diretamente *k* clusters, pois cada aluno pertence ao cluster do seu neurônio vencedor. Ao final do treino, o protótipo de cada neurônio representa o **"aluno típico" do grupo**. A biblioteca usada foi a **MiniSom**.

**2. K-Means (comparação).** Aplicado aos mesmos dados, com o mesmo *k*. As duas partições são comparadas pelo **Adjusted Rand Index (ARI)**, em que 1 significa partições idênticas e 0 significa concordância ao acaso.

**3. Random Forest (complementar).** Conjunto de árvores de decisão que prevê se o aluno terá **nota alta (BB, BA ou AA)**. Ele usa todas as variáveis, inclusive o curso, e indica quais atributos são mais importantes.

### Seleção de atributos e motivação

Testamos primeiro o SOM com **todas as 56 colunas** (variáveis nominais em *one-hot*). A estrutura de clusters ficou muito fraca (silhueta de no máximo 0,08), porque as dezenas de colunas binárias de perfil pessoal e familiar adicionam ruído à distância euclidiana.

Por isso, o SOM foi treinado apenas com o **bloco de hábitos de estudo e histórico acadêmico**, com 10 atributos ordinais ou binários:

| Atributo | Descrição |
|---|---|
| `horas_estudo` | horas de estudo semanais |
| `leitura_nao_cient` | frequência de leitura não científica |
| `leitura_cient` | frequência de leitura científica |
| `seminarios` | participa de seminários/conferências do departamento |
| `presenca` | frequência às aulas |
| `anotacoes` | faz anotações em aula |
| `escuta` | presta atenção em aula |
| `discussao` | a discussão melhora o interesse e o sucesso |
| `gpa_ultimo` | GPA do semestre anterior |
| `gpa_esperado` | GPA esperado na formatura |

Motivações para essa escolha:
* **Interpretabilidade:** os clusters passam a representar comportamentos de estudo, que podem ser trabalhados pela instituição.
* **Separação entre entrada e desfecho:** a **nota**, o **curso** e as **variáveis pessoais e familiares** ficam de fora do treino e são usados depois para *descrever* os clusters. Assim, se um cluster tem notas melhores, isso é um resultado, e não algo que o modelo recebeu pronto.

### Etapas de treinamento e teste

1. **Pré-processamento:** renomeação das colunas (os nomes originais têm problemas de *encoding* e duas colunas com o mesmo nome) e criação dos rótulos legíveis.
2. **Ajuste de sentido das escalas:** em `seminarios` (1 = Sim) e `presenca` (1 = Sempre), o valor 1 representa o comportamento mais engajado. Essas escalas foram invertidas para que, em todos os atributos, **valor maior = mais engajamento**.
3. **Normalização Min-Max** para [0, 1], para que todos os atributos tenham o mesmo peso na distância.
4. **Treinamento do SOM:** grade 1 × *k*, inicialização pelos componentes principais (PCA), 5.000 iterações.
5. **Escolha de *k*:** treinamos SOMs com 2 a 6 neurônios e escolhemos o que deu maior **índice de silhueta**. A silhueta mede, de −1 a 1, o quanto cada aluno está mais próximo do seu próprio grupo do que do grupo vizinho.
6. **Avaliação dos clusters:** como o SOM é não supervisionado, não existe "teste" no sentido clássico. A avaliação foi feita por (a) silhueta; (b) estabilidade, comparando com o K-Means pelo ARI; e (c) validade externa, testando se a nota (não usada no treino) difere entre os clusters com o teste de Kruskal-Wallis.
7. **Random Forest:** avaliado por **validação cruzada estratificada repetida** (5 *folds* × 10 repetições) e comparado com um *baseline* que sempre prevê a classe majoritária.

## Códigos

O código completo está em `gerar_relatorio.py`. O script baixa a base diretamente do Kaggle (biblioteca `kagglehub`), roda toda a análise e salva os gráficos na pasta `figuras/`. Para executar: `pip install -r requirements.txt` e depois `python gerar_relatorio.py`.
**Link:** _[link do repositório]_

**Seleção, ajuste de sentido e normalização dos atributos**
```python
FEATURES = ["horas_estudo", "leitura_nao_cient", "leitura_cient", "seminarios", "presenca",
            "anotacoes", "escuta", "discussao", "gpa_ultimo", "gpa_esperado"]

X = df[FEATURES].copy()
X["seminarios"] = 3 - X["seminarios"]   # 1=Sim -> 2 ; 2=Não -> 1  (maior = mais engajado)
X["presenca"]   = 4 - X["presenca"]     # 1=Sempre -> 3
Xs = MinMaxScaler().fit_transform(X)
```

**Treinamento do SOM (cada neurônio = um cluster)**
```python
def novo_som(n_neuronios, Xs):
    som = MiniSom(1, n_neuronios, Xs.shape[1], sigma=1.0, learning_rate=0.5,
                  neighborhood_function="gaussian", random_seed=42)
    som.pca_weights_init(Xs)                         # inicialização pelos componentes principais
    som.train(Xs, 5000, random_order=True)
    rotulos = np.array([som.winner(x)[1] for x in Xs])   # neurônio vencedor = cluster do aluno
    return som, rotulos
```

**Escolha do número de clusters pela silhueta**
```python
silhuetas = {}
for n in range(2, 7):
    _, rot = novo_som(n, Xs)
    silhuetas[n] = silhouette_score(Xs, rot)
K = max(silhuetas, key=silhuetas.get)              # k com maior silhueta
som, rot = novo_som(K, Xs)
prototipos = som.get_weights()[0]                  # "aluno típico" de cada cluster
```

**Validação externa e estabilidade**
```python
kruskal(*[g["nota"] for _, g in df.groupby("cluster")])   # a nota difere entre clusters?

km = KMeans(n_clusters=3, n_init=50, random_state=42).fit(Xs)
adjusted_rand_score(df["cluster"], km.labels_)            # SOM vs K-Means
```

**Random Forest com validação cruzada**
```python
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced")
cross_val_score(rf, Xm, y, cv=cv, scoring="roc_auc")
```

## Experimentos

### Experimento 1: escolha dos atributos de entrada do SOM

Testamos três conjuntos de atributos. Para cada um, treinamos o SOM com 2 a 6 neurônios; a tabela mostra a melhor silhueta obtida.

| Conjunto de atributos | Dimensão | Melhor silhueta |
|---|---|---|
| Todas as variáveis (nominais em *one-hot*) | 56 | 0,075 |
| Hábitos ordinais + hábitos nominais em *one-hot* | 22 | 0,213 |
| **Hábitos ordinais/binários (escolhido)** | **10** | **0,249** |

Retirar as variáveis nominais (muitas colunas binárias esparsas) produziu os clusters mais coesos.

### Experimento 2: parâmetros do SOM e número de clusters

| Parâmetro | Valor |
|---|---|
| Grade | 1 × *k* (testado *k* = 2 a 6) |
| Raio de vizinhança (σ) | 1,0 |
| Taxa de aprendizado | 0,5 |
| Função de vizinhança | gaussiana |
| Inicialização | PCA (mais estável que a aleatória) |
| Iterações | 5.000 |

| Nº de neurônios (*k*) | 2 | **3** | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Silhueta | 0,234 | **0,249** | 0,162 | 0,200 | 0,183 |

![Escolha de k](figuras/03_escolha_k.png)

O SOM com **3 neurônios** teve a maior silhueta e foi o escolhido. Os clusters foram renumerados em ordem crescente de nota média (C1 = menor nota média).

### Experimento 3: protótipos dos neurônios

![Protótipos do SOM](figuras/04_prototipos_som.png)

Cada linha mostra o protótipo (vetor de pesos) de um neurônio, ou seja, o "aluno típico" de cada cluster, em escala de 0 a 1. As maiores diferenças entre os neurônios estão em **`seminarios`** e **`presenca`**:
* **C3:** alto em seminários **e** em presença, com os maiores valores de GPA anterior e esperado;
* **C2:** participa de seminários, mas tem presença baixa e menos horas de estudo;
* **C1:** presença alta, mas **não** participa de seminários; mais horas de estudo e muitas anotações.

### Experimento 4: perfil dos clusters

| Cluster | Alunos | Perfil de hábitos | Nota média (0–7) | % nota ≥ BB | % Fail/DD |
|---|---|---|---|---|---|
| **C3: Engajados institucionais** | 88 (61%) | sempre vão às aulas **e** participam de seminários; maior GPA anterior | **3,67** | **40%** | **22%** |
| **C2: Pouco engajados** | 33 (23%) | vão às aulas só "às vezes", estudam menos horas, anotam menos, menor GPA anterior e esperado | 2,64 | 24% | 42% |
| **C1: Esforçados sem seminários** | 24 (17%) | **não** participam de seminários, mas são os que mais estudam, sempre anotam e mais valorizam a discussão | 2,42 | 17% | 42% |

![Notas por cluster](figuras/05_notas_por_cluster.png)

**Validação externa:** o teste de Kruskal-Wallis mostra que a nota final **difere entre os clusters** (H = 9,96; p = 0,007), mesmo sem a nota ter sido usada no treino.

**Curso × cluster:**

| Curso | C1 | C2 | C3 | Nota média do curso |
|---|---|---|---|---|
| Curso 8 | 6 | 4 | 4 | 1,36 |
| Curso 9 | **13** | 1 | 7 | 2,19 |
| Curso 1 | 2 | **21** | 43 | 2,29 |
| Curso 7 | 1 | 2 | 12 | 6,33 |
| Demais (2, 3, 4, 5, 6) | 2 | 5 | 22 | 3,0 a 6,1 |

**Variáveis pessoais e familiares** (teste qui-quadrado entre cluster e variável): **sexo** (p < 0,001), **bolsa** (p = 0,02) e **idade** (p = 0,02) se associam aos clusters. O C1 tem 83% de mulheres e a maior proporção de bolsas de 75% ou integral (76%); o C3 tem 73% de homens.

### Experimento 5: estabilidade (SOM × K-Means)

| Método | Silhueta |
|---|---|
| SOM (1 × 3) | 0,249 |
| K-Means (k = 3) | 0,249 |

**ARI = 0,989:** as duas partições são praticamente idênticas, e só 1 aluno muda de grupo. Os perfis são estáveis e não dependem do algoritmo.

![Clusters no plano PCA](figuras/06_pca_clusters.png)

No plano PCA, a componente PC1 é dominada por `presenca` e pelos GPAs, e a PC2 por `seminarios`. Os três grupos aparecem bem separados.

### Experimento 6: Random Forest (complementar)

Alvo: **nota alta (BB, BA ou AA)**, que corresponde a 32,4% dos alunos.

| Modelo | Acurácia balanceada | ROC-AUC |
|---|---|---|
| *Baseline* (classe majoritária) | 0,500 | 0,500 |
| **Random Forest** | **0,744 ± 0,082** | **0,848 ± 0,074** |

![Importância das variáveis](figuras/07_importancia_rf.png)

As variáveis mais importantes são **curso** (Curso 1 e Curso 7) e o **GPA do semestre anterior**, seguidos de sexo, bolsa e salário.

### Discussão dos resultados

1. **O curso pesa mais que os hábitos.** A diferença de nota entre cursos (de 1,4 a 6,3) é muito maior que a diferença entre clusters (de 2,4 a 3,7), e o curso lidera a importância no Random Forest.
2. **O paradoxo do C1.** O grupo que mais declara esforço tem a menor nota média. Isso se explica pelo curso: 13 dos 24 alunos do C1 estão no Curso 9, um dos de nota mais baixa. A ausência de seminários pode refletir uma característica desse curso, e não uma escolha individual.
3. **Sexo é provavelmente um confundidor.** Os homens têm nota média maior (3,8 contra 2,3), mas os cursos 8 e 9, os de nota mais baixa, têm de 79% a 100% de mulheres. O efeito aparente de sexo é, em boa parte, efeito do curso.
4. **Presença e GPA anterior importam.** O grupo C2, que vai às aulas só "às vezes", tem 42% de Fail/DD, quase o dobro do C3. O GPA anterior é a variável de hábito mais correlacionada com a nota (ρ de Spearman ≈ 0,35).

![Correlação com a nota](figuras/02_correlacao_nota.png)

**Limitações:** a amostra é pequena (145 alunos), e alguns cursos têm pouquíssimos alunos (Curso 2 com 2, Curso 4 com 4). Os dados são autodeclarados e discretos. A silhueta de 0,25 indica estrutura **moderada**: os grupos são tendências, não segmentos perfeitamente separados, o que é esperado em dados de questionário.

## Conclusão

**Sim, o trabalho atendeu aos objetivos.**

1. **SOM treinado e perfis encontrados:** o SOM com 3 neurônios (escolhido pela silhueta) revelou **três perfis de estudo**: engajados institucionais, pouco engajados e esforçados sem seminários.
2. **Perfis caracterizados:** os clusters diferem significativamente na nota final (p = 0,007), mesmo sem a nota ter sido usada no treino. O perfil engajado tem mais que o dobro da proporção de notas altas do C1 e cerca de metade da proporção de Fail/DD dos outros dois grupos.
3. **Robustez verificada:** o K-Means reproduziu praticamente a mesma partição (ARI = 0,99).
4. **Insights gerados:** o curso é o fator dominante no desempenho, o GPA anterior e a presença nas aulas são os hábitos mais relevantes, e efeitos aparentes como o de sexo precisam ser lidos à luz do curso.

Como trabalhos futuros, sugerimos coletar mais dados por curso e usar modelos que controlem o efeito do curso (ex.: regressão ordinal com o curso como covariável), para separar o efeito dos hábitos do efeito do curso.

---

**Base de dados:** https://www.kaggle.com/datasets/joebeachcapital/students-performance
