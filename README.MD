<div align="center">
  <img src="docs/assets/pokemon-banner.gif" alt="Pokémon Banner" width="100%" height="280px" style="object-fit: cover; border-radius: 10px;"/>
</div>

<br/>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

</div>

---

## 📚 Projeto Acadêmico

<div align="center">

**Universidade Federal do Agreste de Pernambuco (UFAPE)** **Bacharelado em Ciência da Computação** Disciplina: **Inteligência Artificial - 2026.1** Professor: **Prof. Dr. Luis Filipe** </div>

### 👥 Equipe

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/alinesors">
        <img src="https://github.com/alinesors.png" width="110px" alt="Aline Fernanda"/><br>
        <sub><b>Aline Fernanda Soares Silva</b></sub><br>
        <sub>@alinesors</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ClaudersonXavier">
        <img src="https://github.com/ClaudersonXavier.png" width="110px" alt="Clauderson Branco"/><br>
        <sub><b>Clauderson Branco Xavier</b></sub><br>
        <sub>@ClaudersonXavier</sub>
      </a>
    </td>
  </tr>
</table>

---

## ⚡ Navegação Rápida

| Seção | |
|---|---|
| [Sobre o Projeto](#-sobre-o-projeto) | [Classificador Naive Bayes](#-classificador-naive-bayes) |
| [Base de Dados](#-base-de-dados) | [Divisão Treino/Teste & Resultados](#-divis%C3%A3o-treinoteste--resultados) |
| [Modelagem Univariada](#-modelagem-probabil%C3%ADstica-univariada) | [Estrutura & Execução](#-estrutura-do-projeto--execu%C3%A7%C3%A3o) |
| [Vídeo de Apresentação](#-v%C3%ADdeo-de-apresenta%C3%A7%C3%A3o) | [Declaração de Uso de IA](#-declara%C3%A7%C3%A3o-de-uso-de-ia-generativa) |
---

## 📌 Sobre o Projeto

Este Estudo Dirigido tem como objetivo investigar experimentalmente o funcionamento de **Classificadores Bayesianos** e do **Naive Bayes** utilizando uma base de dados real. 

O trabalho abrange:
1. **Modelagem Probabilística Univariada:** Análise isolada de 3 características ($X_1, X_2, X_3$).
2. **Cálculo de Verossimilhanças & Razão de Verossimilhança:** Análise do poder discriminatório de cada atributo através de $\Lambda(x) = \frac{p(x\vert{}Y=1)}{p(x\vert{}Y=0)}$.
3. **Probabilidade *a Posteriori* & Fronteiras de Decisão:** Aplicação do Teorema de Bayes $P(Y=c\vert{}x)$ e determinação gráfica das regiões de decisão.
4. **Classificador Naive Bayes Multivariado:** Combinando variáveis contínuas e categóricas no domínio logarítmico com Suavização de Laplace.
5. **Avaliação Experimental:** Matriz de confusão, Acurácia, Precisão, Recall e $F_1$-Score sobre o conjunto de teste.

---

## 🗃️ Base de Dados

* **Fonte:** [Nome da Base / Repositório (Kaggle / UCI)](LINK_DA_BASE)[cite: 2]
* **Número Total de Observações:** `[Ex: 1000]`
* **Variável Alvo ($Y$):** `[Nome da Variável Alvo]` (Classes: $Y=0$ e $Y=1$)
* **Distribuição das Classes:** 
  * Classe $Y=0$: `[Ex: 60% (600 amostras)]`[cite: 2]
  * Classe $Y=1$: `[Ex: 40% (400 amostras)]`[cite: 2]

### Atributos Selecionados ($X_1, X_2, X_3$)[cite: 2]

| Atributo | Tipo | Descrição | Distribuição Adotada $P(X_j \Vert{} Y=c)$ |
|---|---|---|---|
| **$X_1$** | Contínuo / Numérico | `[Ex: Idade]` | Normal (Gaussiana) $N(\mu, \sigma^2)$[cite: 2] |
| **$X_2$** | Contínuo / Numérico | `[Ex: Pressão Arterial]` | Normal (Gaussiana) $N(\mu, \sigma^2)$[cite: 2] |
| **$X_3$** | Categórico | `[Ex: Histórico Familiar (Sim/Não)]` | Categórica / Bernoulli (com Laplace)[cite: 2] |

---

## 📐 Modelagem Probabilística Univariada

Cada atributo é analisado individualmente nas seguintes etapas[cite: 2]:

1. **Estimativa de Parâmetros:** Cálculo de médias ($\mu$), desvios-padrão ($\sigma$) e frequências relativas no conjunto de treinamento[cite: 2].
2. **Verossimilhança $p(x \vert{} Y=c)$:** Comparação gráfica e analítica das densidades/probabilidades de cada classe[cite: 2].
3. **Razão de Verossimilhanças ($\Lambda(x)$):**
   * $\Lambda(x) > 1 \rightarrow$ Evidência a favor de $Y = 1$[cite: 2]
   * $\Lambda(x) < 1 \rightarrow$ Evidência a favor de $Y = 0$[cite: 2]
   * $\Lambda(x) = 1 \rightarrow$ Evidência neutra (indiscriminável)[cite: 2]
4. **Fronteira de Decisão:** Ponto(s) de intersecção onde $P(Y=0\vert{}x) = P(Y=1\vert{}x)$[cite: 2].

---

## 🤖 Classificador Naive Bayes

O classificador combina as três evidências assumindo **independência condicional** dada a classe[cite: 2]:

$$P(Y = c \mid x_1, x_2, x_3) \propto P(Y = c) \prod_{j=1}^{3} p(x_j \mid Y = c)$$
[cite: 2]

### Cuidados de Implementação[cite: 2]:
* **Domínio Logarítmico:** Para mitigar problemas de *underflow* numérico decorrentes da multiplicação de probabilidades muito pequenas[cite: 2]:
  $$\hat{y} = \arg\max_{c} \left[ \log P(Y = c) + \sum_{j=1}^{3} \log p(x_j \mid Y = c) \right]$$
[cite: 2]
* **Suavização de Laplace:** Aplicada no atributo categórico $X_3$ para evitar o problema da probabilidade zero[cite: 2].

---

## 🧪 Divisão Treino/Teste & Resultados

* **Proporção Treino / Teste:** `[Ex: 80% Treino / 20% Teste]`[cite: 2]
* **Semente Aleatória (`seed`):** `42`[cite: 2]
* **Amostras no Treinamento:** `[Ex: 800 observações]`[cite: 2]
* **Amostras no Teste:** `[Ex: 200 observações]`[cite: 2]

### Matriz de Confusão (Conjunto de Teste)[cite: 2]

| Real \ Predito | Predito $Y=0$ | Predito $Y=1$ |
|---|---|---|
| **Real $Y=0$** | **VN:** `[Valor]` | **FP:** `[Valor]` |
| **Real $Y=1$** | **FN:** `[Valor]` | **VP:** `[Valor]` |

### Métricas de Desempenho[cite: 2]

$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN} = \text{XX.X\%}$$
[cite: 2]

$$\text{Precisão} = \frac{VP}{VP + FP} = \text{XX.X\%}$$
[cite: 2]

$$\text{Recall} = \frac{VP}{VP + FN} = \text{XX.X\%}$$
[cite: 2]

$$F_1\text{-Score} = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}} = \text{XX.X\%}$$
[cite: 2]

---

## 📂 Estrutura do Projeto & Execução

```text
.
├── data/
│   └── pokemon_go.csv           # Dataset original de Pokémon GO
├── docs/
│   └── assets/                  # GIF do banner e imagens dos gráficos
├── src/
│   ├── data_loader.py           # Leitura e divisão Treino/Teste
│   ├── bayes_univariado.py      # Cálculos e fronteiras para X1, X2 e X3
│   ├── naive_bayes.py           # Algoritmo Naive Bayes (implementado do zero)
│   └── metrics.py               # Cálculo da Matriz de Confusão e Métricas
├── main.py                      # Script principal da pipeline
├── requirements.txt             # Dependências da aplicação
└── README.md                    # Documentação do projeto