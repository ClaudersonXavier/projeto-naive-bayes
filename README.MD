<div align="center">
  <img src="docs/assets/pokemon-banner.gif" alt="Pokémon Banner" width="100%" height="280px" style="object-fit: cover; border-radius: 10px;"/>
</div>

<br/>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen)

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
| [Limitações](#-limita%C3%A7%C3%B5es-da-modelagem) | [Vídeo de Apresentação](#-v%C3%ADdeo-de-apresenta%C3%A7%C3%A3o) |
| [Declaração de Uso de IA](#-declara%C3%A7%C3%A3o-de-uso-de-ia-generativa) | |

---

## 📌 Sobre o Projeto

Este Estudo Dirigido investiga experimentalmente o funcionamento de **Classificadores Bayesianos** e do **Naive Bayes**, usando como problema real **prever se um Pokémon é Lendário/Mítico a partir dos seus atributos de jogo**.

O trabalho abrange:
1. **Modelagem Probabilística Univariada:** Análise isolada de 3 características ($X_1, X_2, X_3$).
2. **Cálculo de Verossimilhanças & Razão de Verossimilhança:** Análise do poder discriminatório de cada atributo através de $\Lambda(x) = \frac{p(x\vert{}Y=1)}{p(x\vert{}Y=0)}$.
3. **Probabilidade *a Posteriori* & Fronteiras de Decisão:** Aplicação do Teorema de Bayes $P(Y=c\vert{}x)$ e determinação gráfica das regiões de decisão.
4. **Classificador Naive Bayes Multivariado:** Combinando uma variável contínua e duas categóricas no domínio logarítmico com Suavização de Laplace.
5. **Avaliação Experimental:** Matriz de confusão, Acurácia, Precisão, Recall e $F_1$-Score sobre o conjunto de teste, com comparação a um baseline trivial e curva ROC/AUC.

---

## 🗃️ Base de Dados

* **Fonte:** [Pokémon Dataset with Stats and Types (Kaggle)](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types)
* **Número Total de Observações:** `1350` (após limpeza; nenhuma linha descartada, pois o trio de características escolhido não possui valores ausentes)
* **Variável Alvo ($Y$):** Pokémon é **Lendário ou Mítico** (`is_legendary | is_mythical`) — $Y=1$; caso contrário, $Y=0$
* **Distribuição das Classes:**
  * Classe $Y=0$ (comum): `1192 amostras (88,3%)`
  * Classe $Y=1$ (lendário/mítico): `158 amostras (11,7%)`

> A base é **fortemente desbalanceada** (razão ~7,6:1). Isso é discutido explicitamente na seção de [Resultados](#-divis%C3%A3o-treinoteste--resultados), onde comparamos o modelo a um baseline trivial para evitar conclusões enganosas baseadas só em acurácia.

### Atributos disponíveis na base bruta

A base original tem 31 colunas. Antes de escolher o trio, mapeamos todas quanto ao tipo (numérica x categórica) e à qualidade (valores ausentes, cardinalidade):

| Categoria | Colunas | Observação |
|---|---|---|
| **Numéricas utilizáveis** | `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`, `base_stat_total`, `height_m`, `weight_kg`, `base_experience` (48 ausentes), `base_happiness`, `capture_rate` | `base_stat_total` é a soma das 6 primeiras — usar as duas seria redundante. `capture_rate` tem boa separação entre classes mas é um forte candidato a categórica (poucos valores distintos concentrados); `base_happiness` e `habitat="rare"` funcionam quase como sinônimo direto de "lendário" no design do jogo, então foram evitados por risco de vazamento conceitual. |
| **Categóricas utilizáveis** | `type_1` (0 ausentes, 18 categorias), `type_2` (585 ausentes, 18), `generation` (0, 9), `color` (0, 10), `shape` (0, 14), `egg_groups` (0, 65), `habitat` (830 ausentes, 9), `growth_rate` (0, 6) | `type_2` tem 43% de ausência (só Pokémon com tipo duplo); `habitat` tem 61% de ausência; `egg_groups` inclui a categoria `no-eggs`, que é quase sinônimo de "não se reproduz" = lendário — evitada pelo mesmo motivo do `base_happiness`. |
| **Identificadores / texto livre / alta cardinalidade** | `pokedex_number`, `name`, `abilities` (715 valores), `hidden_ability` (362 ausentes), `genus` (715), `evolution_chain_id` (541), `flavor_text` (1025), `sprite_url` (1290) | Sem uso direto como característica de classificação. |
| **Booleanas** | `is_legendary`, `is_mythical`, `is_baby` | As duas primeiras compõem a variável alvo $Y$; `is_baby` não foi usada. |

Do conjunto acima, escolhemos o trio abaixo por combinar boa capacidade de discriminação (medida via $\Lambda(x)$ e comparação de F1 em validação) com justificativa de domínio genuína — evitando os atributos que "vazam" a resposta por definição (`habitat="rare"`, `egg_groups="no-eggs"`, `base_happiness=0`).

### Atributos Selecionados ($X_1, X_2, X_3$)

| Atributo | Tipo | Descrição | Distribuição Adotada $P(X_j \vert Y=c)$ |
|---|---|---|---|
| **$X_1$ = `base_stat_total`** | Contínuo / Numérico | Soma dos 6 status base (HP, Ataque, Defesa, Atq. Especial, Def. Especial, Velocidade) — mede o "poder geral" do Pokémon | Normal (Gaussiana) $N(\mu, \sigma^2)$ |
| **$X_2$ = `type_1`** | Categórico (18 valores) | Tipo primário do Pokémon (Água, Fogo, Psíquico, Dragão...) | Categórica discreta (com Laplace) |
| **$X_3$ = `growth_rate`** | Categórico (6 valores) | Taxa de ganho de experiência (`slow`, `medium`, `fast`...) | Categórica discreta (com Laplace) |

**Justificativa de domínio:**
- `base_stat_total` é a soma direta dos atributos de combate — espera-se que Pokémon lendários/míticos, projetados para serem mais fortes, concentrem valores mais altos (confirmado: $\mu_0=431,9$ vs $\mu_1=618,0$ no treino).
- `type_1`: conhecimento consensual sobre a franquia associa os tipos **Psíquico** e **Dragão** a lendários com maior frequência — confirmado quantitativamente ($\Lambda_{\text{Psychic}}=3,57$, $\Lambda_{\text{Dragon}}=2,49$, os dois maiores entre os 18 tipos).
- `growth_rate`: no design do jogo, Pokémon lendários/míticos tipicamente têm taxa de XP `slow` — confirmado ($\Lambda_{\text{slow}}=5,67$, evidência de verossimilhança mais forte entre as 6 categorias).

---

## 📐 Modelagem Probabilística Univariada

Cada atributo é analisado **isoladamente** em `analise_univariada.py`, nas seguintes etapas:

1. **Estimativa de Parâmetros:** médias/desvios (contínua) e frequências relativas por classe com suavização de Laplace $\alpha=1$ (categóricas), no conjunto de treinamento.
2. **Verossimilhança $p(x \vert Y=c)$:** comparação gráfica das densidades/probabilidades de cada classe (`docs/assets/gaussian_base_stat_total.png`, `categorical_likelihood_type_1.png`, `categorical_likelihood_growth_rate.png`).
3. **Razão de Verossimilhanças ($\Lambda(x)$):**
   * $\Lambda(x) > 1 \rightarrow$ evidência a favor de $Y=1$
   * $\Lambda(x) < 1 \rightarrow$ evidência a favor de $Y=0$
   * $\Lambda(x) = 1 \rightarrow$ evidência neutra (indiscriminável)
4. **Fronteira de Decisão:** ponto(s) de intersecção onde $P(Y=0\vert x) = P(Y=1\vert x)$.

### Achado interessante: a fronteira de `base_stat_total` fica *acima* da própria média da classe lendária

A fronteira calculada fica em $x \approx 640,4$ — maior que $\mu_1=618,0$. Isso acontece porque o prior é fortemente desbalanceado ($P(Y=0)=0,883$): mesmo no pico da distribuição lendária, a classe comum ainda tem massa suficiente (por ser mais larga, $\sigma_0=109,2$ vs $\sigma_1=86,4$) para vencer. Só valores bem acima da média lendária têm evidência forte o bastante para superar o prior. Ilustra bem a pergunta-guia: *"Como as probabilidades a priori das classes influenciam uma decisão Bayesiana?"*

### Comparação qualitativa entre os três classificadores univariados

| Classificador | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| $h_1$ (`base_stat_total`) | 0,933 | 1,000 | 0,438 | 0,609 |
| $h_2$ (`type_1`) | 0,881 | 0,000 | 0,000 | 0,000 |
| $h_3$ (`growth_rate`) | 0,881 | 0,000 | 0,000 | 0,000 |
| **Combinado (as 3 juntas)** | **0,937** | **0,683** | **0,875** | **0,767** |

Isoladamente, nem `type_1` nem `growth_rate` conseguem virar o prior sozinhos: mesmo `growth_rate=slow` tendo $\Lambda=5,67$ (evidência de verossimilhança forte), o posterior fica em $P(Y=1\vert\text{slow})=0,428$ — abaixo de 0,5, porque as odds do prior (~0,132) não são superadas por essa evidência isolada. É só **combinando as três evidências** que o classificador consegue discriminar bem (F1 salta de 0/0,609 isolados para 0,767) — a motivação prática por trás do Naive Bayes.

---

## 🤖 Classificador Naive Bayes

O classificador combina as três evidências assumindo **independência condicional** dada a classe:

$$P(Y = c \mid x_1, x_2, x_3) \propto P(Y = c) \prod_{j=1}^{3} p(x_j \mid Y = c)$$

### Cuidados de Implementação:
* **Domínio Logarítmico:** para mitigar problemas de *underflow* numérico decorrentes da multiplicação de probabilidades muito pequenas:
  $$\hat{y} = \arg\max_{c} \left[ \log P(Y = c) + \sum_{j=1}^{3} \log p(x_j \mid Y = c) \right]$$
* **Suavização de Laplace ($\alpha=1$):** aplicada nos atributos categóricos $X_2$ e $X_3$ para evitar o problema da probabilidade zero — `growth_rate` tem 4 categorias (`medium`, `fast`, `slow-then-very-fast`, `fast-then-very-slow`) sem nenhum lendário/mítico no treino; sem suavização, `P(categoria|Y=1)=0` anularia o posterior de qualquer Pokémon com esses valores, não importa a evidência das outras características.

---

## 🧪 Divisão Treino/Teste & Resultados

* **Proporção Treino / Teste:** `80% Treino / 20% Teste`
* **Semente Aleatória (`seed`):** `42`
* **Amostras no Treinamento:** `1080` (126 lendários/míticos, 954 comuns)
* **Amostras no Teste:** `270` (32 lendários/míticos, 238 comuns)
* **Desbalanceamento:** preservado nas duas partições pela divisão estratificada (`stratified_split`), mantendo a proporção ~88,3%/11,7% em ambas.

### Matriz de Confusão (Conjunto de Teste, limiar 0,50)

| Real \ Predito | Predito $Y=0$ | Predito $Y=1$ |
|---|---|---|
| **Real $Y=0$** | **VN:** `225` | **FP:** `13` |
| **Real $Y=1$** | **FN:** `4` | **VP:** `28` |

### Métricas de Desempenho

$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN} = 93,70\%$$

$$\text{Precisão} = \frac{VP}{VP + FP} = 68,29\%$$

$$\text{Recall} = \frac{VP}{VP + FN} = 87,50\%$$

$$F_1\text{-Score} = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}} = 76,71\%$$

**AUC-ROC = 0,9585** (`docs/assets/roc_curve.png`) — a curva sobe rápido: com apenas ~4% de falso alarme entre os comuns, já se identificam ~81% dos lendários/míticos.

### Por que acurácia sozinha não conta a história toda

| | Acurácia | Recall | F1 | Lendários/míticos detectados |
|---|---|---|---|---|
| Baseline "sempre comum" | 88,1% | 0,0% | 0,000 | 0 de 32 |
| **Nosso modelo** | **93,7%** | **87,5%** | **0,767** | **28 de 32** |

Numa base 88/12, um classificador que nunca aprende nada já acerta 88,1% só chutando a classe majoritária — e tem recall zero. O ganho real do Naive Bayes não está nos poucos pontos de acurácia a mais, e sim em passar de **zero** lendários detectados para **28 de 32**, reduzindo os erros totais de 32 (todos falsos negativos) para 17 (13 FP + 4 FN) — uma redução de 46,9%.

### Interpretação dos erros (via `avaliacao_final.py`)

- **Falsos Positivos (13):** quase todos têm `growth_rate=slow` combinado com tipos de $\Lambda$ alto (Dragão, Psíquico, Aço, Lutador) — casos onde as evidências de $X_2$/$X_3$ são fortes o bastante para o modelo errar a favor de "lendário" mesmo sem ser.
- **Falsos Negativos (4):** incluem `Cosmog` (`base_stat_total=200`), um mítico "pré-evolução" com status de bebê — um outlier real da própria mecânica do jogo, difícil de acertar com qualquer uma das 3 características usadas.


---

## 📂 Estrutura do Projeto & Execução

```text
.
├── data/
│   └── pokemon.csv                      # Base baixada do Kaggle (gerado na 1ª execução)
├── docs/
│   ├── assets/                          # GIF do banner e todos os gráficos gerados
├── src/
│   ├── data_loader.py                   # Download/leitura da base e divisão treino/teste
│   ├── bayes_univariado.py              # Distribuições, Λ(x), fronteiras e regras de decisão
│   ├── naive_bayes.py                   # Naive Bayes combinado (implementado do zero)
│   ├── metrics.py                       # Matriz de confusão, métricas, ROC/AUC
│   └── plots.py                         # Todos os gráficos do projeto
├── main.py                              # Pipeline principal: treina e avalia o NB combinado
├── analise_univariada.py                # Análise isolada de X1, X2, X3 (§4 do enunciado)
├── avaliacao_final.py                   # Avaliação aprofundada: baseline, FP/FN, ROC, limiar
├── prever.py                            # Predição de instâncias individuais, passo a passo
├── requirements.txt                     # Dependências da aplicação
└── README.md                            # Esta documentação
```

### Instalação

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell ou Prompt de Comando)**

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Execução

**Linux / macOS**

```bash
.venv/bin/python main.py                 # Treina o Naive Bayes combinado e avalia no teste
.venv/bin/python analise_univariada.py   # Análises isoladas de X1, X2, X3 (§4)
.venv/bin/python avaliacao_final.py      # Avaliação aprofundada (§7): baseline, FP/FN, ROC
.venv/bin/python prever.py Mew           # Predição de um Pokémon, passo a passo (ver abaixo)
```

**Windows**

```powershell
.venv\Scripts\python main.py
.venv\Scripts\python analise_univariada.py
.venv\Scripts\python avaliacao_final.py
.venv\Scripts\python prever.py Mew
```

### Testando instâncias individuais

`prever.py` aplica o modelo já treinado a um Pokémon de cada vez e abre o cálculo passo a passo — prior, verossimilhança de cada característica, razão de verossimilhança $\Lambda(x)$ e o posterior normalizado:

```bash
.venv/bin/python prever.py Mew Cosmog Rayquaza          # pelo nome, com o rótulo real ao final
.venv/bin/python prever.py --stats 680 --tipo Dragon --growth slow   # Pokémon hipotético
.venv/bin/python prever.py --vocabulario                # valores categóricos aceitos
```

Ao buscar pelo nome, a saída informa se o Pokémon estava no treino ou no teste — um acerto em exemplo de treino não é evidência de generalização. Valores categóricos fora do vocabulário não geram erro: caem na probabilidade de categoria desconhecida da suavização de Laplace, e o script avisa quando isso acontece.

Na primeira execução, `data_loader.py` baixa a base automaticamente do Kaggle via `kagglehub` e salva em `data/pokemon.csv`; execuções seguintes reutilizam o arquivo local.

