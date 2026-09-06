# Roteiro do Vídeo de Apresentação

Checklist dos 20 itens de conteúdo obrigatório do §10 do enunciado, cada
um com o material já produzido (número, gráfico ou arquivo) que serve de
apoio na hora de falar. Vídeo em até **15 minutos**, formato de **conversa
técnica entre os dois integrantes** — os dois devem participar das
explicações, não só um narrando.

Antes de gravar: revisem juntos a seção "Declaração de Uso de IA
Generativa" do `README.MD` e ajustem para refletir o que cada um de vocês
efetivamente entendeu e validou — o texto lá é um rascunho baseado no
processo desta conversa, não uma versão final para vocês simplesmente
lerem.

---

| # | Item do enunciado (§10) | O que falar / material de apoio |
|---|---|---|
| 1 | O problema de classificação escolhido e a origem da base de dados | Prever se um Pokémon é Lendário/Mítico. Base: [Kaggle - patelris/pokemon-dataset-with-stats-and-types](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), 1350 observações. |
| 2 | A variável alvo e o significado das classes | `is_legendary \| is_mythical` → Y=1 (158, 11,7%) vs Y=0 comum (1192, 88,3%). Mencionar o desbalanceamento explicitamente. |
| 3 | As três características escolhidas e a justificativa para sua escolha | `base_stat_total`, `type_1`, `growth_rate` — seção "Atributos Selecionados" do README, com a justificativa de domínio de cada uma. |
| 4 | Quais características são contínuas e quais são categóricas | `base_stat_total` contínua; `type_1` (18 valores) e `growth_rate` (6 valores) categóricas. |
| 5 | O comportamento observado de cada característica dentro de cada classe | `docs/assets/gaussian_base_stat_total.png`, `categorical_likelihood_type_1.png`, `categorical_likelihood_growth_rate.png` — saída de `analise_univariada.py`, seção 1/2. |
| 6 | As distribuições probabilísticas escolhidas e a justificativa dessas escolhas | Normal para `base_stat_total` (soma de status, comportamento aproximadamente contínuo); categórica discreta com Laplace para as duas outras. Mencionar o empate técnico Normal/Gamma (seção "Limitações" do README) como exemplo de rigor na escolha. |
| 7 | O significado de p(x\|Y=c) no contexto do problema | Usar o Exemplo A (Mew) de `analise_univariada.py`, seção 3: p(x\|Y=c) é densidade/probabilidade de observar aquele valor dado a classe. |
| 8 | Exemplos de cálculo e interpretação das verossimilhanças | Mesmo exemplo do Mew: p(x\|Y=0) vs p(x\|Y=1) para `base_stat_total=600`, `type_1='Psychic'`, `growth_rate='medium-slow'`. |
| 9 | O significado e a interpretação da razão de verossimilhanças | `categorical_likelihood_ratio_growth_rate.png`: Λ(slow)=5,61 (evidência forte pró-lendário), Λ(medium)=0,017 (evidência forte pró-comum). Também `likelihood_ratio_base_stat_total.png`. |
| 10 | A aplicação do Teorema de Bayes para calcular P(Y=c\|x) | Mesmo exemplo do Mew: combinar prior (P(Y=0)=0,883) com a verossimilhança e normalizar — `analise_univariada.py` seção 3 imprime o passo a passo completo. |
| 11 | A diferença conceitual entre verossimilhança p(x\|Y=c) e posterior P(Y=c\|x) | Já explicitado na seção 3 do script: p(x\|Y=c) pode passar de 1 (densidade), P(Y=c\|x) está sempre em [0,1] e soma 1 entre as classes. |
| 12 | A regra ou fronteira de decisão obtida para cada uma das três características | `decision_boundary_base_stat_total.png` (fronteira em x≈640,4, **acima** da média da classe lendária — explicar o efeito do prior); tabelas de regra por categoria para `type_1`/`growth_rate` (`analise_univariada.py` seção 4). |
| 13 | A hipótese de independência condicional utilizada pelo Naive Bayes | Fórmula no README (seção "Classificador Naive Bayes"); mencionar a violação real medida (V de Cramér 0,27-0,37 entre `type_1` e `growth_rate`, seção "Limitações"). |
| 14 | Como características contínuas e categóricas foram combinadas no classificador | Produto (soma em log) de uma densidade Normal com duas probabilidades categóricas — `src/naive_bayes.py::predict_proba`. |
| 15 | A implementação da regra de decisão utilizando as três características | `main.py` — treina com as 3 características e decide por argmax (limiar 0,50) sobre o log-posterior. |
| 16 | A matriz de confusão obtida no conjunto de teste | Matriz oficial do README: VN=225, FP=13, FN=4, VP=28 (`avaliacao_final.py`, seção 1). |
| 17 | O significado de VP, VN, FP e FN no problema estudado | VP = lendário identificado corretamente; FN = lendário que passou despercebido; FP = comum confundido com lendário; VN = comum identificado corretamente. |
| 18 | Os principais erros observados e possíveis explicações | `avaliacao_final.py` seção 3: FP concentrados em `growth_rate=slow` + tipo de Λ alto; FN inclui `Cosmog` (outlier de baixo status). |
| 19 | As limitações das hipóteses probabilísticas adotadas | Seção "Limitações da Modelagem" do README — violação de independência, empate Normal/Gamma, formas alternativas (Mega/Gmax), efeito do prior sobre evidências individuais. |
| 20 | O que foi delegado a ferramentas de IA generativa e como o material foi verificado, corrigido e validado | Seção "Declaração de Uso de IA Generativa" do README — **revisem e personalizem antes de gravar**. |

---

## Observações finais

- Tempo sugerido: não precisa ser rígido, mas 20 itens em 15 minutos são
  ~45s de fala por item em média — vale ensaiar antes para não estourar.
- Ambos os integrantes devem demonstrar domínio de todo o conteúdo, não
  só da parte que cada um implementou — o enunciado explicitamente exige
  isso ("ambos deverão participar das explicações e demonstrar
  conhecimento sobre todo o estudo").
- "Não será aceito código produzido por IA que os integrantes não sejam
  capazes de explicar, testar, modificar e validar" — ao ensaiar, tentem
  responder "por que essa fórmula/decisão e não outra?" para cada item
  acima sem consultar o código.
