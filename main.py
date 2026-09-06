from src.data_loader import load_data
from src.naive_bayes import PokemonNaiveBayes
from src.metrics import calculate_metrics
from src.plots import plot_univariate_distributions, plot_confusion_matrix

def main():
    print("==================================================")
    print("   CLASSIFICADOR NAIVE BAYES - LENDÁRIOS POKÉMON   ")
    print("==================================================\n")

    # 1. Carregamento dos dados via Kaggle / Local
    print("[1/4] Carregando base de dados...")
    X_train, X_test, y_train, y_test = load_data(seed=42)
    print(f"-> Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

    # 2. Treinamento do modelo
    print("[2/4] Ajustando distribuições e treinando Naive Bayes...")
    model = PokemonNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    print("-> Priors P(Y):", {c: round(p, 4) for c, p in model.priors_.items()})
    print("-> Parâmetros ajustados com sucesso!\n")
    
    # 3. Predição e Métricas
    print("[3/4] Avaliando desempenho no conjunto de teste...")
    y_pred = model.predict(X_test)
    results = calculate_metrics(y_test, y_pred)
    
    tn, fp = results['matrix'][0]
    fn, tp = results['matrix'][1]
    
    print("\n---------------- Matriz de Confusão ----------------")
    print(f" Verdadeiros Negativos (VN): {tn:<4} | Falsos Positivos (FP): {fp}")
    print(f" Falsos Negativos (FN):     {fn:<4} | Verdadeiros Positivos (TP): {tp}")
    print("----------------------------------------------------")
    print(f" Acurácia:  {results['accuracy']*100:.2f}%")
    print(f" Precisão:  {results['precision']*100:.2f}%")
    print(f" Recall:    {results['recall']*100:.2f}%")
    print(f" F1-Score:  {results['f1_score']*100:.2f}%")
    print("----------------------------------------------------\n")
    
    # 4. Geração de Gráficos para o README e Apresentação
    print("[4/4] Gerando gráficos univariados e matriz em 'docs/assets/'...")
    plot_univariate_distributions(X_train, y_train, 'base_stat_total')
    plot_confusion_matrix(results['matrix'])
    
    print("-> Sucesso! Todos os gráficos foram salvos na pasta 'docs/assets/'.\n")

if __name__ == "__main__":
    main()