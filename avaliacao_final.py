"""
Fase 4 - Avaliação final do Naive Bayes combinado (§7 do PDF e itens
13-15 do §8), com o material extra que sustenta uma leitura crítica dos
resultados: comparação com o baseline "sempre comum", interpretação
nominal dos falsos positivos/negativos, curva ROC/AUC e a varredura de
limiar que já embasou a escolha de 0,50 na Fase 1.

Complementa main.py (que já treina o modelo e imprime a matriz de
confusão/métricas no limiar padrão) - aqui aprofunda a avaliação sem
repetir a implementação do classificador.

Uso: .venv/bin/python avaliacao_final.py
"""
import numpy as np

from src.data_loader import load_data_with_names
from src.metrics import calculate_metrics, metrics_at_threshold, roc_curve_auc
from src.naive_bayes import PokemonNaiveBayes
from src.plots import plot_confusion_matrix, plot_roc_curve


def treinar(X_train, y_train):
    modelo = PokemonNaiveBayes()
    modelo.fit(X_train, y_train)
    return modelo


def secao_1_matriz_oficial(modelo, X_test, y_test):
    print("=" * 78)
    print("1) MATRIZ DE CONFUSÃO E MÉTRICAS OFICIAIS (limiar 0,50, teste)")
    print("=" * 78)
    idx1 = modelo.classes_.index(1)
    p1 = modelo.predict_proba(X_test)[:, idx1]
    m = metrics_at_threshold(y_test, p1, 0.50)
    tn, fp = m["matrix"][0]
    fn, tp = m["matrix"][1]
    print(f"\n              Predito 0   Predito 1")
    print(f"  Real 0        {tn:>6}      {fp:>6}")
    print(f"  Real 1        {fn:>6}      {tp:>6}")
    print(f"\n  Acurácia:  {m['accuracy']*100:.2f}%")
    print(f"  Precisão:  {m['precision']*100:.2f}%")
    print(f"  Recall:    {m['recall']*100:.2f}%")
    print(f"  F1-Score:  {m['f1_score']*100:.2f}%")
    plot_confusion_matrix(m["matrix"])
    print("\n  -> confusion_matrix.png")
    return p1, m


def secao_2_baseline(y_test, m_modelo):
    print("\n" + "=" * 78)
    print("2) BASELINE 'SEMPRE COMUM' vs MODELO")
    print("=" * 78)
    baseline_pred = np.zeros(len(y_test), dtype=int)
    m_base = calculate_metrics(y_test, baseline_pred)

    n_legendarios = int(np.sum(np.asarray(y_test) == 1))
    erros_baseline = n_legendarios
    erros_modelo = m_modelo["fp"] + m_modelo["fn"]

    print(f"\n{'':<24}{'Acurácia':>10}{'Recall':>10}{'F1':>10}")
    print(f"{'baseline (sempre 0)':<24}{m_base['accuracy']:>10.3f}{m_base['recall']:>10.3f}{m_base['f1_score']:>10.3f}")
    print(f"{'modelo (limiar 0,50)':<24}{m_modelo['accuracy']:>10.3f}{m_modelo['recall']:>10.3f}{m_modelo['f1_score']:>10.3f}")

    print(f"\n  Lendários/míticos detectados: baseline 0/{n_legendarios}   modelo {m_modelo['tp']}/{n_legendarios}")
    print(f"  Erros totais: baseline {erros_baseline} (todos FN)   modelo {erros_modelo} (FP={m_modelo['fp']}, FN={m_modelo['fn']})")
    print(f"  Redução relativa de erros: {100 * (1 - erros_modelo / erros_baseline):.1f}%")
    print(
        "\n  >> A acurácia sozinha é enganosa aqui: o baseline já atinge "
        f"{m_base['accuracy']*100:.1f}% só por prever sempre a classe majoritária, "
        "mas tem recall zero - nunca identifica um lendário/mítico."
    )


def secao_3_fp_fn(X_test_named, y_test, p1, threshold=0.50):
    print("\n" + "=" * 78)
    print("3) INTERPRETAÇÃO DOS FALSOS POSITIVOS E FALSOS NEGATIVOS")
    print("=" * 78)
    y_true = np.asarray(y_test)
    y_pred = (p1 >= threshold).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    cols = ["name", "base_stat_total", "type_1", "growth_rate"]
    print(f"\n  Falsos Positivos ({int(fp_mask.sum())}) - comuns classificados como lendário/mítico:")
    for _, row in X_test_named.loc[fp_mask, cols].iterrows():
        print(f"    {row['name']:<24} base_stat_total={row['base_stat_total']:>5.0f}  type_1={row['type_1']:<10} growth_rate={row['growth_rate']}")

    print(f"\n  Falsos Negativos ({int(fn_mask.sum())}) - lendários/míticos classificados como comum:")
    for _, row in X_test_named.loc[fn_mask, cols].iterrows():
        print(f"    {row['name']:<24} base_stat_total={row['base_stat_total']:>5.0f}  type_1={row['type_1']:<10} growth_rate={row['growth_rate']}")


def secao_4_roc(y_test, p1):
    print("\n" + "=" * 78)
    print("4) CURVA ROC E AUC")
    print("=" * 78)
    fpr, tpr, auc = roc_curve_auc(y_test, p1)
    plot_roc_curve(fpr, tpr, auc)
    print(f"\n  AUC-ROC = {auc:.4f}")
    print("  -> roc_curve.png")


def secao_5_limiar(y_test, p1):
    print("\n" + "=" * 78)
    print("5) VARREDURA DE LIMIAR (justificativa para manter 0,50)")
    print("=" * 78)
    print(f"\n{'limiar':>8}{'Acur':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    melhor = None
    for th in np.arange(0.05, 1.00, 0.05):
        m = metrics_at_threshold(y_test, p1, th)
        marcador = " <-- padrão" if abs(th - 0.50) < 1e-9 else ""
        print(f"{th:>8.2f}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1_score']:>8.3f}{marcador}")
        if melhor is None or m["f1_score"] > melhor[0]:
            melhor = (m["f1_score"], th)

    f1_melhor, th_melhor = melhor
    m_padrao = metrics_at_threshold(y_test, p1, 0.50)
    print(f"\n  Melhor F1 no split oficial: {f1_melhor:.3f} (limiar {th_melhor:.2f})")
    print(f"  F1 no limiar padrão 0,50:    {m_padrao['f1_score']:.3f}")
    print("  Diferença desprezível -> mantemos o limiar padrão 0,50 (argmax), como decidido na Fase 1.")


def main():
    X_train, X_test, y_train, y_test = load_data_with_names(seed=42)
    Xtr = X_train.drop(columns=["name"])
    Xte = X_test.drop(columns=["name"])

    modelo = treinar(Xtr, y_train)
    p1, m_modelo = secao_1_matriz_oficial(modelo, Xte, y_test)
    secao_2_baseline(y_test, m_modelo)
    secao_3_fp_fn(X_test, y_test, p1)
    secao_4_roc(y_test, p1)
    secao_5_limiar(y_test, p1)


if __name__ == "__main__":
    main()
