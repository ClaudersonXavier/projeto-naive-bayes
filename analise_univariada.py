"""
Fase 3 - Análises univariadas (X1, X2, X3 isoladas), conforme §4 do PDF.

Complementa main.py (que implementa o Naive Bayes combinado, §5): aqui cada
característica é analisada isoladamente - hipótese distribucional,
verossimilhança, razão de verossimilhança, teorema de Bayes com exemplo
numérico completo, e regra/fronteira de decisão - fechando com a
comparação qualitativa entre os três classificadores univariados
h1(X1), h2(X2), h3(X3) exigida ao final do §4.5.

Uso: .venv/bin/python analise_univariada.py
"""
import numpy as np

from src.bayes_univariado import (
    categorical_pmf,
    decision_boundary_continuous,
    decision_rule_categorical,
    fit_categorical,
    gaussian_pdf,
    likelihood_ratio_categorical,
)
from src.data_loader import load_data_with_names
from src.metrics import calculate_metrics
from src.naive_bayes import PokemonNaiveBayes
from src.plots import (
    plot_categorical_likelihood_ratio,
    plot_categorical_likelihoods,
    plot_decision_boundary,
    plot_univariate_distributions,
)


def secao_1_parametros(h1, h2, h3):
    print("=" * 78)
    print("1) PARÂMETROS ESTIMADOS NO TREINO")
    print("=" * 78)

    gp = h1.gaussian_params_["base_stat_total"]
    print("\nX1 = base_stat_total ~ Normal(mu, sigma^2)")
    for c in (0, 1):
        print(f"  Y={c}: mu={gp[c]['mean']:8.2f}  sigma={gp[c]['std']:7.2f}")

    for nome, h, col in [("X2 = type_1", h2, "type_1"), ("X3 = growth_rate", h3, "growth_rate")]:
        cp = h.categorical_params_[col]
        print(f"\n{nome} ~ Categórica discreta (Laplace, alpha={h.alpha})")
        for cat in cp[0]["vocab"]:
            p0, p1 = categorical_pmf(cat, cp[0]), categorical_pmf(cat, cp[1])
            print(f"  {cat:<24} P(.|Y=0)={p0:.4f}   P(.|Y=1)={p1:.4f}")


def secao_2_probabilidade_zero(X_train, y_train, h3):
    """Demonstra concretamente o problema da probabilidade zero (§5.2):
    compara, para growth_rate, a estimativa SEM suavização (alpha=0, a
    frequência relativa crua) com a estimativa COM suavização de Laplace
    (alpha=1) nas categorias que não têm nenhum lendário/mítico no treino."""
    print("\n" + "=" * 78)
    print("2) O PROBLEMA DA PROBABILIDADE ZERO (§5.2)")
    print("=" * 78)

    params_sem_suavizacao = fit_categorical(X_train["growth_rate"], y_train, alpha=0.0)
    params_com_suavizacao = h3.categorical_params_["growth_rate"]

    vocab = params_sem_suavizacao[1]["vocab"]
    zero_count = [cat for cat in vocab if categorical_pmf(cat, params_sem_suavizacao[1]) == 0.0]

    print(f"\nCategorias de growth_rate SEM nenhum lendário/mítico no treino: {zero_count}")
    print(f"\n{'categoria':<24}{'P(.|Y=1) sem Laplace (a=0)':>28}{'P(.|Y=1) com Laplace (a=1)':>28}")
    for cat in zero_count:
        p_sem = categorical_pmf(cat, params_sem_suavizacao[1])
        p_com = categorical_pmf(cat, params_com_suavizacao[1])
        print(f"  {cat:<22}{p_sem:>28.4f}{p_com:>28.4f}")

    print(
        "\n  >> Sem suavização, P(categoria|Y=1)=0 para essas 3 categorias. Como o "
        "Naive Bayes soma log-probabilidades, log(0) = -infinito: QUALQUER Pokémon "
        "com uma dessas categorias em growth_rate seria classificado como Y=0 com "
        "certeza absoluta, não importa quão fortes sejam as evidências de "
        "base_stat_total e type_1 apontando para Y=1. A suavização de Laplace "
        "(alpha=1) resolve isso, atribuindo uma probabilidade pequena mas positiva "
        "a cada categoria não observada naquela classe."
    )


def secao_3_graficos(X_train, y_train, h1, h2, h3):
    print("\n" + "=" * 78)
    print("3) GRÁFICOS (salvos em docs/assets/)")
    print("=" * 78)

    plot_univariate_distributions(X_train, y_train, "base_stat_total")
    print("  -> gaussian_base_stat_total.png / likelihood_ratio_base_stat_total.png")

    for col, h in [("type_1", h2), ("growth_rate", h3)]:
        cp = h.categorical_params_[col]
        plot_categorical_likelihoods(cp[0], cp[1], col)
        plot_categorical_likelihood_ratio(cp[0], cp[1], col)
        print(f"  -> categorical_likelihood_{col}.png / categorical_likelihood_ratio_{col}.png")

    gp = h1.gaussian_params_["base_stat_total"]
    x_min = X_train["base_stat_total"].min() - 10
    x_max = X_train["base_stat_total"].max() + 10
    x_range = np.linspace(x_min, x_max, 500)
    p0 = gaussian_pdf(x_range, gp[0]["mean"], gp[0]["std"])
    p1 = gaussian_pdf(x_range, gp[1]["mean"], gp[1]["std"])
    raizes = decision_boundary_continuous(gp[0], gp[1], h1.priors_[0], h1.priors_[1])
    plot_decision_boundary(x_range, p0, p1, raizes, "base_stat_total")
    print(f"  -> decision_boundary_base_stat_total.png  (fronteira(s): {[round(r, 1) for r in raizes]})")
    return raizes


def _explicar_bayes_continuo(nome, valor, gp, prior_0, prior_1):
    p0 = float(gaussian_pdf(valor, gp[0]["mean"], gp[0]["std"]))
    p1 = float(gaussian_pdf(valor, gp[1]["mean"], gp[1]["std"]))
    n0, n1 = prior_0 * p0, prior_1 * p1
    total = n0 + n1
    post0, post1 = n0 / total, n1 / total
    print(f"\n  {nome}: base_stat_total = {valor:.0f}")
    print(f"    p(x|Y=0) = {p0:.6f}   (verossimilhança - densidade, PODE passar de 1)")
    print(f"    p(x|Y=1) = {p1:.6f}")
    print(f"    P(Y=0) = {prior_0:.4f}   P(Y=1) = {prior_1:.4f}")
    print(f"    P(Y=0)*p(x|Y=0) = {n0:.6f}   P(Y=1)*p(x|Y=1) = {n1:.6f}")
    print(f"    P(Y=0|x) = {post0:.4f}   P(Y=1|x) = {post1:.4f}   (posterior - SEMPRE em [0,1])")
    return post0, post1


def _explicar_bayes_categorico(nome_feature, valor, cp, prior_0, prior_1):
    p0 = categorical_pmf(valor, cp[0])
    p1 = categorical_pmf(valor, cp[1])
    n0, n1 = prior_0 * p0, prior_1 * p1
    total = n0 + n1
    post0, post1 = n0 / total, n1 / total
    print(f"\n  {nome_feature} = '{valor}'")
    print(f"    P(x|Y=0) = {p0:.4f}   P(x|Y=1) = {p1:.4f}")
    print(f"    P(Y=0)*P(x|Y=0) = {n0:.4f}   P(Y=1)*P(x|Y=1) = {n1:.4f}")
    print(f"    P(Y=0|x) = {post0:.4f}   P(Y=1|x) = {post1:.4f}")
    return post0, post1


def secao_4_razao_verossimilhanca(h2, h3, limiar_neutro=0.10):
    """Responde explicitamente as 3 perguntas do §4.3 para as categóricas:
    quais categorias favorecem Y=0, quais favorecem Y=1, e quais são
    praticamente neutras (Lambda perto de 1, dentro de +-limiar_neutro em
    escala log10)."""
    print("\n" + "=" * 78)
    print("4) RAZÃO DE VEROSSIMILHANÇA POR CATEGORIA - Λ(x) (§4.3)")
    print("=" * 78)

    for nome, h, col in [("X2 = type_1", h2, "type_1"), ("X3 = growth_rate", h3, "growth_rate")]:
        cp = h.categorical_params_[col]
        lambdas = {cat: likelihood_ratio_categorical(cat, cp[1], cp[0]) for cat in cp[0]["vocab"]}
        ordenado = sorted(lambdas.items(), key=lambda kv: kv[1], reverse=True)

        print(f"\n{nome}:")
        print(f"  {'categoria':<24}{'Λ(x)':>10}   interpretação")
        for cat, lam in ordenado:
            if lam > 10 ** limiar_neutro:
                rotulo = "favorece Y=1"
            elif lam < 10 ** (-limiar_neutro):
                rotulo = "favorece Y=0"
            else:
                rotulo = "neutro (pouca informação)"
            print(f"  {cat:<24}{lam:>10.3f}   {rotulo}")

        favor_1 = [c for c, l in ordenado if l > 10 ** limiar_neutro]
        favor_0 = [c for c, l in ordenado if l < 10 ** (-limiar_neutro)]
        neutros = [c for c, l in ordenado if 10 ** (-limiar_neutro) <= l <= 10 ** limiar_neutro]
        print(f"\n  Resumo: favorecem Y=1 -> {favor_1}")
        print(f"          favorecem Y=0 -> {favor_0}")
        print(f"          neutras       -> {neutros if neutros else '(nenhuma)'}")


def secao_5_exemplo_numerico(X_test, y_test, h1, h2, h3):
    print("\n" + "=" * 78)
    print("5) EXEMPLO NUMÉRICO COMPLETO - Teorema de Bayes (§4.4)")
    print("=" * 78)

    idx_lendario = y_test[y_test == 1].index[0]
    idx_comum = y_test[y_test == 0].index[0]
    exemplo_l = X_test.loc[idx_lendario]
    exemplo_c = X_test.loc[idx_comum]

    print(f"\nExemplo A - '{exemplo_l['name']}' (real: lendário/mítico, Y=1)")
    _explicar_bayes_continuo(
        "X1 (base_stat_total)", exemplo_l["base_stat_total"],
        h1.gaussian_params_["base_stat_total"], h1.priors_[0], h1.priors_[1],
    )
    _explicar_bayes_categorico(
        "X2 (type_1)", exemplo_l["type_1"],
        h2.categorical_params_["type_1"], h2.priors_[0], h2.priors_[1],
    )
    _explicar_bayes_categorico(
        "X3 (growth_rate)", exemplo_l["growth_rate"],
        h3.categorical_params_["growth_rate"], h3.priors_[0], h3.priors_[1],
    )

    print(f"\nExemplo B - '{exemplo_c['name']}' (real: comum, Y=0), só X1 para contraste")
    _explicar_bayes_continuo(
        "X1 (base_stat_total)", exemplo_c["base_stat_total"],
        h1.gaussian_params_["base_stat_total"], h1.priors_[0], h1.priors_[1],
    )

    print(
        "\n  >> Diferença conceitual (visível no Exemplo A, X1): p(x|Y=c) é uma "
        "densidade de verossimilhança - mede o quão bem a classe c explica o "
        "valor observado, e pode ultrapassar 1. P(Y=c|x) é a probabilidade "
        "a posteriori da classe dado x - sempre entre 0 e 1 e soma 1 entre "
        "as classes. O Teorema de Bayes é a ponte entre as duas, ponderando "
        "a verossimilhança pelo prior P(Y=c)."
    )


def secao_6_fronteiras(h2, h3, raizes_x1):
    print("\n" + "=" * 78)
    print("6) REGRA / FRONTEIRA DE DECISÃO POR CARACTERÍSTICA (§4.5)")
    print("=" * 78)

    print(f"\nX1 (base_stat_total): fronteira(s) em x = {[round(r, 1) for r in raizes_x1]}")
    if len(raizes_x1) == 2:
        print(
            f"  Como sigma(Y=1) < sigma(Y=0), a região Y=1 é o INTERVALO "
            f"({raizes_x1[0]:.1f}, {raizes_x1[1]:.1f}); fora dele, decide Y=0."
        )
    elif len(raizes_x1) == 1:
        print(f"  Decide Y=1 para x > {raizes_x1[0]:.1f} (ou o inverso - ver gráfico).")
    else:
        print("  Nenhuma fronteira real: uma classe domina em toda a faixa observada.")

    for nome, h, col in [("X2 (type_1)", h2, "type_1"), ("X3 (growth_rate)", h3, "growth_rate")]:
        cp = h.categorical_params_[col]
        regra = decision_rule_categorical(cp[0], cp[1], h.priors_[0], h.priors_[1])
        print(f"\n{nome}: regra de decisão por categoria")
        for cat, (classe, post1) in sorted(regra.items(), key=lambda kv: kv[1][1], reverse=True):
            print(f"  {cat:<24} -> Y={classe}   (P(Y=1|categoria)={post1:.4f})")


def secao_7_comparacao(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 78)
    print("7) COMPARAÇÃO QUALITATIVA - h1(X1), h2(X2), h3(X3) e o combinado")
    print("=" * 78)

    modelos = {
        "h1: base_stat_total": PokemonNaiveBayes(continuous_features=("base_stat_total",), categorical_features=()),
        "h2: type_1": PokemonNaiveBayes(continuous_features=(), categorical_features=("type_1",)),
        "h3: growth_rate": PokemonNaiveBayes(continuous_features=(), categorical_features=("growth_rate",)),
        "combinado (Fase 2)": PokemonNaiveBayes(),
    }

    print(f"\n{'classificador':<24}{'Acur':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    print("-" * 56)
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        m = calculate_metrics(y_test, y_pred)
        print(f"{nome:<24}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1_score']:>8.3f}")


def main():
    X_train, X_test, y_train, y_test = load_data_with_names(seed=42)
    Xtr_sem_nome = X_train.drop(columns=["name"])
    Xte_sem_nome = X_test.drop(columns=["name"])

    h1 = PokemonNaiveBayes(continuous_features=("base_stat_total",), categorical_features=()).fit(Xtr_sem_nome, y_train)
    h2 = PokemonNaiveBayes(continuous_features=(), categorical_features=("type_1",)).fit(Xtr_sem_nome, y_train)
    h3 = PokemonNaiveBayes(continuous_features=(), categorical_features=("growth_rate",)).fit(Xtr_sem_nome, y_train)

    secao_1_parametros(h1, h2, h3)
    secao_2_probabilidade_zero(Xtr_sem_nome, y_train, h3)
    raizes_x1 = secao_3_graficos(Xtr_sem_nome, y_train, h1, h2, h3)
    secao_4_razao_verossimilhanca(h2, h3)
    secao_5_exemplo_numerico(X_test, y_test, h1, h2, h3)
    secao_6_fronteiras(h2, h3, raizes_x1)
    secao_7_comparacao(Xtr_sem_nome, y_train, Xte_sem_nome, y_test)


if __name__ == "__main__":
    main()
