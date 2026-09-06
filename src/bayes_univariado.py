from collections import Counter

import numpy as np


def gaussian_pdf(x, mean, std):
    """Calcula a densidade de probabilidade gaussiana para um valor x."""
    std = float(std)
    std = max(std, 1e-9)
    x_arr = np.asarray(x, dtype=float)
    exponent = -0.5 * ((x_arr - mean) / std) ** 2
    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    return normalization * np.exp(exponent)


def fit_univariate_gaussian(x_values, y_values):
    """Calcula média e desvio padrão de X_j por classe Y=c."""
    stats = {}
    for label in sorted(np.unique(y_values)):
        values = np.asarray(x_values[y_values == label], dtype=float)
        if values.size == 0:
            raise ValueError(f"Classe {label} não possui observações para o ajuste.")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 1.0
        stats[int(label)] = {"mean": mean, "std": std if std > 0 else 1e-9}
    return stats


def likelihood_ratio(x, params_positive, params_negative):
    """Retorna Lambda(x) = p(x | Y=1) / p(x | Y=0)."""
    numerator = gaussian_pdf(x, params_positive["mean"], params_positive["std"])
    denominator = gaussian_pdf(x, params_negative["mean"], params_negative["std"])
    denominator = np.where(denominator <= 0, 1e-9, denominator)
    return numerator / denominator


def fit_categorical(x_values, y_values, alpha=1.0):
    """Estima P(X_j = a_k | Y = c) por classe, com suavização de Laplace.

    O vocabulário de categorias é construído a partir de TODAS as
    observações de treino (não só as de uma classe), para que uma
    categoria vista apenas na classe oposta também receba uma
    probabilidade suavizada (em vez de ficar de fora do vocabulário).

    Retorna, por classe c: {"vocab": [...], "probs": {categoria: P(cat|Y=c)},
    "unknown_prob": P(categoria nunca vista|Y=c), "n": n_c}.
    """
    x_values = np.asarray(x_values, dtype=object)
    y_values = np.asarray(y_values)
    vocab = sorted({str(v) for v in x_values})
    k = len(vocab)

    stats = {}
    for label in sorted(np.unique(y_values)):
        subset = x_values[y_values == label]
        n_c = int(subset.size)
        counts = Counter(str(v) for v in subset)
        denom = n_c + alpha * (k + 1)
        probs = {v: (counts.get(v, 0) + alpha) / denom for v in vocab}
        unknown_prob = alpha / denom
        stats[int(label)] = {
            "vocab": vocab,
            "probs": probs,
            "unknown_prob": unknown_prob,
            "n": n_c,
        }
    return stats


def categorical_pmf(value, params):
    """Retorna P(X_j = value | Y = c) a partir dos parâmetros de fit_categorical."""
    return params["probs"].get(str(value), params["unknown_prob"])


def likelihood_ratio_categorical(value, params_positive, params_negative):
    """Retorna Lambda(value) = P(value | Y=1) / P(value | Y=0) para uma categórica."""
    numerator = categorical_pmf(value, params_positive)
    denominator = categorical_pmf(value, params_negative)
    denominator = denominator if denominator > 0 else 1e-9
    return numerator / denominator


def decision_boundary_continuous(params_0, params_1, prior_0, prior_1):
    """Resolve P(Y=0)*p(x|Y=0) = P(Y=1)*p(x|Y=1) para duas gaussianas.

    Em log-domínio a equação vira A*x^2 + B*x + C = 0. Quando os desvios
    padrão das duas classes são diferentes (caso comum), a curva pode
    cruzar em 0, 1 ou 2 pontos - a classe de menor variância pode ficar
    "encaixotada" num intervalo em vez de um único lado da reta.

    Retorna uma lista ordenada com as raízes reais (pode ter 0, 1 ou 2
    elementos).
    """
    mu0, sd0 = params_0["mean"], max(params_0["std"], 1e-9)
    mu1, sd1 = params_1["mean"], max(params_1["std"], 1e-9)

    K = np.log(prior_1 / prior_0) + np.log(sd0 / sd1)
    A = 1.0 / (2 * sd1**2) - 1.0 / (2 * sd0**2)
    B = mu0 / sd0**2 - mu1 / sd1**2
    C = (mu1**2) / (2 * sd1**2) - (mu0**2) / (2 * sd0**2) - K

    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        return [float(-C / B)]

    discriminante = B**2 - 4 * A * C
    if discriminante < 0:
        return []
    if discriminante == 0:
        return [float(-B / (2 * A))]

    sqrt_disc = np.sqrt(discriminante)
    raizes = sorted([float((-B - sqrt_disc) / (2 * A)), float((-B + sqrt_disc) / (2 * A))])
    return raizes


def decision_rule_categorical(params_0, params_1, prior_0, prior_1):
    """Para cada categoria do vocabulário, decide a classe pelo posterior
    proporcional a P(Y=c)*P(categoria|Y=c).

    Retorna {categoria: (classe_predita, P(Y=1|categoria))}.
    """
    vocab = params_0["vocab"]
    regra = {}
    for categoria in vocab:
        score_0 = prior_0 * categorical_pmf(categoria, params_0)
        score_1 = prior_1 * categorical_pmf(categoria, params_1)
        total = score_0 + score_1
        posterior_1 = score_1 / total if total > 0 else 0.5
        classe_predita = 1 if score_1 >= score_0 else 0
        regra[categoria] = (classe_predita, posterior_1)
    return regra


__all__ = [
    "gaussian_pdf",
    "fit_univariate_gaussian",
    "likelihood_ratio",
    "fit_categorical",
    "categorical_pmf",
    "likelihood_ratio_categorical",
    "decision_boundary_continuous",
    "decision_rule_categorical",
]