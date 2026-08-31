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


__all__ = ["gaussian_pdf", "fit_univariate_gaussian", "likelihood_ratio"]