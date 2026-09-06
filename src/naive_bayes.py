import numpy as np
import pandas as pd

from src.bayes_univariado import (
    categorical_pmf,
    fit_categorical,
    fit_univariate_gaussian,
    gaussian_pdf,
)


class PokemonNaiveBayes:
    """Classificador Naive Bayes implementado do zero.

    Combina, por padrão, uma característica contínua (`base_stat_total`,
    modelada como Normal por classe) e duas categóricas (`type_1` e
    `growth_rate`, modeladas como distribuições discretas com suavização de
    Laplace), conforme validado no estudo dirigido.
    """

    def __init__(
        self,
        continuous_features=("base_stat_total",),
        categorical_features=("type_1", "growth_rate"),
        alpha=1.0,
    ):
        self.continuous_features = tuple(continuous_features)
        self.categorical_features = tuple(categorical_features)
        self.alpha = float(alpha)
        self.classes_ = None
        self.priors_ = {}
        self.gaussian_params_ = {}
        self.categorical_params_ = {}

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.classes_ = sorted(y.unique().tolist())
        self.priors_ = {label: float(np.mean(y == label)) for label in self.classes_}

        self.gaussian_params_ = {}
        for feature in self.continuous_features:
            values = pd.to_numeric(X[feature], errors="coerce")
            self.gaussian_params_[feature] = fit_univariate_gaussian(values, y)

        self.categorical_params_ = {}
        for feature in self.categorical_features:
            values = X[feature].astype(str)
            self.categorical_params_[feature] = fit_categorical(values, y, alpha=self.alpha)

        return self

    def _log_continuous_density(self, value, feature, label):
        params = self.gaussian_params_[feature][label]
        pdf = gaussian_pdf(value, params["mean"], params["std"])
        pdf = max(float(pdf), 1e-300)
        return float(np.log(pdf))

    def _log_categorical_probability(self, value, feature, label):
        params = self.categorical_params_[feature][label]
        prob = categorical_pmf(value, params)
        return float(np.log(max(prob, 1e-300)))

    def predict_proba(self, X):
        """Retorna P(Y=c | x) para cada classe, uma coluna por classe em self.classes_."""
        X = X.copy().reset_index(drop=True)
        for feature in self.continuous_features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        for feature in self.categorical_features:
            X[feature] = X[feature].astype(str)

        n_samples = len(X)
        probabilities = np.zeros((n_samples, len(self.classes_)))

        for i, row in X.iterrows():
            log_scores = []
            for label in self.classes_:
                score = np.log(self.priors_[label])
                for feature in self.continuous_features:
                    score += self._log_continuous_density(float(row[feature]), feature, label)
                for feature in self.categorical_features:
                    score += self._log_categorical_probability(row[feature], feature, label)
                log_scores.append(score)
            max_log = max(log_scores)
            unnormalized = np.exp(np.array(log_scores) - max_log)
            total = np.sum(unnormalized)
            probabilities[i, :] = (
                unnormalized / total if total > 0 else np.ones(len(self.classes_)) / len(self.classes_)
            )

        return probabilities

    def predict(self, X, threshold=0.5):
        """Prediz a classe. Em problemas binários, decide por P(Y=1|x) >= threshold
        (threshold=0.5 equivale ao argmax padrão do Naive Bayes)."""
        probs = self.predict_proba(X)
        if len(self.classes_) == 2:
            idx_classe_1 = self.classes_.index(1) if 1 in self.classes_ else 1
            return (probs[:, idx_classe_1] >= threshold).astype(int)

        predictions = [self.classes_[int(np.argmax(row))] for row in probs]
        return np.asarray(predictions, dtype=int)


__all__ = ["PokemonNaiveBayes"]
