import numpy as np
import pandas as pd

from src.bayes_univariado import fit_univariate_gaussian, gaussian_pdf


class PokemonNaiveBayes:
    """Implementação manual do classificador Naive Bayes para Pokémon GO."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.priors = {}
        self.priors_ = {}
        self.feature_means = {}
        self.feature_means_ = {}
        self.feature_stds = {}
        self.feature_stds_ = {}
        self.type_probabilities = {}
        self.type_probabilities_ = {}
        self.type_vocab_ = []

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        X["base_attack"] = pd.to_numeric(X["base_attack"], errors="coerce")
        X["base_capture_rate"] = pd.to_numeric(X["base_capture_rate"], errors="coerce")
        X["type"] = X["type"].fillna("Unknown").astype(str)

        self.classes_ = sorted(y.unique().tolist())
        total_samples = len(y)
        self.priors_ = {label: np.mean(y == label) for label in self.classes_}
        self.priors = self.priors_

        self.feature_means_ = {}
        self.feature_stds_ = {}
        self.feature_means = {}
        self.feature_stds = {}
        for feature in ["base_attack", "base_capture_rate"]:
            self.feature_means_[feature] = {}
            self.feature_stds_[feature] = {}
            self.feature_means[feature] = {}
            self.feature_stds[feature] = {}
            stats = fit_univariate_gaussian(X[feature], y)
            for label in self.classes_:
                self.feature_means_[feature][label] = stats[int(label)]["mean"]
                self.feature_stds_[feature][label] = stats[int(label)]["std"]
                self.feature_means[feature][label] = self.feature_means_[feature][label]
                self.feature_stds[feature][label] = self.feature_stds_[feature][label]

        self.type_vocab_ = sorted(X["type"].dropna().unique().tolist())
        self.type_probabilities_ = {}
        self.type_probabilities = {}

        for label in self.classes_:
            subset = X.loc[y == label, "type"]
            counts = subset.value_counts()
            prob = {}
            for type_name in self.type_vocab_:
                count = float(counts.get(type_name, 0.0))
                prob[type_name] = (count + self.alpha) / (len(subset) + self.alpha * len(self.type_vocab_))
            self.type_probabilities_[label] = prob
            self.type_probabilities[label] = prob

        return self

    def _log_continuous_density(self, value, feature, label):
        mean = self.feature_means_[feature][label]
        std = self.feature_stds_[feature][label]
        pdf = gaussian_pdf(value, mean, std)
        pdf = max(pdf, 1e-300)
        return float(np.log(pdf))

    def _log_categorical_probability(self, type_value, label):
        type_value = str(type_value)
        if type_value not in self.type_vocab_:
            denominator = len(self.type_vocab_) + self.alpha * len(self.type_vocab_)
            return np.log(self.alpha / denominator)
        prob = self.type_probabilities_[label].get(type_value, self.alpha / (len(self.type_vocab_) + self.alpha * len(self.type_vocab_)))
        return float(np.log(max(prob, 1e-300)))

    def predict_proba(self, X):
        X = X.copy().reset_index(drop=True)
        X["base_attack"] = pd.to_numeric(X["base_attack"], errors="coerce").fillna(0.0)
        X["base_capture_rate"] = pd.to_numeric(X["base_capture_rate"], errors="coerce").fillna(0.0)
        X["type"] = X["type"].fillna("Unknown").astype(str)

        n_samples = len(X)
        probabilities = np.zeros((n_samples, len(self.classes_)))

        for i, row in X.iterrows():
            log_scores = []
            for label in self.classes_:
                score = np.log(self.priors_[label])
                score += self._log_continuous_density(float(row["base_attack"]), "base_attack", label)
                score += self._log_continuous_density(float(row["base_capture_rate"]), "base_capture_rate", label)
                score += self._log_categorical_probability(str(row["type"]), label)
                log_scores.append(score)
            max_log = max(log_scores)
            unnormalized = np.exp(np.array(log_scores) - max_log)
            total = np.sum(unnormalized)
            probabilities[i, :] = unnormalized / total if total > 0 else np.ones(len(self.classes_)) / len(self.classes_)

        return probabilities

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = []
        for row in probs:
            idx = int(np.argmax(row))
            predictions.append(self.classes_[idx])
        return np.asarray(predictions, dtype=int)


__all__ = ["PokemonNaiveBayes"]