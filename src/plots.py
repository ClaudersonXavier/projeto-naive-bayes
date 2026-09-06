import os
import numpy as np
import matplotlib.pyplot as plt
from src.bayes_univariado import (
    categorical_pmf,
    fit_univariate_gaussian,
    gaussian_pdf,
    likelihood_ratio,
    likelihood_ratio_categorical,
)

def plot_univariate_distributions(X_train, y_train, col_name, output_dir="docs/assets"):
    """Gera o gráfico de curvas Gaussianas p(x_j | Y=c) e a Razão de Verossimilhança Lambda(x)."""
    os.makedirs(output_dir, exist_ok=True)
    params = fit_univariate_gaussian(X_train[col_name], y_train)
    
    x_min = X_train[col_name].min() - 10
    x_max = X_train[col_name].max() + 10
    x_range = np.linspace(x_min, x_max, 500)
    
    p0 = gaussian_pdf(x_range, params[0]['mean'], params[0]['std'])
    p1 = gaussian_pdf(x_range, params[1]['mean'], params[1]['std'])
    
    # 1. Gráfico das Gaussianas p(x | Y)
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, p0, label=f'Comum (Y=0) - µ={params[0]["mean"]:.1f}', color='dodgerblue', lw=2)
    plt.plot(x_range, p1, label=f'Lendário/Mítico (Y=1) - µ={params[1]["mean"]:.1f}', color='crimson', lw=2)
    plt.fill_between(x_range, p0, alpha=0.2, color='dodgerblue')
    plt.fill_between(x_range, p1, alpha=0.2, color='crimson')
    plt.title(f'Verossimilhança p({col_name} | Y)', fontsize=12, fontweight='bold')
    plt.xlabel(col_name)
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gaussian_{col_name}.png'), dpi=300)
    plt.close()

    # 2. Gráfico da Razão de Verossimilhanças Lambda(x)
    l_ratio = likelihood_ratio(x_range, params[1], params[0])
    plt.figure(figsize=(8, 4))
    plt.plot(x_range, l_ratio, color='purple', lw=2, label='Λ(x) = p(x|Y=1) / p(x|Y=0)')
    plt.axhline(y=1.0, color='black', linestyle='--', label='Evidência Neutra (Λ = 1)')
    plt.title(f'Razão de Verossimilhança - {col_name}', fontsize=12, fontweight='bold')
    plt.xlabel(col_name)
    plt.ylabel('Razão Λ(x)')
    plt.yscale('log') # Escala logarítmica para melhor visualização
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'likelihood_ratio_{col_name}.png'), dpi=300)
    plt.close()

def plot_categorical_likelihoods(params_0, params_1, col_name, output_dir="docs/assets"):
    """Gráfico de barras comparando P(categoria | Y=0) x P(categoria | Y=1),
    ordenado pela razão de verossimilhança (Λ) decrescente."""
    os.makedirs(output_dir, exist_ok=True)
    vocab = params_0["vocab"]
    lambdas = {cat: likelihood_ratio_categorical(cat, params_1, params_0) for cat in vocab}
    categorias = sorted(vocab, key=lambda c: lambdas[c], reverse=True)
    p0 = [categorical_pmf(c, params_0) for c in categorias]
    p1 = [categorical_pmf(c, params_1) for c in categorias]

    x = np.arange(len(categorias))
    largura = 0.4
    plt.figure(figsize=(max(8, len(categorias) * 0.5), 4.5))
    plt.bar(x - largura / 2, p0, width=largura, label='Comum (Y=0)', color='dodgerblue')
    plt.bar(x + largura / 2, p1, width=largura, label='Lendário/Mítico (Y=1)', color='crimson')
    plt.xticks(x, categorias, rotation=60, ha='right')
    plt.title(f'Verossimilhança P({col_name} | Y)', fontsize=12, fontweight='bold')
    plt.ylabel('P(categoria | Y)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'categorical_likelihood_{col_name}.png'), dpi=300)
    plt.close()


def plot_categorical_likelihood_ratio(params_0, params_1, col_name, output_dir="docs/assets"):
    """Gráfico de barras da razão de verossimilhança Λ por categoria, escala log."""
    os.makedirs(output_dir, exist_ok=True)
    vocab = params_0["vocab"]
    lambdas = {cat: likelihood_ratio_categorical(cat, params_1, params_0) for cat in vocab}
    categorias = sorted(vocab, key=lambda c: lambdas[c], reverse=True)
    valores = [lambdas[c] for c in categorias]

    plt.figure(figsize=(max(8, len(categorias) * 0.5), 4.5))
    cores = ['crimson' if v > 1 else 'dodgerblue' for v in valores]
    plt.bar(categorias, valores, color=cores)
    plt.axhline(y=1.0, color='black', linestyle='--', label='Evidência Neutra (Λ = 1)')
    plt.xticks(rotation=60, ha='right')
    plt.yscale('log')
    plt.title(f'Razão de Verossimilhança - {col_name}', fontsize=12, fontweight='bold')
    plt.ylabel('Razão Λ (escala log)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'categorical_likelihood_ratio_{col_name}.png'), dpi=300)
    plt.close()


def plot_decision_boundary(x_range, p0, p1, boundary_roots, col_name, output_dir="docs/assets"):
    """Estende o gráfico de densidades sombreando as regiões de decisão Y=0/Y=1
    e marcando a(s) fronteira(s) com linhas verticais tracejadas.

    boundary_roots: lista com 0, 1 ou 2 raízes (ver decision_boundary_continuous).
    A classe vencedora em cada faixa é decidida comparando p0 e p1 no ponto
    médio da faixa.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_range, p0, color='dodgerblue', lw=2, label='p(x | Y=0) - Comum')
    plt.plot(x_range, p1, color='crimson', lw=2, label='p(x | Y=1) - Lendário/Mítico')

    limites = [x_range[0]] + sorted(boundary_roots) + [x_range[-1]]
    for inicio, fim in zip(limites[:-1], limites[1:]):
        meio_idx = np.searchsorted(x_range, (inicio + fim) / 2)
        meio_idx = min(max(meio_idx, 0), len(x_range) - 1)
        vencedora_e_1 = p1[meio_idx] > p0[meio_idx]
        cor = 'crimson' if vencedora_e_1 else 'dodgerblue'
        mask = (x_range >= inicio) & (x_range <= fim)
        plt.fill_between(x_range[mask], 0, np.maximum(p0[mask], p1[mask]), color=cor, alpha=0.15)

    # Raízes fora do intervalo observado (x_range) não são plotadas: não há
    # dado real ali para visualizar, e desenhá-las expandiria o eixo x do
    # gráfico (axvline/annotate participam do autoscale do matplotlib),
    # comprimindo visualmente a região que de fato importa.
    raizes_visiveis = [r for r in boundary_roots if x_range[0] <= r <= x_range[-1]]
    for raiz in raizes_visiveis:
        plt.axvline(x=raiz, color='black', linestyle='--', lw=1.5)
        plt.annotate(f'  fronteira: {raiz:.1f}', xy=(raiz, plt.ylim()[1] * 0.9), rotation=90, va='top')

    plt.title(f'Regiões e Fronteira de Decisão - {col_name}', fontsize=12, fontweight='bold')
    plt.xlabel(col_name)
    plt.ylabel('Densidade de Probabilidade')
    plt.xlim(x_range[0], x_range[-1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decision_boundary_{col_name}.png'), dpi=300)
    plt.close()


def plot_roc_curve(fpr, tpr, auc, output_dir="docs/assets"):
    """Plota a curva ROC (FPR x TPR) com a diagonal de referência do
    classificador aleatório e a AUC no título."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Naive Bayes (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Classificador aleatório')
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(matrix, output_dir="docs/assets"):
    """Gera um gráfico visual para a Matriz de Confusão."""
    os.makedirs(output_dir, exist_ok=True)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    cm_data = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm_data, cmap=plt.cm.Blues, alpha=0.7)

    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=f"{cm_data[i, j]}", va='center', ha='center', fontsize=14, fontweight='bold')

    plt.xlabel('Predito', fontsize=11)
    plt.ylabel('Real', fontsize=11)
    plt.xticks([0, 1], ['Comum (0)', 'Lendário/Mítico (1)'])
    plt.yticks([0, 1], ['Comum (0)', 'Lendário/Mítico (1)'])
    plt.title('Matriz de Confusão', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()