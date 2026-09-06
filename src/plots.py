import os
import numpy as np
import matplotlib.pyplot as plt
from src.bayes_univariado import gaussian_pdf, fit_univariate_gaussian, likelihood_ratio

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
    plt.plot(x_range, p1, label=f'Lendário (Y=1) - µ={params[1]["mean"]:.1f}', color='crimson', lw=2)
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
    plt.xticks([0, 1], ['Comum (0)', 'Lendário (1)'])
    plt.yticks([0, 1], ['Comum (0)', 'Lendário (1)'])
    plt.title('Matriz de Confusão', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()