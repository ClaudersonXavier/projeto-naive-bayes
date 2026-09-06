"""
Predição de instâncias individuais com o Naive Bayes treinado.

Enquanto main.py avalia o conjunto de teste inteiro, aqui o modelo é
aplicado a um Pokémon de cada vez e o cálculo é aberto passo a passo:
prior, verossimilhança de cada característica, razão de verossimilhança
Λ(x) e o posterior normalizado. Serve tanto para conferir casos
específicos quanto para demonstrar o classificador ao vivo.

Uso:
    .venv/bin/python prever.py Mew Venusaur Cosmog
    .venv/bin/python prever.py --stats 680 --tipo Dragon --growth slow
    .venv/bin/python prever.py --vocabulario
"""
import argparse
import sys

import pandas as pd

from src.bayes_univariado import (
    categorical_pmf,
    gaussian_pdf,
)
from src.data_loader import load_data_with_names
from src.naive_bayes import PokemonNaiveBayes

LARGURA = 78
CARACTERISTICAS = ["base_stat_total", "type_1", "growth_rate"]


def treinar():
    """Treina o modelo no mesmo split usado em todo o projeto (80/20, seed=42)
    e devolve também a base completa, para buscar Pokémon pelo nome."""
    X_train, X_test, y_train, y_test = load_data_with_names(seed=42)

    modelo = PokemonNaiveBayes(alpha=1.0)
    modelo.fit(X_train[CARACTERISTICAS], y_train)

    # Base completa com a origem de cada linha: saber se o Pokémon estava no
    # treino evita a leitura errada de "o modelo acertou" num caso que ele
    # já tinha visto.
    treino = X_train.copy()
    treino["target"] = y_train.to_numpy()
    treino["particao"] = "treino"

    teste = X_test.copy()
    teste["target"] = y_test.to_numpy()
    teste["particao"] = "teste"

    base = pd.concat([treino, teste], ignore_index=True)
    return modelo, base


def buscar_por_nome(base, nome):
    """Busca exata e, se falhar, por prefixo (case-insensitive)."""
    nomes = base["name"].astype(str)
    exata = base[nomes.str.lower() == nome.lower()]
    if len(exata) > 0:
        return exata.iloc[0], None

    parciais = base[nomes.str.lower().str.startswith(nome.lower())]
    if len(parciais) == 1:
        return parciais.iloc[0], None
    if len(parciais) > 1:
        sugestoes = parciais["name"].head(8).tolist()
        return None, f"'{nome}' é ambíguo. Você quis dizer: {', '.join(sugestoes)}?"
    return None, f"'{nome}' não foi encontrado na base."


def explicar(modelo, instancia, rotulo, particao=None, verdadeiro=None):
    """Imprime o cálculo completo para uma única instância."""
    linha = instancia.iloc[0]
    stats = float(linha["base_stat_total"])
    tipo = str(linha["type_1"])
    growth = str(linha["growth_rate"])

    print("\n" + "=" * LARGURA)
    cabecalho = rotulo
    if particao is not None:
        cabecalho += f"   [na partição de {particao}]"
    print(cabecalho)
    print("=" * LARGURA)
    print(f"  X1 base_stat_total = {stats:.0f}")
    print(f"  X2 type_1          = {tipo}")
    print(f"  X3 growth_rate     = {growth}")

    prior_0 = modelo.priors_[0]
    prior_1 = modelo.priors_[1]

    gauss = modelo.gaussian_params_["base_stat_total"]
    p_stats_0 = gaussian_pdf(stats, gauss[0]["mean"], gauss[0]["std"])
    p_stats_1 = gaussian_pdf(stats, gauss[1]["mean"], gauss[1]["std"])

    tipo_params = modelo.categorical_params_["type_1"]
    p_tipo_0 = categorical_pmf(tipo, tipo_params[0])
    p_tipo_1 = categorical_pmf(tipo, tipo_params[1])

    growth_params = modelo.categorical_params_["growth_rate"]
    p_growth_0 = categorical_pmf(growth, growth_params[0])
    p_growth_1 = categorical_pmf(growth, growth_params[1])

    print("\n  " + "-" * (LARGURA - 4))
    print(f"  {'':22s} {'p(x|Y=0)':>12s} {'p(x|Y=1)':>12s} {'Λ(x)':>12s}  evidência")
    print("  " + "-" * (LARGURA - 4))

    def fmt(valor, casas=6):
        """Notação científica para valores muito pequenos ou muito grandes,
        que de outro modo apareceriam como 0,000000 na coluna."""
        limite = 10 ** -casas
        if valor != 0 and (abs(valor) < limite * 10 or abs(valor) >= 1e6):
            return f"{valor:12.3e}"
        return f"{valor:12.{casas}f}"

    def linha_evidencia(nome, p0, p1):
        lam = p1 / p0 if p0 > 0 else float("inf")
        if lam > 1.5:
            veredito = "pró-lendário"
        elif lam < 0.67:
            veredito = "pró-comum"
        else:
            veredito = "neutra"
        print(f"  {nome:22s} {fmt(p0)} {fmt(p1)} {fmt(lam, casas=3)}  {veredito}")
        return lam

    linha_evidencia("prior P(Y=c)", prior_0, prior_1)
    linha_evidencia("base_stat_total", p_stats_0, p_stats_1)
    linha_evidencia("type_1", p_tipo_0, p_tipo_1)
    linha_evidencia("growth_rate", p_growth_0, p_growth_1)
    print("  " + "-" * (LARGURA - 4))

    conjunta_0 = prior_0 * p_stats_0 * p_tipo_0 * p_growth_0
    conjunta_1 = prior_1 * p_stats_1 * p_tipo_1 * p_growth_1
    lam_total = conjunta_1 / conjunta_0 if conjunta_0 > 0 else float("inf")
    print(f"  {'produto (não norm.)':22s} {conjunta_0:12.3e} {conjunta_1:12.3e} {fmt(lam_total, casas=3)}")

    # Recalcular pelo próprio modelo garante que o passo a passo acima e a
    # decisão oficial nunca divirjam (o modelo soma em log e normaliza por
    # softmax; aqui o produto direto serve só para exibição).
    probs = modelo.predict_proba(instancia)[0]
    idx_1 = modelo.classes_.index(1)
    p1 = float(probs[idx_1])
    predicao = int(modelo.predict(instancia)[0])

    print(f"\n  Posterior:  P(Y=0|x) = {1 - p1:.4f}   P(Y=1|x) = {p1:.4f}")
    print(f"  Decisão (limiar 0,50): Y = {predicao} "
          f"({'LENDÁRIO/MÍTICO' if predicao == 1 else 'comum'})")

    if verdadeiro is not None:
        rotulo_real = "LENDÁRIO/MÍTICO" if verdadeiro == 1 else "comum"
        acertou = "acertou" if predicao == verdadeiro else "ERROU"
        print(f"  Rótulo real: Y = {verdadeiro} ({rotulo_real})  ->  o modelo {acertou}")


def imprimir_vocabulario(modelo):
    print("\nValores categóricos aceitos (exatamente como estão no treino):\n")
    for feature in modelo.categorical_features:
        vocab = modelo.categorical_params_[feature][1]["vocab"]
        print(f"  {feature} ({len(vocab)}):")
        print(f"    {', '.join(vocab)}\n")

    gauss = modelo.gaussian_params_["base_stat_total"]
    print("  base_stat_total: numérico. Referência das gaussianas ajustadas -")
    print(f"    Y=0  mu={gauss[0]['mean']:.1f}  sigma={gauss[0]['std']:.1f}")
    print(f"    Y=1  mu={gauss[1]['mean']:.1f}  sigma={gauss[1]['std']:.1f}\n")


def validar_categoria(modelo, feature, valor):
    """Avisa quando o valor cai no unknown_prob de Laplace - a predição ainda
    funciona, mas aquela característica deixa de discriminar."""
    vocab = modelo.categorical_params_[feature][1]["vocab"]
    if str(valor) not in vocab:
        print(f"\n[aviso] '{valor}' não está no vocabulário de {feature}. A predição "
              f"vai usar a probabilidade de categoria desconhecida (Laplace),")
        print(f"        o que torna essa característica praticamente neutra. "
              f"Use --vocabulario para ver os valores válidos.")


def main():
    parser = argparse.ArgumentParser(
        description="Prediz se um Pokémon é Lendário/Mítico, abrindo o cálculo passo a passo.",
        epilog="Exemplos: prever.py Mew Cosmog  |  "
               "prever.py --stats 680 --tipo Dragon --growth slow",
    )
    parser.add_argument("nomes", nargs="*", help="Um ou mais Pokémon da base, pelo nome.")
    parser.add_argument("--stats", type=float, help="base_stat_total de um Pokémon hipotético.")
    parser.add_argument("--tipo", help="type_1 de um Pokémon hipotético.")
    parser.add_argument("--growth", help="growth_rate de um Pokémon hipotético.")
    parser.add_argument("--vocabulario", action="store_true",
                        help="Lista os valores categóricos aceitos e sai.")
    args = parser.parse_args()

    modelo, base = treinar()

    if args.vocabulario:
        imprimir_vocabulario(modelo)
        return 0

    manual = [args.stats, args.tipo, args.growth]
    if any(v is not None for v in manual):
        if any(v is None for v in manual):
            parser.error("--stats, --tipo e --growth precisam ser informados juntos.")
        validar_categoria(modelo, "type_1", args.tipo)
        validar_categoria(modelo, "growth_rate", args.growth)
        instancia = pd.DataFrame([{
            "base_stat_total": args.stats,
            "type_1": args.tipo,
            "growth_rate": args.growth,
        }])
        explicar(modelo, instancia, "POKÉMON HIPOTÉTICO")
        return 0

    if not args.nomes:
        parser.print_help()
        return 1

    houve_erro = False
    for nome in args.nomes:
        registro, erro = buscar_por_nome(base, nome)
        if erro is not None:
            print(f"\n[erro] {erro}")
            houve_erro = True
            continue
        instancia = pd.DataFrame([registro[CARACTERISTICAS].to_dict()])
        explicar(
            modelo,
            instancia,
            registro["name"],
            particao=registro["particao"],
            verdadeiro=int(registro["target"]),
        )

    print()
    return 1 if houve_erro else 0


if __name__ == "__main__":
    sys.exit(main())
