import glob
import os
import shutil

import kagglehub
import numpy as np
import pandas as pd


DATA_DIR = "data"
TARGET_CSV = os.path.join(DATA_DIR, "pokemon.csv")
SOURCE_CSV_NAME = "pokemon_complete.csv"


def get_csv_path() -> str:
    """Baixa o dataset do Kaggle ou reutiliza uma cópia local."""
    if os.path.exists(TARGET_CSV):
        return TARGET_CSV

    print("[data_loader] Baixando dataset do Kaggle via kagglehub...")
    dataset_path = kagglehub.dataset_download("patelris/pokemon-dataset-with-stats-and-types")

    csv_files = glob.glob(os.path.join(dataset_path, "**", SOURCE_CSV_NAME), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"Arquivo '{SOURCE_CSV_NAME}' não foi encontrado no dataset baixado."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    source_path = csv_files[0]
    shutil.copy2(source_path, TARGET_CSV)
    print(f"[data_loader] Arquivo salvo em: {TARGET_CSV}")
    return TARGET_CSV


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = 42):
    """Divide os dados preservando a proporção das classes."""
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for label in sorted(y.unique()):
        label_idx = np.where(y.to_numpy() == label)[0]
        rng.shuffle(label_idx)
        n_test = max(1, int(round(len(label_idx) * test_size))) if len(label_idx) > 1 else 0
        n_test = min(n_test, len(label_idx) - 1) if len(label_idx) > 1 else 0
        test_idx.extend(label_idx[:n_test].tolist())
        train_idx.extend(label_idx[n_test:].tolist())

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def _load_clean_dataframe() -> pd.DataFrame:
    """Lê o dataset e devolve o DataFrame limpo com 'name', o trio de
    características e o alvo já calculado. Uso interno de load_data() e
    load_data_with_names()."""
    filepath = get_csv_path()
    df = pd.read_csv(filepath)

    required_columns = ["name", "base_stat_total", "type_1", "growth_rate", "is_legendary", "is_mythical"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes no dataset: {missing}")

    df = df.loc[:, required_columns].copy()
    df["base_stat_total"] = pd.to_numeric(df["base_stat_total"], errors="coerce")
    df["type_1"] = df["type_1"].astype(str)
    df["growth_rate"] = df["growth_rate"].astype(str)
    df["target"] = (df["is_legendary"].astype(bool) | df["is_mythical"].astype(bool)).astype(int)

    # Defensivo: nenhuma das colunas do trio possui valores ausentes na base
    # (verificado), mas mantemos o dropna para não propagar NaN silenciosamente
    # caso uma versão futura do dataset introduza lacunas.
    return df.dropna(subset=["base_stat_total", "type_1", "growth_rate", "target"]).copy()


def load_data(seed: int = 42, test_size: float = 0.20):
    """Carrega o dataset e retorna treino/teste com as colunas essenciais.

    Características selecionadas:
        X1 - base_stat_total (contínua): soma dos 6 status base do Pokémon.
        X2 - type_1 (categórica, 18 valores): tipo primário.
        X3 - growth_rate (categórica, 6 valores): taxa de ganho de XP.

    Alvo Y: 1 se o Pokémon é Lendário OU Mítico (is_legendary | is_mythical),
    0 caso contrário.
    """
    df_clean = _load_clean_dataframe()
    X = df_clean[["base_stat_total", "type_1", "growth_rate"]].reset_index(drop=True)
    y = df_clean["target"].astype(int).reset_index(drop=True)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_size, seed=seed)
    return X_train, X_test, y_train, y_test


def load_data_with_names(seed: int = 42, test_size: float = 0.20):
    """Mesmo split de load_data(), mas preservando a coluna 'name' - usada
    para apontar quais Pokémon caem em cada exemplo/erro (análises
    univariadas e interpretação de falsos positivos/negativos)."""
    df_clean = _load_clean_dataframe()
    X = df_clean[["name", "base_stat_total", "type_1", "growth_rate"]].reset_index(drop=True)
    y = df_clean["target"].astype(int).reset_index(drop=True)

    return stratified_split(X, y, test_size=test_size, seed=seed)


__all__ = ["get_csv_path", "load_data", "load_data_with_names", "stratified_split"]
