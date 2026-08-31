import glob
import os
import shutil
from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd


DATA_DIR = "data"
TARGET_CSV = os.path.join(DATA_DIR, "pokemon_go.csv")


def get_csv_path() -> str:
    """Baixa o dataset do Kaggle ou reutiliza uma cópia local."""
    if os.path.exists(TARGET_CSV):
        return TARGET_CSV

    print("[data_loader] Baixando dataset do Kaggle via kagglehub...")
    dataset_path = kagglehub.dataset_download("shreyasur965/pokemon-go")

    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV foi encontrado no dataset baixado.")

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


def load_data(seed: int = 42, test_size: float = 0.20):
    """Carrega o dataset e retorna treino/teste com as colunas essenciais."""
    filepath = get_csv_path()
    df = pd.read_csv(filepath)

    required_columns = ["rarity", "base_attack", "base_capture_rate", "type"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes no dataset: {missing}")

    df = df.loc[:, required_columns].copy()
    df["base_attack"] = pd.to_numeric(df["base_attack"], errors="coerce")
    df["base_capture_rate"] = pd.to_numeric(df["base_capture_rate"], errors="coerce")
    df["type"] = df["type"].fillna("Unknown").astype(str)
    df["target"] = df["rarity"].map(lambda value: 1 if str(value).lower() in {"legendary", "mythic"} else 0)

    df_clean = df.dropna(subset=["base_attack", "base_capture_rate", "type", "target"]).copy()
    X = df_clean[["base_attack", "base_capture_rate", "type"]].reset_index(drop=True)
    y = df_clean["target"].astype(int).reset_index(drop=True)

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=test_size, seed=seed)
    return X_train, X_test, y_train, y_test


__all__ = ["get_csv_path", "load_data", "stratified_split"]