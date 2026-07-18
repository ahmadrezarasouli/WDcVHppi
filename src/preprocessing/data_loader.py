import pandas as pd


def load_excel(file_path, sheet_name=0):
    return pd.read_excel(file_path, sheet_name=sheet_name)


def load_csv(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t", header=None)


def save_excel(df, file_path):
    df.to_excel(file_path, index=False)


def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
