import pandas as pd


def select_columns(df):
    selected = df.iloc[:, [0, 1, 9, 10, 14]].copy()

    selected.columns = [
        "human_protein",
        "virus_protein",
        "taxid_host",
        "taxid_virus",
        "virhostnet_miscore",
    ]

    return selected


def filter_human(df):
    return df[df["taxid_host"] == "taxid:9606"].reset_index(drop=True)


def filter_min_interactions(df, minimum=100):
    counts = df["taxid_virus"].value_counts()

    valid_taxids = counts[counts >= minimum].index

    return df[df["taxid_virus"].isin(valid_taxids)].reset_index(drop=True)


def extract_taxids(df):
    return (
        df["taxid_virus"]
        .str.replace("taxid:", "", regex=False)
        .astype(int)
        .unique()
    )


def create_binary_label(df, target_taxid):
    df = df.copy()

    df["label"] = (
        df["taxid_virus"] == f"taxid:{target_taxid}"
    ).astype(int)

    return df
