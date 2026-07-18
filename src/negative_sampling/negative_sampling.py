import numpy as np
import pandas as pd
from itertools import product
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load


def generate_candidate_pairs(human_proteins, virus_proteins, interactions):

    pairs = pd.DataFrame(
        product(human_proteins, virus_proteins),
        columns=["human_protein", "virus_protein"],
    )

    pairs = pairs.merge(
        interactions,
        on=["human_protein", "virus_protein"],
        how="left",
        indicator=True,
    )

    return (
        pairs[pairs["_merge"] == "left_only"]
        .drop(columns="_merge")
        .reset_index(drop=True)
    )


def calculate_similarity(fasta_file, threshold=0.30):

    sequences = list(SeqIO.parse(fasta_file, "fasta"))

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = load("BLOSUM62")
    aligner.open_gap_score = -15
    aligner.extend_gap_score = -1

    similar_pairs = []

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):

            score = aligner.score(
                str(sequences[i].seq),
                str(sequences[j].seq),
            )

            norm = score / np.sqrt(
                aligner.score(str(sequences[i].seq), str(sequences[i].seq))
                * aligner.score(str(sequences[j].seq), str(sequences[j].seq))
            )

            if norm >= threshold:

                v1 = "uniprotkb:" + sequences[i].id.split("|")[1]
                v2 = "uniprotkb:" + sequences[j].id.split("|")[1]

                similar_pairs.append((v1, v2))

    return pd.DataFrame(
        similar_pairs,
        columns=["Virus_1", "Virus_2"],
    )


def remove_similar_pairs(candidate_pairs, similar_pairs, interactions):

    virus_map = {}

    for _, row in similar_pairs.iterrows():

        virus_map.setdefault(row["Virus_1"], set()).add(row["Virus_2"])
        virus_map.setdefault(row["Virus_2"], set()).add(row["Virus_1"])

    expanded = interactions.copy()

    new_rows = []

    for _, row in interactions.iterrows():

        virus = row["virus_protein"]

        if virus in virus_map:

            for v in virus_map[virus]:

                new_rows.append(
                    {
                        "human_protein": row["human_protein"],
                        "virus_protein": v,
                    }
                )

    expanded = pd.concat(
        [expanded, pd.DataFrame(new_rows)],
        ignore_index=True,
    ).drop_duplicates()

    result = candidate_pairs.merge(
        expanded,
        on=["human_protein", "virus_protein"],
        how="left",
        indicator=True,
    )

    return (
        result[result["_merge"] == "left_only"]
        .drop(columns="_merge")
        .reset_index(drop=True)
    )


def sample_negative_pairs(candidate_pairs, ratio, positive_size):

    n = ratio * positive_size

    idx = np.random.choice(
        len(candidate_pairs),
        size=n,
        replace=True,
    )

    return candidate_pairs.iloc[idx].reset_index(drop=True)
