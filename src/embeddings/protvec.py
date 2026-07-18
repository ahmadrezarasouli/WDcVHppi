import csv

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_protvec(path):
    embeddings = []

    with open(path) as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            embeddings.append(row[0].split("\t"))

    kmers = [row[0] for row in embeddings]

    embedding_matrix = np.array(
        [[float(x) for x in row[1:]] for row in embeddings]
    )

    kmer_dict = {kmer: i for i, kmer in enumerate(kmers)}

    return kmer_dict, embedding_matrix


def kmer_lists(sequence):

    k0 = []
    k1 = []
    k2 = []

    for i in range(len(sequence) - 2):

        if i % 3 == 0:
            k0.append(sequence[i:i + 3])

        elif i % 3 == 1:
            k1.append(sequence[i:i + 3])

        else:
            k2.append(sequence[i:i + 3])

    return k0 + k1 + k2


def sequence_to_vector(sequence, kmer_dict, embedding_matrix):

    vector = np.zeros(100)

    for kmer in kmer_lists(sequence):

        index = kmer_dict.get(kmer, kmer_dict["<unk>"])

        vector += embedding_matrix[index]

    return vector


def sequences_to_dataframe(
    sequences,
    kmer_dict,
    embedding_matrix,
    n_jobs=-1
):

    vectors = Parallel(n_jobs=n_jobs)(
        delayed(sequence_to_vector)(
            seq,
            kmer_dict,
            embedding_matrix
        )
        for seq in sequences
    )

    return pd.DataFrame(vectors)
