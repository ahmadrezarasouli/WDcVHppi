import networkx as nx
import pandas as pd
from node2vec import Node2Vec


def load_graph(edge_file):
    edges = pd.read_csv(edge_file, sep="\t")

    edges.columns = [
        "source",
        "interaction",
        "target",
    ]

    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    return graph


def train_node2vec(
    graph,
    dimensions=100,
    walk_length=30,
    num_walks=200,
    workers=4,
):

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
    )

    return node2vec.fit()


def get_embeddings(graph, model):

    embeddings = []

    for node in graph.nodes():
        embeddings.append(model.wv[str(node)])

    return pd.DataFrame(
        embeddings,
        index=list(graph.nodes()),
    )


def save_embeddings(embedding_df, output_file):

    embedding_df.to_excel(output_file)
