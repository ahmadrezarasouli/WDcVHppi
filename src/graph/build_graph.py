import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_interaction_graph(df):

    graph = nx.from_pandas_edgelist(
        df,
        source="human_protein",
        target="virus_protein",
        create_using=nx.DiGraph(),
    )

    return graph


def node_colors(df, graph):

    human_nodes = set(df["human_protein"])

    return [
        "#FFB6C1" if node in human_nodes else "#87CEFA"
        for node in graph.nodes()
    ]


def draw_graph(
    graph,
    colors,
    output_file=None,
    figsize=(20, 20),
    dpi=600,
):

    plt.figure(figsize=figsize, dpi=dpi)

    nx.draw(
        graph,
        node_color=colors,
        node_size=150,
        font_size=2,
        with_labels=True,
    )

    if output_file is not None:
        plt.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.close()


def graph_statistics(graph):

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
    }
