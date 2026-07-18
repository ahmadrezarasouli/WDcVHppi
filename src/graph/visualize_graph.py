import matplotlib.pyplot as plt
import networkx as nx


def build_graph(
    dataframe,
    source,
    target,
    directed=True,
):

    if directed:
        graph = nx.from_pandas_edgelist(
            dataframe,
            source=source,
            target=target,
            create_using=nx.DiGraph(),
        )
    else:
        graph = nx.from_pandas_edgelist(
            dataframe,
            source=source,
            target=target,
        )

    return graph


def draw_graph(
    graph,
    node_size=80,
    font_size=6,
    figsize=(12, 12),
    save_path=None,
):

    plt.figure(figsize=figsize)

    nx.draw(
        graph,
        with_labels=True,
        node_size=node_size,
        font_size=font_size,
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def draw_bipartite_graph(
    dataframe,
    source,
    target,
    save_path=None,
):

    graph = nx.from_pandas_edgelist(
        dataframe,
        source=source,
        target=target,
        create_using=nx.DiGraph(),
    )

    node_colors = []

    source_nodes = set(dataframe[source])

    for node in graph.nodes():

        if node in source_nodes:
            node_colors.append("#FFB6C1")
        else:
            node_colors.append("#87CEFA")

    plt.figure(figsize=(16, 16))

    nx.draw(
        graph,
        with_labels=True,
        node_color=node_colors,
        node_size=100,
        font_size=5,
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
