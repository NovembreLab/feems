from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import networkx as nx
import numpy as np
import msprime


def setup_graph(
    n_rows=8,
    n_columns=12,
    barrier_startpt=2.5,
    barrier_endpt=8.5,
    anisotropy_scaler=1.0,
    barrier_w=0.1,
    corridor_w=0.5,
    n_samples_per_node=10,
    barrier_prob=0.1,
    corridor_left_prob=1,
    corridor_right_prob=1,
    sample_prob=1.0,
):
    """Setup graph (triangular lattice) for simulation

    Arguments
    ---------
    n_rows : int
        number of rows in the lattice

    n_columns : int
        number of rows in the lattice

    barrier_startpt : float
        geographic position of the starting pt of the barrier from left to right

    barrier_endpt : float
        geographic position of the starting pt of the barrier from left to right

    anisotropy_scaler : float
        scaler on horizontal edge weights to create an anisotropic simulation

    barrier_w : float
        migration value for nodes in the barrier

    corridor_w : float
        migration value for nodes in the corridor

    n_samples_per_node : int
        total number of samples in an node

    barrier_prob : float
        probability of sampling an individual in the barrier

    corridor_left_prob : float
        probability of sampling a individual in the left corridor

    corridor_right_prob : float
        probability of sampling an individual in the right corridor

    sample_prob : float
        probability of sampling a node

    Returns
    -------
    tuple of graph objects
    """
    graph = nx.generators.lattice.triangular_lattice_graph(
        n_rows - 1, 2 * n_columns - 2, with_positions=True
    )
    graph = nx.convert_node_labels_to_integers(graph)
    pos_dict = nx.get_node_attributes(graph, "pos")
    for i in graph.nodes:

        # node position
        x, y = graph.nodes[i]["pos"]

        if x <= barrier_startpt:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, corridor_left_prob
            )
        elif x >= barrier_endpt:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, corridor_right_prob
            )
        else:
            graph.nodes[i]["sample_size"] = 2 * np.random.binomial(
                n_samples_per_node, barrier_prob
            )

        # sample a node or not
        graph.nodes[i]["sample_size"] = graph.nodes[i][
            "sample_size"
        ] * np.random.binomial(1, sample_prob)

    # assign edge weights
    for i, j in graph.edges():
        x = np.mean([graph.nodes[i]["pos"][0], graph.nodes[j]["pos"][0]])
        y = np.mean([graph.nodes[i]["pos"][1], graph.nodes[j]["pos"][1]])
        if x <= barrier_startpt:
            graph[i][j]["w"] = corridor_w
        elif x >= barrier_endpt:
            graph[i][j]["w"] = corridor_w
        else:
            graph[i][j]["w"] = barrier_w

        # if horizontal edge
        if graph.nodes[i]["pos"][1] == graph.nodes[j]["pos"][1]:
            graph[i][j]["w"] = anisotropy_scaler * graph[i][j]["w"]

    grid = np.array(list(pos_dict.values()))
    edge = np.array(graph.edges)
    edge += 1  # 1 indexed nodes for feems

    # create sample coordinates array
    sample_sizes_dict = nx.get_node_attributes(graph, "sample_size")
    pops = [[i] * int(sample_sizes_dict[i] / 2) for i in graph.nodes]
    pops = list(it.chain.from_iterable(pops))
    coord = grid[pops, :]
    return (graph, coord, grid, edge)


def simulate_genotypes(
    graph, chrom_length=1, mu=1e-3, n_e=1, target_n_snps=1000, n_print=50
):
    """Simulates genotypes under the stepping-stone model with a habitat specified by the graph

    Arguments
    ---------
    graph : Graph
        networkx graph object

    chrom_length : float
        length of the chromosome to simulate from

    mu : float
        mutation rate

    n_e : float
        effective population size

    target_n_snps : int
        target number of variants

    n_print : int
        interval to print simulation updates for each rep

    Returns
    -------
    genotypes : 2D ndarray
        genotype matrix
    """
    assert target_n_snps > n_print, "n_rep must be greater than n_print"

    # number of nodes
    d = len(graph.nodes)

    # sample sizes per node
    sample_sizes = list(nx.get_node_attributes(graph, "sample_size").values())

    # population config
    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[i]) for i in range(d)
    ]

    # tree sequences
    ts = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=nx.adj_matrix(graph, weight="w").toarray().tolist(),
        length=chrom_length,
        mutation_rate=mu,
        num_replicates=target_n_snps,
        Ne=n_e,
    )

    # simulate haplotypes
    haplotypes = []
    for i, tree_sequence in enumerate(ts):

        # extract haps from ts
        H = tree_sequence.genotype_matrix()
        p, n = H.shape

        # select a random marker per linked replicate
        if p == 0:
            continue
        else:
            idx = np.random.choice(np.arange(p), 1)
            h = H[idx, :]

        haplotypes.append(h)

        if i % n_print == 0:
            print("Simulating ~SNP {}".format(i))

    # stack haplotypes over replicates
    H = np.vstack(haplotypes)

    # convert to genotype matrix: s/o to @aabiddanda
    genotypes = H[:, ::2] + H[:, 1::2]
    return genotypes.T
