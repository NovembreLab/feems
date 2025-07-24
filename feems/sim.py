from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import networkx as nx
import numpy as np
import msprime
import matplotlib.pyplot as plt

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
    option=0,
    ss=None,
):
    """Setup graph (triangular lattice) for simulation

    Arguments
    ---------
    n_rows : int
        number of rows in the lattice

    n_columns : int
        number of columns in the lattice

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
    graph = nx.generators.triangular_lattice_graph(
        n_rows - 1, 2 * n_columns - 2, with_positions=True
    )
    graph = nx.convert_node_labels_to_integers(graph)
    pos_dict = nx.get_node_attributes(graph, "pos")
    for i in graph.nodes:

        # node position
        x, y = graph.nodes[i]["pos"]

        if option == 0:
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
        ## changing the simulation code to take in exact value of individuals (as proportions)
        elif option == 1:
            if x <= barrier_startpt:
                graph.nodes[i]["sample_size"] = int(2 * n_samples_per_node * corridor_left_prob)
            elif x >= barrier_endpt:
                graph.nodes[i]["sample_size"] = int(2 * n_samples_per_node * corridor_right_prob)
            else:
                graph.nodes[i]["sample_size"] = int(2 * n_samples_per_node * barrier_prob)
        ## setting sample sizes explicitly for each node separately
        else:
            graph.nodes[i]["sample_size"] = ss[i]

              
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

# replicate above function to simulate graph with one long range migration edge
# (this could be the same rectangular structure as above with connections from a corridor node to a barrier node)
def setup_graph_long_range(
    n_rows=4,
    n_columns=8,
    barrier_startpt=2.5,
    barrier_endpt=5.5,
    anisotropy_scaler=1.0,
    barrier_w=0.1,
    corridor_w=0.5,
    n_samples_per_node=10,
    barrier_prob=0.1,
    corridor_left_prob=1,
    corridor_right_prob=1,
    sample_prob=1.0,
    long_range_nodes=[(0,12)],
    long_range_edges=[0.5]
):
    """Setup graph (triangular lattice) for simulation + edges with long range migration

    Arguments
    ---------
    n_rows : int
        number of rows in the lattice

    n_columns : int
        number of columns in the lattice

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

    long_range_nodes : integer list
        list of 2-tuples -- nodes connected by long range migrations

    long_range_edges : float list
        corresponding edge weights (migration rates) between the nodes

    Returns
    -------
    tuple of graph objects
    """
    assert type(long_range_nodes) == list, "long_range_nodes must be a list of 2-tuples"
    assert len(long_range_edges) == len(long_range_nodes), "unequal number of pairs of nodes and corresponding edges for long range"

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

        # sample a node or not (leave long_range_nodes as is, hard to pick another set)
        if i not in np.ravel(long_range_nodes):
            graph.nodes[i]["sample_size"] = graph.nodes[i]["sample_size"] * np.random.binomial(1, sample_prob) 

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

    # adding edge weights (migration rates) to corresponding edges
    for idx, val in enumerate(long_range_nodes):
        graph.add_edge(*val)
        graph[val[0]][val[1]]["w"] = long_range_edges[idx]

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
    graph, chrom_length=1, mu=1e-3, n_e=1, target_n_snps=1000, n_print=50, asymmetric=False, long_range_nodes=[(0,0)], long_range_edges=[0],
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
    assert len(long_range_edges) == len(long_range_nodes), "unequal number of pairs of nodes and corresponding edges for long range"

    if asymmetric:
        if len(long_range_nodes) < 1:
            raise ValueError('There should be at least one edge for asymmetric migration. ')

    # number of nodes
    d = len(graph.nodes)

    # sample sizes per node
    sample_sizes = list(nx.get_node_attributes(graph, "sample_size").values())

    # population config
    if hasattr(n_e, "__len__"):
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_sizes[i], initial_size=n_e[i]) for i in range(d)
        ]
    else:
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_sizes[i], initial_size=n_e) for i in range(d)
        ]

    if asymmetric:
        migmat = np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist())
        for id, node in enumerate(long_range_nodes):
            migmat[node[1], node[0]] = long_range_edges[id]
            migmat[node[0], node[1]] = 0.
    else:
        migmat = np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist())

    # plt.imshow(migmat,cmap='Greys'); plt.colorbar()

    # tree sequences
    ts = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=migmat,
        length=chrom_length,
        mutation_rate=mu,
        num_replicates=target_n_snps,
        Ne=1,
    )

    # simulate haplotypes
    haplotypes = []
    for i, tree_sequence in enumerate(ts):

        # tree_sequence.dump(f"results/trees/mytesttreelargeNe1s{i}.tree")

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

def simulate_genotypes_w_admixture(
    graph, chrom_length=1, mu=1e-3, n_e=1, target_n_snps=1000, n_print=50, long_range_nodes=[(0,0)], admixture_props=[0], time_of_adm=[1], replic=0, dump=False
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

    long_range_nodes : list of tuples
        nodes connected by admixture event [(source1, destination1), (source2, destination2), ...]

    admixture_props : list of floats
        proportion of admixture events [c1, c2, ...]
    
    time_of_adm : list of floats
        time of admixture events [t1, t2, ...]

    dump : boolean
        whether to dump tree sequences or not (default: False)

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
    if hasattr(n_e, "__len__"):
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_sizes[i], initial_size=n_e[i]) for i in range(d)
        ]
    else:
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=sample_sizes[i], initial_size=n_e) for i in range(d)
        ]

    # if we want to simulate low migration around destination 
    migmat = np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist())
    # migmat[32,33] = 0.001; migmat[33,32] = 0.001
    # migmat[32,44] = 0.001; migmat[44,32] = 0.001
    # migmat[32,43] = 0.001; migmat[43,32] = 0.001
    # migmat[33,44] = 0.001; migmat[44,33] = 0.001
    # migmat[33,45] = 0.001; migmat[45,33] = 0.001
    # migmat[43,44] = 0.001; migmat[44,43] = 0.001
    # migmat[43,56] = 0.001; migmat[56,43] = 0.001
    # migmat[44,45] = 0.001; migmat[45,44] = 0.001
    # migmat[44,56] = 0.001; migmat[56,44] = 0.001
    # migmat[44,57] = 0.001; migmat[57,44] = 0.001
    # migmat[45,57] = 0.001; migmat[57,45] = 0.001
    # migmat[56,57] = 0.001; migmat[57,56] = 0.001
    
    # tree sequences
    ts = msprime.simulate(
        population_configurations=population_configurations,
        migration_matrix=migmat,
        demographic_events=[msprime.MassMigration(time_of_adm[i], source=long_range_nodes[i][1], dest=long_range_nodes[i][0], proportion=admixture_props[i]) for i in range(len(admixture_props))],
        length=chrom_length,
        mutation_rate=mu,
        num_replicates=target_n_snps,
        Ne=1
    )

    # Create a demography object
    # demography = msprime.Demography.from_old_style(population_configurations=population_configurations,
    #                                                migration_matrix=np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist()),
    #                                                ignore_sample_size=True, 
    #                                                Ne=n_e)
    # demography.add_mass_migration(time=time_of_adm[0], source=long_range_nodes[0][1], dest=long_range_nodes[0][0], proportion=admixture_props[0])

    # # Simulate tree sequences
    # ts = msprime.sim_ancestry(
    #     samples=[msprime.SampleSet(sample_sizes[i]//2,i,ploidy=2) for i in range(d)],
    #     demography=demography,
    #     sequence_length=chrom_length,
    #     recombination_rate=0,  # No recombination
    #     record_migrations=False, record_full_arg=False,
    #     # model=[msprime.DiscreteTimeWrightFisher(duration=time_of_adm[0]+1), msprime.StandardCoalescent()],
    #     num_replicates=target_n_snps
    # )

    # if long_range_nodes!=[(0,0)]:
    #     migmat = np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist())
    #     for id, node in enumerate(long_range_nodes):
    #         migmat[node[1], node[0]] = admixture_props[id]
    #         migmat[node[0], node[1]] = 0.
    # else:
    #     migmat = np.array(nx.adjacency_matrix(graph, weight="w").toarray().tolist())
    # plt.imshow(migmat,cmap='Greys'); plt.colorbar()

    # simulate haplotypes
    haplotypes = []
    for i, tree_sequence in enumerate(ts):

        if dump:
            tree_sequence.dump("/Volumes/GoogleDrive/Other computers/My Mac mini/Documents/feemsResults/trees/tree{:d}_Ne{:d}_8x10_c{:.1f}_t{:d}.tree".format(i,n_e,admixture_props[0],time_of_adm[0]))

        # extract haps from ts
        # H = msprime.sim_mutations(tree_sequence, rate=mu, model=msprime.BinaryMutationModel()).genotype_matrix()
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