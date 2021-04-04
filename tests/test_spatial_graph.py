from __future__ import absolute_import, division, print_function

import unittest

import networkx as nx
import numpy as np
from feems import SpatialGraph, query_node_attributes


class TestSpatialGraph(unittest.TestCase):
    """Tests for the feems SpatialGraph
    """
    # generate a simple networkx graph with 4 nodes and 4 samples observed on
    # only 2 of the 4 nodes
    graph = nx.triangular_lattice_graph(1, 2, with_positions=True)
    graph = nx.convert_node_labels_to_integers(graph)
    pos = nx.get_node_attributes(graph, "pos")
    node_pos = np.array(list(pos.values()))
    sample_pos = np.empty((4, 2))
    sample_pos[0, :] = node_pos[0, ]
    sample_pos[1, :] = node_pos[0, ]
    sample_pos[2, :] = node_pos[2, ]
    sample_pos[3, :] = node_pos[2, ]
    edges = np.array(list(graph.edges)) + 1
    n_snps = 100
    genotypes = np.random.binomial(n=2, p=.5,
                                   size=(sample_pos.shape[0], n_snps))

    # setup the spatial graph
    sp_graph = SpatialGraph(genotypes, sample_pos, node_pos, edges)

    def test_idx(self):
        """Tests original node indicies
        """
        idx = query_node_attributes(self.sp_graph, "idx")
        self.assertEqual(idx.tolist(), [0, 1, 2, 3])

    def test_node_pos(self):
        """Tests for the correct node positions
        """
        pos = query_node_attributes(self.sp_graph, "pos")
        self.assertEqual(pos.tolist(), self.node_pos.tolist())

    def test_sample_idx(self):
        """Tests assignment of samples to nodes
        """
        sample_idx_dict = nx.get_node_attributes(self.sp_graph, "sample_idx")
        self.assertEqual(sample_idx_dict[0], [0, 1])
        self.assertEqual(sample_idx_dict[1], [])
        self.assertEqual(sample_idx_dict[2], [2, 3])
        self.assertEqual(sample_idx_dict[3], [])

    def test_permuted_idx(self):
        """Tests permutation of nodes
        """
        permuted_idx = query_node_attributes(self.sp_graph, "permuted_idx")
        self.assertEqual(permuted_idx.tolist(), [0, 2, 1, 3])

    def test_n_samples_per_obs_node_permuted(self):
        """Tests permutation of n samples
        """
        ns = self.sp_graph.n_samples_per_obs_node_permuted
        self.assertEqual(ns.tolist(), [2, 2])

    def test_n_observed_nodes(self):
        """Tests the right number of observed nodes
        """
        self.assertEqual(self.sp_graph.n_observed_nodes, 2)

    def test_estimate_allele_frequencies(self):
        """Tests allele frequency estimation on the observed nodes of the
        graph
        """
        node_0_freqs = np.mean(self.genotypes[[0, 1], :], axis=0)  # / 4.0
        node_2_freqs = np.mean(self.genotypes[[2, 3], :], axis=0)  # / 4.0
        exp_freqs = np.vstack([node_0_freqs, node_2_freqs])
        self.assertEqual(self.sp_graph.frequencies.tolist(),
                         exp_freqs.tolist())


if __name__ == '__main__':
    unittest.main()
