from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
import pkg_resources
from feems import Objective, SpatialGraph
from feems.utils import prepare_graph_inputs
from pandas_plink import read_plink
from sklearn.impute import SimpleImputer


"""
TODO: This will be filled out with various matrix operations computed
using dense matrices and slow direct approaches for computing inverses etc.
"""


class TestObjective(unittest.TestCase):
    """Tests for the feems Objective
    """
    # path to example data
    data_path = pkg_resources.resource_filename("feems", "data/")

    # read the genotype data and mean impute missing data
    (bim, fam, G) = read_plink("{}/wolvesadmix".format(data_path))
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    genotypes = imp.fit_transform((np.array(G)).T)

    # setup graph
    coord = np.loadtxt("{}/wolvesadmix.coord".format(data_path))
    outer = np.loadtxt("{}/wolvesadmix.outer".format(data_path))
    grid_path = "{}/grid_250.shp".format(data_path)
    outer, edges, grid, ipmap = prepare_graph_inputs(coord=coord,
                                                     ggrid=grid_path,
                                                     translated=True,
                                                     buffer=0,
                                                     outer=outer)
    sp_graph = SpatialGraph(genotypes, coord, grid, edges)
    obj = Objective(sp_graph)

    def test_n_observed_nodes(self):
        """Tests the right number of observed nodes
        """
        self.assertEqual(self.sp_graph.n_observed_nodes, 78)


def dense_neg_log_lik():
    """TODO: fill in function for computation of negative log-likelihood
    using dense matrix operations
    """
    pass


if __name__ == '__main__':
    unittest.main()
