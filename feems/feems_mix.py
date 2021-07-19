from __future__ import absolute_import, division, print_function

import sys

import networkx as nx
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from copy import deepcopy

from .cross_validation import comp_mats, run_cv
from .objective import Objective
from .spatial_graph import SpatialGraph, query_node_attributes
from .utils import prepare_graph_inputs
from .viz import Viz
from .helper_funcs import comp_genetic_vs_fitted_distance, plot_default_vs_long_range


class FeemsMix(object):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True, 
                n_lre=0, n_folds=None):
        """Represents the meta-list of spatial graphs which the data is defined on and
        performs relevant computations and operations to help choose the long range edges. 

        Args:
            genotypes (:obj:`numpy.ndarray`): genotypes for samples
            sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
            scale_snps (:obj:`Bool`): boolean to scale SNPs by SNP specific
                Binomial variance estimates
            n_lre (:obj:`int`): number of long range edges to add
            n_folds (:obj:`int`): number of folds to run CV over - default is leave-one-out
        """
        # check inputs
        assert len(genotypes.shape) == 2
        assert len(sample_pos.shape) == 2
        assert np.all(~np.isnan(genotypes)), "no missing genotypes are allowed"
        assert np.all(~np.isinf(genotypes)), "non inf genotypes are allowed"
        assert (
            genotypes.shape[0] == sample_pos.shape[0]
        ), "genotypes and sample positions must be the same size"
        assert type(n_lre)==int or n_lre is None, "n_lre should be an integer"

        # creating a projection vector
        projection = ccrs.EquidistantConic(central_longitude=np.median(sample_pos[:,0]), central_latitude=np.median(sample_pos[:,1]))

        # creating the default graph
        self.def_graph = SpatialGraph(genotypes, sample_pos, node_pos, edges)

        # plotting the samples map with overlaid grid
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=projection)  
        v = Viz(ax, self.def_graph, projection=projection, edge_width=.5, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=10, 
                obs_node_size=7.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_map()
        v.draw_samples()
        v.draw_edges(use_weights=False)
        v.draw_obs_nodes(use_ids=False)
        
        # running the CV step to compute 'best' lambda fit
        lamb_grid = np.geomspace(1e-6, 1e2, 20)[::-1]
        cv_err = run_cv(self.def_graph, lamb_grid, n_folds=n_folds, factr=1e10)
        lamb_cv = float(lamb_grid[np.argmin(np.mean(cv_err, axis=0))])

        print("\n")
        # plotting the genetic vs fitted distance to assess goodness of fit & returning pair of nodes with maximum abs residual
        max_res_nodes = comp_genetic_vs_fitted_distance(self.def_graph, lamb=lamb_cv, plotFig=True, n_lre=1)

        obj = Objective(self.def_graph)
        obj._solve_lap_sys()
        obj._comp_mat_block_inv()
        obj._comp_inv_cov()
        self.def_nll = obj.neg_log_lik()

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=projection)  
        v = Viz(ax, self.def_graph, projection=projection, edge_width=.5, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
                obs_node_size=7.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_map()
        v.draw_edges(use_weights=True)
        v.draw_obs_nodes(use_ids=False) 
        v.draw_edge_colorbar()

        # plot the lower triangular residual matrix here

        if n_lre is None:
            self.graph = list()
            # fill in code here for running the case when we stop based on a condition...
        elif n_lre>0:
            self.graph = list()
            self.nll_lr = list()

            # creating a copy of the edges of default graph to set us off
            temp_edges = deepcopy(edges.tolist())
            for n in np.arange(n_lre):
                # need to select based on a flag here...
                temp_edges.append(list(x+1 for x in max_res_nodes[0]))

                self.graph.append(SpatialGraph(genotypes, sample_pos, node_pos, np.array(temp_edges)))
                lamb_grid = np.geomspace(1e-6, 1e2, 20)[::-1]
                cv_err = run_cv(self.graph[n], lamb_grid, n_folds=n_folds, factr=1e10)
                lamb_cv_lr = float(lamb_grid[np.argmin(np.mean(cv_err, axis=0))])

                obj = Objective(self.graph[n])
                obj._solve_lap_sys()
                obj._comp_mat_block_inv()
                obj._comp_inv_cov()
                self.nll_lr.append(obj.neg_log_lik())

                # do nll p-value calc here and output more informative message
                if(self.nll_lr[n] < self.def_nll):
                    print("Model with long-range edges fits better than default by %.2f log units"%(-2*(self.nll_lr[n] - self.def_nll)))

                max_res_nodes = comp_genetic_vs_fitted_distance(self.graph[n], lamb=lamb_cv, plotFig=False, n_lre=1)
            
            # plot the final graph against the default (do row-wise potentially -- better for bigger graphs)
            plot_default_vs_long_range(self.def_graph, self.graph[n], max_res_nodes=temp_edges, lamb=np.array((lamb_cv,lamb_cv_lr)))



    def _add_edge(edges, mrn):
        edges_lr = edges.tolist()
        edges_lr.append(list(x+1 for x in mrn))
        return(edges_lr)
    
    # function to add an edge and return the negative log-likelihood value
    def _add_edge_get_nll(edges, mrn, lamb):
        edges_lr = deepcopy(edge)
        edges_lr = edges_lr.tolist()
        edges_lr.append(list(x+1 for x in mrn))
        sp_Graph = SpatialGraph(gen_test, coord, grid, np.array(edges_lr))
        sp_Graph.fit(lamb = float(lamb), verbose=False)
        obj = Objective(sp_Graph)
        obj._solve_lap_sys()
        obj._comp_mat_block_inv()
        obj._comp_inv_cov()
        return obj.neg_log_lik()
