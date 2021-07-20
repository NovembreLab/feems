from __future__ import absolute_import, division, print_function

import sys

import networkx as nx
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats.distributions import chi2

from .cross_validation import comp_mats, run_cv
from .objective import Objective
from .spatial_graph import SpatialGraph, query_node_attributes
from .utils import prepare_graph_inputs
from .viz import Viz
from .helper_funcs import comp_genetic_vs_fitted_distance, plot_default_vs_long_range


class FeemsMix:
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True, 
                n_lre=0, n_folds=None, 
                search='Hull'):
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
            search (:obj:`str`): type of search to find best fit long range edge -  default is convex hull
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
        assert search in ['Global','Hull','Top-N'], "search should be a string, one of Global/Hull/Top-N"

        # creating a projection vector
        projection = ccrs.EquidistantConic(central_longitude=np.median(sample_pos[:,0]), central_latitude=np.median(sample_pos[:,1]))

        # creating the default graph
        self.graph = []
        self.graph.append(SpatialGraph(genotypes, sample_pos, node_pos, edges))

        # plotting the samples map with overlaid grid
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection=projection)  
        v = Viz(ax, self.graph[0], projection=projection, edge_width=.5, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=10, 
                obs_node_size=7.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_map()
        v.draw_samples()
        v.draw_edges(use_weights=False)
        v.draw_obs_nodes(use_ids=False)
        
        # running the CV step to compute 'best' lambda fit
        lamb_grid = np.geomspace(1e-6, 1e2, 20)[::-1]
        cv_err = run_cv(self.graph[0], lamb_grid, n_folds=n_folds, factr=1e10)
        lamb_cv = float(lamb_grid[np.argmin(np.mean(cv_err, axis=0))])

        print("\n")
        # plotting the genetic vs fitted distance to assess goodness of fit & returning pair of nodes with maximum abs residual
        max_res_nodes = comp_genetic_vs_fitted_distance(self.graph[0], lamb=lamb_cv, plotFig=True, n_lre=1)

        obj = Objective(self.graph[0])
        obj._solve_lap_sys()
        obj._comp_mat_block_inv()
        obj._comp_inv_cov()
        self.nll = list()
        self.nll.append(obj.neg_log_lik())

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection=projection)  
        v = Viz(ax, self.graph[0], projection=projection, edge_width=.5, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
                obs_node_size=7.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_map()
        v.draw_edges(use_weights=True)
        v.draw_obs_nodes(use_ids=False) 
        v.draw_edge_colorbar()

        # TODO: plot the lower triangular residual matrix here

        if n_lre is None:
            # TODO: fill in code here for running the case when we stop based on a condition...
            1+1 # placeholder
        elif n_lre>0:
            # creating a copy of the edges of default graph to set us off
            temp_edges = deepcopy(edges.tolist())
            for n in np.arange(1,n_lre+1):
                # need to select based on a flag here...
                temp_edges.append(list(x+1 for x in max_res_nodes[0]))

                self.graph.append(SpatialGraph(genotypes, sample_pos, node_pos, np.array(temp_edges)))

                lamb_grid = np.geomspace(1e-6, 1e2, 20)[::-1]
                cv_err = run_cv(self.graph[n], lamb_grid, n_folds=n_folds, factr=1e10)
                lamb_cv_lr = float(lamb_grid[np.argmin(np.mean(cv_err, axis=0))])

                # this is where the search functions will go - graph already created, has one max_res_node
                best_fit_nodes = _search_hull(n, max_res_nodes, lamb_cv_lr)
                
                # create a vector of nll fits for best long range edge
                self.nll.append(best_fit_nodes.loc[1,'nll'])

                # replace the max_res_node edge with the best fit
                if(max_res_nodes[0]!=best_fit_nodes.loc[1,'nodes']):
                    self.graph[n].remove_edge(*best_fit_nodes.loc[0,'nodes'])
                    self.graph[n].add_edge(*best_fit_nodes.loc[1,'nodes'])
                    temp_edges[temp_edges.index(list(x+1 for x in max_res_nodes[0]))] = list(x+1 for x in best_fit_nodes.loc[1,'nodes'])

                # TODO: do nll p-value calc here and output more informative message
                if(self.nll[n] < self.nll[0]):
                    print("Model with long-range edges fits better than default by %.2f log units with p-value of %.2e"%(2.*(self.nll[0] - self.nll[n]),chi2.sf(2.*(self.nll[0] - self.nll[n]),n)))
                else: 
                    print("Default model fits better than model with long range edges")

                max_res_nodes = comp_genetic_vs_fitted_distance(self.graph[n], lamb=lamb_cv, plotFig=False, n_lre=1)
            
            # plot the final graph against the default (do row-wise potentially -- better for bigger graphs)
            self.lre = list(set([tuple((x[0]-1,x[1]-1)) for x in temp_edges]) - set(list(self.graph[0].edges)))
            plot_default_vs_long_range(self.graph[0], self.graph[n_lre], max_res_nodes=self.lre, lamb=np.array((lamb_cv,lamb_cv_lr)))

    def _search_global(n):
        # TODO: put a progress bar
        lr = (tuple(i) for i in it.product(tuple(range(self.graph[n].n_observed_nodes)), repeat=2) if tuple(reversed(i)) > tuple(i))
        final_lr = [x for x in list(lr) if x not in list(self.graph[n].edges)]

        df = pd.DataFrame(index = np.arange(len(final_lr)), columns = ['nodes', 'nll'])

        df['nodes'] = final_lr
        df['nll'] = list(map(add_edge_get_nll, df.iloc[np.arange(len(final_lr)),0]))

        # print nodes connected by THE edge to give lowest negative log likelihood
        return(df.loc[df['nll'].astype(float).idxmin(),'nodes'])

    def _search_hull(n, max_res_nodes):
        # TODO: put a progress bar
        spl = dict(nx.all_pairs_shortest_path_length(self.graph[n],cutoff=4))

        # get closest (within distance 3) AND sampled nodes to create a set of nodes to search over
        n1 = [k for (k, v) in spl[max_res_nodes[0][0]].items() if v>0 and v<4 and k in np.array(np.where(query_node_attributes(self.graph[n],"n_samples")>0))]
        n2 = [k for (k, v) in spl[max_res_nodes[0][1]].items() if v>0 and v<4 and k in np.array(np.where(query_node_attributes(self.graph[n],"n_samples")>0))]

        n1.append(max_res_nodes[0][0])
        n2.append(max_res_nodes[0][1])

        lr_hull = (tuple(i) for i in it.product(n1, n2))
        # removing nodes that are already connected in the default graph 
        final_lr_hull = [x for x in list(lr_hull) if x not in list(self.graph[n].edges)]

        df_hull = pd.DataFrame(index = np.arange(len(final_lr_hull)), columns = ['nodes', 'nll'])

        df_hull['nodes'] = final_lr_hull
        obj = Objective(self.graph[n])
        obj._solve_lap_sys()
        obj._comp_mat_block_inv()
        obj._comp_inv_cov()
        df_hull.iloc[len(df_hull), 1] = obj.neg_log_lik()
        for idx in np.arange(0,len(df_hull)-1)[::-1]:
            df_hull.iloc[idx, 1] = add_edge_get_nll(n, df_hull.iloc[idx+1, 0], df_hull.iloc[idx, 0], lamb_cv_lr)
            
        # print nodes connected by THE edge to give lowest negative log likelihood
        return(df_hull.loc[(0,df_hull['nll'].astype(float).idxmin()),:])
    
    # function to add an edge and return the negative log-likelihood value
    def _add_edge_get_nll(n, mrn, new_mrn, lamb):
        self.graph[n].remove_edge(*mrn)
        self.graph[n].add_edge(*new_mrn)
        
        self.graph[n].Delta_q = nx.incidence_matrix(self.graph[1], oriented=True).T.tocsc()
        self.graph[n].adj_base = sp.triu(nx.adjacency_matrix(self.graph[n]), k=1)
        self.graph[n].nnz_idx = self.graph[n].adj_base.nonzero()
        self.graph[n].Delta = self.graph[n]._create_incidence_matrix()
        self.graph[n].diag_oper = self.graph[n]._create_vect_matrix()
        self.graph[n].w = np.ones(self.graph[n].size())
        self.graph[n].comp_grad_w()

        self.graph[n].fit(lamb = float(lamb), verbose=False)
        obj = Objective(self.graph[n])
        obj._solve_lap_sys()
        obj._comp_mat_block_inv()
        obj._comp_inv_cov()
        return obj.neg_log_lik()

    def _add_edge(edges, mrn):
        edges_lr = edges.tolist()
        edges_lr.append(list(x+1 for x in mrn))
        return(edges_lr)
