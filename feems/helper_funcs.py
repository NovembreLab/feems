""" This file contains functions to be used in miscellaneous tasks like comparing simulated to estimated results, etc
"""
# base
import math
import numpy as np
import statsmodels.api as sm
from copy import deepcopy
import pandas as pd

# viz
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

# feems
from .utils import prepare_graph_inputs
from .sim import setup_graph, setup_graph_long_range, simulate_genotypes
from .spatial_graph import SpatialGraph, query_node_attributes
from .cross_validation import comp_mats, run_cv
from .objective import Objective
from .viz import Viz

# change matplotlib fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = "Arial"

def cov_to_dist(S):
    """Convert a covariance matrix to a distance matrix
    """
    s2 = np.diag(S).reshape(-1, 1)
    ones = np.ones((s2.shape[0], 1))
    D = s2 @ ones.T + ones @ s2.T - 2 * S
    return(D)

def plot_default_vs_long_range(
    sp_Graph_def, 
    sp_Graph, 
    max_res_nodes=None, 
    lamb=np.array((1.0,1.0))
):
    """Function to plot default graph with NO long range edges next to full graph with long range edges
    (useful for comparison of feems default fit with extra parameters)
    """
    assert all(lamb>=0.0), "lamb must be non-negative"
    assert type(max_res_nodes) == list, "max_res_nodes must be a list of int 2-tuples"

    fig = plt.figure(dpi=100)
    #sp_Graph_def.fit(lamb = float(lamb[0]))
    ax = fig.add_subplot(1, 2, 1)  
    v = Viz(ax, sp_Graph_def, edge_width=2, 
            edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
            obs_node_size=7.5, sample_pt_color="black", 
            cbar_font_size=10)
    v.draw_edges(use_weights=True)
    v.draw_obs_nodes(use_ids=False) 

    #sp_Graph.fit(lamb = float(lamb[1]))
    ax = fig.add_subplot(1, 2, 2)  
    v = Viz(ax, sp_Graph, edge_width=2.0, 
            edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
            obs_node_size=7.5, sample_pt_color="black", 
            cbar_font_size=10)
    v.draw_edges(use_weights=True)
    v.draw_obs_nodes(use_ids=False) 
    lre_idx = [list(sp_Graph.edges).index(val) for val in max_res_nodes]
    # paste correlation between the two weights
    ax.text(0.5, 1.0, "cor={:.2f}".format(np.corrcoef(sp_Graph.w[~np.in1d(np.arange(len(sp_Graph.w)), lre_idx)],sp_Graph_def.w)[0,1]), transform=ax.transAxes)

    return(None)

def comp_genetic_vs_fitted_distance(
    sp_Graph_def, 
    lrn=None, 
    lamb=None, 
    n_lre=3, 
    plotFig=True
):
    """Function to plot genetic vs fitted distance to visualize outliers in residual calculations, 
    passes back 3 pairs of nodes (default) with largest residuals if plotFig=False
    """
    if lamb is not None:
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float, "lambda must be float"
    assert type(n_lre) == int, "n_lre must be int"

    tril_idx = np.tril_indices(sp_Graph_def.n_observed_nodes, k=-1)
    
    if lamb is None:
        lamb_grid = np.geomspace(1e-6, 1e2, 20)[::-1]
        cv_err = run_cv(sp_Graph_def, lamb_grid, n_folds=sp_Graph_def.n_observed_nodes, factr=1e10)

        # average over folds
        mean_cv_err = np.mean(cv_err, axis=0)

        # argmin of cv error
        lamb = float(lamb_grid[np.argmin(mean_cv_err)])

    sp_Graph_def.fit(lamb=lamb,
                    lb=math.log(1e-6), 
                    ub=math.log(1e+6))
    sp_Graph_def.comp_graph_laplacian(sp_Graph_def.w)

    obj = Objective(sp_Graph_def)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[tril_idx]
    emp_dist = cov_to_dist(emp_cov)[tril_idx]

    # using code from supp fig 6 of feems-analysis
    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist, X)
    res = mod.fit()
    muhat, betahat = res.params
    if(plotFig):
        if lrn is not None:
            # computing the vector index for lower triangular matrix of long range nodes (i+j(j+1)/2-j for lower triangle)
            lrn_idx = [np.int(val[0] + 0.5*val[1]*(val[1]+1) - val[1]) if val[0]<val[1] else np.int(val[1] + 0.5*val[0]*(val[0]+1) - val[0]) for val in lrn]

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot()
        ax.scatter(fit_dist, emp_dist, 
                marker=".", alpha=1, zorder=0, color="grey", s=3)
        if lrn is not None:
            ax.scatter(fit_dist[lrn_idx], emp_dist[lrn_idx], 
                    marker=".", alpha=1, zorder=0, color="black", s=10)
        x_ = np.linspace(np.min(fit_dist), np.max(fit_dist), 20)
        ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
        ax.text(0.8, 0.15, "$\lambda$={:.3}".format(lamb), transform=ax.transAxes)
        ax.text(0.8, 0.05, "R²={:.4f}".format(res.rsquared), transform=ax.transAxes)
        ax.set_ylabel("genetic distance")
        ax.set_xlabel("fitted distance")

        # extract indices with maximum absolute residuals
        max_idx = np.argpartition(np.abs(res.resid), -n_lre)[-n_lre:]
        # np.argpartition does not return indices in order of max to min, so another round of ordering
        max_idx = max_idx[np.argsort(np.abs(res.resid)[max_idx])][::-1]
        # can also choose outliers based on z-score
        #max_idx = np.where(np.abs((res.resid-np.mean(res.resid))/np.std(res.resid))>3)[0]
        # getting the labels for pairs of nodes from the array index
        max_res_node = []
        for k in max_idx:
            x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1
            y = np.int(k - 0.5*x*(x-1))
            max_res_node.append(tuple(sorted((x,y))))

        return(max_res_node)
    else:
        # extract indices with maximum absolute residuals
        max_idx = np.argpartition(np.abs(res.resid), -n_lre)[-n_lre:]
        # np.argpartition does not return indices in order of max to min, so another round of ordering
        max_idx = max_idx[np.argsort(np.abs(res.resid)[max_idx])][::-1]
        # can also choose outliers based on z-score
        #max_idx = np.where(np.abs((res.resid-np.mean(res.resid))/np.std(res.resid))>3)[0]
        # getting the labels for pairs of nodes from the array index
        max_res_node = []
        for k in max_idx:
            x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1
            y = np.int(k - 0.5*x*(x-1))
            max_res_node.append(tuple(sorted((x,y))))

        return(max_res_node)

def plot_estimated_vs_simulated_edges(
    graph,
    sp_Graph,
    lrn=None,
    max_res_nodes=None, 
    lamb=1.0
):
    """Function to plot estimated vs simulated edge weights to look for significant deviations
    """
    assert lamb >= 0.0, "lambda must be non-negative"
    assert type(lamb) == float, "lambda must be float"
    # both variables below are long range nodes but lrn is from the simulated and max_res_nodes is from the empirical
    assert type(lrn) == list, "lrn must be a list of int 2-tuples"
    assert type(max_res_nodes) == list, "max_res_nodes must be a list of int 2-tuples"

    # getting edges from the simulated graph
    idx = [list(graph.edges).index(val) for val in lrn]
    sim_edges = np.append(np.array([graph[val[0]][val[1]]["w"] for i, val in enumerate(graph.edges) if i not in idx]), 
                          np.array([graph[val[0]][val[1]]["w"] for i, val in enumerate(graph.edges) if i in idx]))

    idx = [list(sp_Graph.edges).index(val) for val in max_res_nodes]
    w_plot = np.append(sp_Graph.w[[i for i in range(len(sp_Graph.w)) if i not in idx]], sp_Graph.w[idx])

    X = sm.add_constant(sim_edges)
    mod = sm.OLS(w_plot[range(len(graph.edges))], X)
    res = mod.fit()
    muhat, betahat = res.params

    # getting index of long range edges
    lre_idx = [list(graph.edges).index(val) for val in lrn]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot()
    ax.scatter(sim_edges, w_plot[range(len(sim_edges))], 
            marker=".", alpha=1, zorder=0, color="grey", s=3)
    ax.scatter(sim_edges[-len(lrn)::], w_plot[-len(lrn)::], 
            marker=".", alpha=1, zorder=0, color="black", s=10)
    x_ = np.linspace(np.min(sim_edges), np.max(sim_edges), 20)
    ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
    ax.text(0.8, 0.05, "R²={:.4f}".format(res.rsquared), transform=ax.transAxes)
    ax.text(0.8, 0.15, "$\lambda$={:.3}".format(lamb), transform=ax.transAxes)
    ax.set_xlabel("simulated edge weights")
    ax.set_ylabel("estimated edge weights")

    return(None)

def plot_residual_matrix(
    sp_Graph,
    lamb_cv,
    pop_labs_file=None
):
    """Function to plot the residual matrix of the pairs of populations 
    """
    # TODO: finalize way to map samples to pops and pops to nodes

    # reading in file with sample and pop labels
    pop_labs_file = pd.read_csv()

    permuted_idx = query_node_attributes(sp_graph, "permuted_idx")
    obs_perm_ids = permuted_idx[: sp_Graph.n_observed_nodes]

    # code for mapping nodes back to populations (since multiple pops can be assigned to the same nodes)
    node_to_pop = pd.DataFrame(index = np.arange(sp_Graph.n_observed_nodes), columns = ['nodes', 'pops'])
    node_to_pop['nodes'] = obs_perm_ids
    node_to_pop['pops'] = [np.unique(sample_data['popId'][query_node_attributes(sp_Graph,"sample_idx")[x]]) for x in obs_perm_ids]

    tril_idx = np.tril_indices(sp_Graph.n_observed_nodes, k=-1)
    sp_graph.fit(lamb=lamb_cv)
    obj = Objective(sp_Graph)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[tril_idx]
    emp_dist = cov_to_dist(emp_cov)[tril_idx]

    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist, X)
    res = mod.fit()
    
    resnode = np.zeros((sp_Graph.n_observed_nodes,sp_Graph.n_observed_nodes))
    resnode[np.tril_indices_from(resmat, k=-1)] = np.abs(res.resid)
    mask = np.zeros_like(resnode)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig = plt.figure(dpi=100)
        # try clustermap(col_cluster=False)
        ax = sns.heatmap(resnode, mask=mask, square=True,  cmap=sns.color_palette("crest", as_cmap=True), xticklabels=node_to_pop['pops'])
        plt.show()

    return(None)