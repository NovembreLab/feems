""" This file contains functions to be used in miscellaneous tasks like comparing simulated to estimated results, etc
"""
# base
import math
import numpy as np
import statsmodels.api as sm

# viz
import matplotlib.pyplot as plt
from matplotlib import gridspec

# feems
from feems.utils import prepare_graph_inputs
from feems import SpatialGraph, Viz, Objective
from feems.sim import setup_graph, setup_graph_long_range, simulate_genotypes
from feems.spatial_graph import query_node_attributes
from feems.cross_validation import comp_mats

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
    lrn
):
    """Function to plot default graph with NO long range edges next to full graph with long range edges
    (useful for comparison of feems default fit with extra parameters)
    """
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(1, 2, 1)  
    v = Viz(ax, sp_Graph_def, edge_width=1.5, 
            edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
            obs_node_size=7.5, sample_pt_color="black", 
            cbar_font_size=10)
    v.draw_edges(use_weights=True)
    v.draw_obs_nodes(use_ids=False) 

    sp_Graph.fit(lamb = 1.0)
    ax = fig.add_subplot(1, 2, 2)  
    v = Viz(ax, sp_Graph, edge_width=1.5, 
            edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
            obs_node_size=7.5, sample_pt_color="black", 
            cbar_font_size=10)
    v.draw_edges(use_weights=True)
    v.draw_obs_nodes(use_ids=False) 
    lre_idx = [list(sp_Graph.edges).index(val) for val in lrn]
    # paste correlation between the two weights
    ax.text(0.5, 1.0, "cor={:.2f}".format(np.corrcoef(sp_Graph.w[~np.in1d(np.arange(len(sp_Graph.w)), lre_idx)],sp_Graph_def.w)[0,1]), transform=ax.transAxes)

    return(None)

def comp_genetic_vs_fitted_distance(
    sp_Graph, 
    lrn, 
    lamb=1.0, 
    plotFig=True
):
    """Function to plot genetic vs fitted distance to visualize outliers in residual calculations
    """
    tril_idx = np.tril_indices(sp_Graph.n_observed_nodes, k=-1)
    
    sp_Graph.fit(lamb=lamb,
                lb=math.log(1e-6), 
                ub=math.log(1e+6))
    sp_Graph.comp_graph_laplacian(sp_Graph.w)

    obj = Objective(sp_Graph)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[tril_idx]
    emp_dist = cov_to_dist(emp_cov)[tril_idx]

    # computing the vector index for lower triangular matrix of long range nodes (i+j(j+1)/2-j for lower triangle)
    lrn_idx = [np.int(val[0] + 0.5*val[1]*(val[1]+1) - val[1]) if val[0]<val[1] else np.int(val[1] + 0.5*val[0]*(val[0]+1) - val[0]) for val in lrn]

    # using code from supp fig 6 of feems-analysis
    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist, X)
    res = mod.fit()
    muhat, betahat = res.params
    if(plotFig):
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot()
        ax.scatter(fit_dist, emp_dist, 
                marker=".", alpha=1, zorder=0, color="grey", s=3)
        ax.scatter(fit_dist[lrn_idx], emp_dist[lrn_idx], 
                marker=".", alpha=1, zorder=0, color="black", s=10)
        x_ = np.linspace(np.min(fit_dist), np.max(fit_dist), 20)
        ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
        ax.text(0.8, 0.15, "$\lambda$={:1.0e}".format(lamb), transform=ax.transAxes)
        ax.text(0.8, 0.05, "R²={:.4f}".format(res.rsquared), transform=ax.transAxes)
        ax.set_ylabel("genetic distance")
        ax.set_xlabel("fitted distance")
        return(None)
    else:
        # TODO: return pairs of nodes with absoluate deviations from residuals > 2?
        return([(1,13)])

def plot_estimated_vs_simulated_edges(
    graph,
    sp_Graph,
    lrn,
    lamb=1.0
):
    """Function to plot estimated vs simulated edge weights to look for significant deviations
    """
    # getting edges that are long range in the simulated graph
    sim_edges = np.array([graph[val[0]][val[1]]["w"] for _, val in enumerate(list(graph.edges))])

    X = sm.add_constant(sim_edges)
    mod = sm.OLS(sp_Graph.w, X)
    res = mod.fit()
    muhat, betahat = res.params

    # getting index of long range edges
    lre_idx = [list(sp_Graph.edges).index(val) for val in lrn]

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()
    ax.scatter(sim_edges, sp_Graph.w, 
            marker=".", alpha=1, zorder=0, color="grey", s=3)
    ax.scatter(sim_edges[lre_idx], sp_Graph.w[lre_idx], 
            marker=".", alpha=1, zorder=0, color="black", s=10)
    x_ = np.linspace(np.min(sim_edges), np.max(sim_edges), 20)
    ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
    ax.text(0.8, 0.05, "R²={:.4f}".format(res.rsquared), transform=ax.transAxes)
    ax.text(0.8, 0.15, "$\lambda$={:1.0e}".format(lamb), transform=ax.transAxes)
    ax.set_xlabel("simulated edge weights")
    ax.set_ylabel("estimated edge weights")

    return(None)