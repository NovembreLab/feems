""" This file contains functions to be used in miscellaneous tasks like comparing simulated to estimated results, etc
"""
# base
import math
import numpy as np
import statsmodels.api as sm
from copy import deepcopy
import pandas as pd
import scipy as sp
import os 

# viz
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx

# feems
from .utils import prepare_graph_inputs
from .sim import setup_graph, setup_graph_long_range, simulate_genotypes
from .spatial_graph import query_node_attributes
from .cross_validation import comp_mats, run_cv
from .objective import Objective
from .viz import Viz
from .joint_ver import FEEMSmix_SpatialGraph, FEEMSmix_Objective

# change matplotlib fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = "Arial"

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
    v.draw_edge_colorbar()

    #sp_Graph.fit(lamb = float(lamb[1]))
    ax = fig.add_subplot(1, 2, 2)  
    v = Viz(ax, sp_Graph, edge_width=2.0, 
            edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
            obs_node_size=7.5, sample_pt_color="black", 
            cbar_font_size=10)
    v.draw_edges(use_weights=True)
    v.draw_obs_nodes(use_ids=False) 
    v.draw_edge_colorbar()
    lre_idx = [list(sp_Graph.edges).index(val) for val in max_res_nodes]
    # paste correlation between the two weights
    ax.text(0.5, 1.0, "cor={:.2f}".format(np.corrcoef(sp_Graph.w[~np.in1d(np.arange(len(sp_Graph.w)), lre_idx)],sp_Graph_def.w)[0,1]), transform=ax.transAxes)

    return(None)

def comp_genetic_vs_fitted_distance(
    sp_Graph_def, 
    lrn=None,
    lamb=None, 
    n_lre=3, 
    plotFig=True, 
    joint=False,
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

    # sp_Graph_def.fit(lamb=lamb,lb=math.log(1e-6),ub=math.log(1e+6))
    # sp_Graph_def.comp_graph_laplacian(sp_Graph_def.w)

    if(joint):
        obj = Joint_Objective(sp_Graph_def)
    else:
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
        if lrn is not None: # then find the max_res_node (otherwise it has been passed in)
            # code below will extract maximum absolute residuals
            # max_idx = np.argpartition(np.abs(res.resid), -n_lre)[-n_lre:]
            # max_idx = max_idx[np.argsort(np.abs(res.resid)[max_idx])][::-1]

            # code below will extract maximum negative residuals 
            max_idx = np.argpartition(res.resid, n_lre)[:n_lre]
            max_idx = max_idx[np.argsort(res.resid[max_idx])]
            # getting the labels for pairs of nodes from the array index
            max_res_node = []
            for k in max_idx:
                x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1
                y = int(k - 0.5*x*(x-1))
                max_res_node.append(tuple(sorted((x,y))))

        
        # computing the vector index for lower triangular matrix of long range nodes (i+j(j+1)/2-j for lower triangle)
        lrn_idx = [int(val[0] + 0.5*val[1]*(val[1]+1) - val[1]) if val[0]<val[1] else int(val[1] + 0.5*val[0]*(val[0]+1) - val[0]) for val in max_res_node]

        fig = plt.figure(dpi=80)
        ax = fig.add_subplot()
        ax.scatter(fit_dist, emp_dist, marker=".", alpha=0.75, zorder=0, color="grey", s=3)
        ax.scatter(fit_dist[lrn_idx], emp_dist[lrn_idx], marker=".", alpha=1, zorder=0, color="black", s=10)
        x_ = np.linspace(np.min(fit_dist), np.max(fit_dist), 20)
        ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
        ax.text(0.8, 0.15, "$\lambda$={:.3}".format(lamb), transform=ax.transAxes)
        ax.text(0.8, 0.05, "R²={:.3f}".format(res.rsquared), transform=ax.transAxes)
        ax.set_ylabel("genetic distance")
        ax.set_xlabel("fitted distance")

        # fig = plt.figure(dpi=80)
        # plt.imshow(fit_cov-emp_cov); plt.colorbar(); plt.title('fit_cov_lr - emp_cov_lr')

        return(max_res_node)
    else:
        # extract indices with maximum absolute residuals
        max_idx = np.argpartition(res.resid, n_lre)[:n_lre]
        # np.argpartition does not return indices in order of max to min, so another round of ordering
        max_idx = max_idx[np.argsort(res.resid[max_idx])]
        # getting the labels for pairs of nodes from the array index
        max_res_node = []
        for k in max_idx:
            x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1
            y = int(k - 0.5*x*(x-1))
            max_res_node.append(tuple(sorted((x,y))))

        return(max_res_node)

def get_best_lre(sp_graph, k=1, lamb_cv=3., top=20, nboot=20, nchoose=100, option='base'):
    obj = Joint_Objective(sp_graph)
    ## the graph has not been fit yet, so fit it again with the given lambda
    if not hasattr(sp_graph, 'W'):
        sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph.s2))
    
    # get a list of existing edges in the NO admixture edge object
    # edges_lr = deepcopy(sp_graph.edges)
    # edges_lr = edges_lr.tolist()
    
    ll_edges = np.empty((top,k))
    top_edges = pd.DataFrame(index=range(top), columns=range(k))
    te = []
    # te = [(85,294)]

    # sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=0.5, long_range_edges=te, option='onlyc')

    for ik in range(k):
        print("Starting search for edge {}...".format(ik+1))
        if option=='base':
            max_res_nodes = comp_genetic_vs_fitted_distance(sp_graph, n_lre=top, lamb=lamb_cv, plotFig=False, joint=True)
        else:
            max_res_nodes = get_boot_edges(sp_graph, obj, lamb=lamb_cv, nreps=nboot, ntop=top, nchoose=nchoose, option=option, te=te)[:top]
        
        # code to exclude edges that have already been added in the previous steps [te should take care of this...]
        # if ik>0:
        #     max_res_nodes = [ele for ele in max_res_nodes if ele not in te]

        # iterate across remaining edges
        # print('Top 20 nodes: ',max_res_nodes)
        df = pd.DataFrame(np.random.rand(top,2), columns=['A','B']).astype('int')
        for ie, e in enumerate(max_res_nodes):
            df.iloc[ie,0] = e[0]; df.iloc[ie,1] = e[1]
        print('Potential deme A:')
        print(df.A.value_counts().head(3))
        print('Potential deme B:')
        print(df.B.value_counts().head(3))
        
        ## with admix. prop. c estimation 
        for ie, e in enumerate(max_res_nodes):
            print(e)      
            sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=0.1, verbose=False,)
            try:
                sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', option='onlyc', verbose=False, long_range_edges=te + max_res_nodes[ie:(ie+1)])
            except:
                sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', option='onlyc', verbose=False, long_range_edges=max_res_nodes[ie:(ie+1)])
                print('Could not jointly fit edge {} with previous ones, fitting independently.'.format(e))
            obj_lr = Joint_Objective(sp_graph); obj_lr.inv(); obj_lr.grad(reg=False)
            llsame = -obj_lr.neg_log_lik_c(sp_graph.c)

            ## trying both directions when fitting the edge 
            sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=0.1, verbose=False,)
            try:
                sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', option='onlyc', verbose=False, long_range_edges=te + [max_res_nodes[ie][::-1]])
            except:
                sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', option='onlyc', verbose=False, long_range_edges=[max_res_nodes[ie][::-1]])
                print('Could not jointly fit edge {} with previous ones, fitting independently.'.format(e[::-1]))
            obj_lr = Joint_Objective(sp_graph); obj_lr.inv(); obj_lr.grad(reg=False)
            llopp = -obj_lr.neg_log_lik_c(sp_graph.c)
            ll_edges[ie,ik] = llopp if llopp > llsame else llsame 

            max_res_nodes[ie] = (max_res_nodes[ie][1],max_res_nodes[ie][0]) if llopp > llsame else (max_res_nodes[ie][0],max_res_nodes[ie][1])

        # print(ll_edges, max_res_nodes)
        print("{}, found at index {}.".format(max_res_nodes[np.argmax(ll_edges[:,ik])], np.argmax(ll_edges[:,ik])))
        top_edges.iloc[:,ik] = max_res_nodes
        te.append(top_edges.iloc[np.argmax(ll_edges[:,ik]),ik])

    sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1., verbose=False)
    sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1., verbose=False, option='onlyc', long_range_edges=te)
    
    np.set_printoptions(precision=3, suppress=True)
    print("admixture proportions:")
    print(sp_graph.c)
        
    return ll_edges, top_edges

def get_boot_edges(sp_Graph, obj, nreps=20, ntop=20, nchoose=100, lamb=3., option='hard', te=[]):
    emp_dist = np.zeros((sp_Graph.n_observed_nodes*(sp_Graph.n_observed_nodes-1)//2,nreps+1))
    fit_dist = np.zeros(sp_Graph.n_observed_nodes*(sp_Graph.n_observed_nodes-1)//2)
    rng = np.random.default_rng(2022)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[np.tril_indices(sp_Graph.n_observed_nodes, k=-1)]
    emp_dist[:,0] = cov_to_dist(emp_cov)[np.tril_indices(sp_Graph.n_observed_nodes, k=-1)]

    for n2 in range(1,nreps+1):
        bootgenotypes = deepcopy(sp_Graph.genotypes)
        ## if we bootstrap over SNPs, then we can do the below
        ## bootgenotypes[:,rng.choice(range(gen_test_1e.shape[1]),gen_test_1e.shape[1])]
        ## bootstrapping over inds (since this is the source of uncertainty)
        ctr = 0
        for deme in range(sp_Graph.n_observed_nodes):
            bootgenotypes[ctr:(ctr+sp_Graph.n_samples_per_obs_node_permuted[deme]),:] = sp_Graph.genotypes[rng.choice(range(ctr,(ctr+sp_Graph.n_samples_per_obs_node_permuted[deme])),sp_Graph.n_samples_per_obs_node_permuted[deme],replace=True),:]
            ctr += sp_Graph.n_samples_per_obs_node_permuted[deme]
        ## remove SNPs that are invariant after bootstrapping
        bootgenotypes = np.delete(bootgenotypes,np.where(bootgenotypes.sum(axis=0)==0)[0],1)
        bootgenotypes = np.delete(bootgenotypes,np.where(bootgenotypes.sum(axis=0)==2*bootgenotypes.shape[0])[0],1)
        
        ## only run bootstrapping on the empirical distances
        sp_Graph.genotypes = bootgenotypes
        sp_Graph._estimate_allele_frequencies()
        sp_Graph.mu = sp_Graph.frequencies.mean(axis=0) / 2
        sp_Graph.frequencies = sp_Graph.frequencies / np.sqrt(sp_Graph.mu * (1 - sp_Graph.mu))
        frequencies_centered = sp_Graph.frequencies - sp_Graph.frequencies.mean(axis=0)
        emp_cov = frequencies_centered @ frequencies_centered.T / sp_Graph.n_snps

        ## fit the model with the bootstrapped genotypes
        # sp_graph_boot = Joint_SpatialGraph(bootgenotypes, coord, grid, edge)
        # sp_graph_boot.fit(lamb=lamb, optimize_q='n-dim',verbose=False)
        # obj_boot = Joint_Objective(sp_graph_boot)
        # fit_cov, _, emp_cov = comp_mats(obj_boot)
        # fit_dist[:,n2] = cov_to_dist(fit_cov)[np.tril_indices(sp_graph_boot.n_observed_nodes, k=-1)]
        emp_dist[:,n2] = cov_to_dist(emp_cov)[np.tril_indices(sp_Graph.n_observed_nodes, k=-1)]
    
    # computing the best-fit linear regression on the whole data set
    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist[:,0], X)
    res = mod.fit()
    muhat, betahat = res.params

    # finding the largest outliers, taking into account the SE might overlap with best-fit line
    max_idx = np.argpartition(res.resid, nchoose)[:nchoose]
    max_idx = max_idx[np.argsort(res.resid[max_idx])]
    new_max_idx = []

    se_resid = np.zeros(fit_dist.shape[0])
    for i in range(fit_dist.shape[0]):
        se_resid[i] = np.std(((betahat*fit_dist[i]+muhat)-emp_dist[i,:])-res.resid[i])/np.sqrt(nreps)

    if option=='hard':       
        # computing the standard errors from before
        se_emp = np.zeros(fit_dist.shape[0])
        # se_fit = np.zeros_like(se_emp)
        for i in range(fit_dist.shape[0]):
            # se_fit[i] = np.sqrt(np.sum((fit_dist[i,:]-np.mean(fit_dist[i,:]))**2)/(nreps-1))
            se_emp[i] = np.sqrt(np.sum((emp_dist[i,:]-np.mean(emp_dist[i,:]))**2)/(nreps-1))

        for i in max_idx:
            # if ((fit_dist[i,0]-se_fit[i])>(emp_dist[i,0]-muhat)/betahat) & ((betahat*fit_dist[i,0]+muhat) > (emp_dist[i,0]+se_emp[i])):
            if (betahat*fit_dist[i]+muhat) > (emp_dist[i,0]+se_emp[i]):
                new_max_idx.append(i)

    elif option=='ashr':
        np.savetxt('/Users/vivaswatshastry/Desktop/feemsashr.txt',np.vstack((res.resid,se_resid)).T)
        os.system('Rscript /Users/vivaswatshastry/Desktop/run_ashr.R')
        resash = np.loadtxt('/Users/vivaswatshastry/Desktop/feemsresash.txt',)
        new_max_idx = np.argpartition(resash[:,0], nchoose)[:nchoose]
        new_max_idx = new_max_idx[np.argsort(resash[new_max_idx,0])]

    new_max_res_node = []
    for k in new_max_idx:
        x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1
        y = int(k - 0.5*x*(x-1))
        new_max_res_node.append(tuple(sorted((x,y))))

    ## only return nodes that were not previously found (cos its already been modeled)
    badnodes = [item for sublist in te for item in sublist]
    new_max_res_node = [x for x in new_max_res_node if x[0] not in badnodes and x[1] not in badnodes]

    return new_max_res_node[:ntop]

## no longer fitting an edge, so don't have to worry about the penalty term
# for ie, e in enumerate(max_res_nodes):
#     edges_t = deepcopy(edges_lr)
#     edges_t.append(list(x+1 for x in e))
#     sp_graph_lr = Joint_SpatialGraph(gen_test, coord, grid, np.array(edges_t), long_range_edges=max_res_nodes[ie:(ie+1)])
#     try:
#         sp_graph_lr.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph_lr.s2),verbose=False)
#     except:
#         sp_graph_lr.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=100., alpha_q=1./np.mean(sp_graph_lr.s2),verbose=False)
#     obj_lr = Joint_Objective(sp_graph_lr); obj_lr.inv(); 
#     ll_edges[ie,ik] = -obj_lr.neg_log_lik()
## adding the best fit edge to the sp_graph object
# edges_lr.append(list(x+1 for x in max_res_nodes[np.argmax(ll_edges[:,ik])]))
# sp_graph = Joint_SpatialGraph(gen_test, coord, grid, np.array(edges_lr), long_range_edges=[max_res_nodes[np.argmax(ll_edges[:,ik])]])
# sp_graph.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph.s2))

# def get_best_lre(sp_graph_lr, gen_test, coord, grid, edge_def, k=5, nfolds=None, lamb_cv=3., top=20):
#     sp_graph_lr.fit(lamb=lamb_cv)#, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph_lr.s2))
#     edges_lr = deepcopy(edge_def)
#     edges_lr = edges_lr.tolist()
#     ll_edges = np.empty((top,k))
#     top_edges = pd.DataFrame(index=range(top), columns=range(k))
#     for ik in range(k):
#         print("Starting search for edge {}...".format(ik+1))
#         lrn = comp_genetic_vs_fitted_distance(sp_graph_lr, n_lre=top, lamb=lamb_cv, plotFig=False, joint=True)
#         # lrn = list(map(tuple,node_to_pop['nodes'][np.ravel(max_res_nodes)].values.reshape(top,2)))
#         edges_lr.append(list(x+1 for x in lrn[0]))
#         # print(len(edges_lr))
#         sp_graph_lr = Joint_SpatialGraph(gen_test, coord, grid, np.array(edges_lr), long_range_edges=lrn[0:1])
#         sp_graph_lr.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph_lr.s2),verbose=False)
#         # print(len(sp_graph_lr.edges))
#         obj_lr = Joint_Objective(sp_graph_lr); obj_lr.inv(); 
#         ll_edges[0,ik] = -obj_lr.neg_log_lik()
#         for ie, e in enumerate(lrn[1:]):
#             ll_edges[ie+1,ik] = sub_edge_get_ll(sp_graph_lr, lrn[ie], lrn[ie+1], 3.)
#         print("{}, found at index {}.".format(lrn[np.argmax(ll_edges[:,ik])], np.argmax(ll_edges[:,ik])))
#         top_edges.iloc[:,ik] = lrn
#         sub_edge_get_ll(sp_graph_lr, lrn[len(lrn)-1], lrn[np.argmax(ll_edges[:,ik])], 3.)
#         edges_lr = [list(x+1 for x in lrn[np.argmax(ll_edges[:,ik])]) if item == list(x+1 for x in lrn[0]) else item for item in edges_lr] 
#         # print(len(edges_lr))
#         # print(lrn, ll_edges[:,ik])
#         # edges_lr.append(list(x+1 for x in lrn[np.argmax(ll_edges[:,ik])]))
#         # sp_graph = Joint_SpatialGraph(gen_test, coord, grid, np.array(edges_lr), long_range_edges=[lrn[np.argmax(ll_edges[:,ik])]])
#         sp_graph_lr.fit(lamb=lamb_cv, optimize_q='n-dim', lamb_q=1., alpha_q=1./np.mean(sp_graph_lr.s2))
       
#     return ll_edges, top_edges

def plot_estimated_vs_simulated_edges(
    # graph,
    sp_Graph,
    w_true, 
    lrn=None,
    # max_res_nodes=None, 
    lamb=1.0, 
    beta=0., 
    joint=False
):
    """Function to plot estimated vs simulated edge weights to look for significant deviations
    """
    assert lamb >= 0.0, "lambda must be non-negative"
    assert type(lamb) == float, "lambda must be float"
    # both variables below are long range nodes but lrn is from the simulated and max_res_nodes is from the empirical
    # assert type(lrn) == list, "lrn must be a list of int 2-tuples"
    # assert type(max_res_nodes) == list, "max_res_nodes must be a list of int 2-tuples"

    # getting edges from the simulated graph
    # idx = [list(graph.edges).index(val) for val in lrn]
    # sim_edges = np.append(np.array([graph[val[0]][val[1]]["w"] for i, val in enumerate(graph.edges) if i not in idx]), 
    #                       np.array([graph[val[0]][val[1]]["w"] for i, val in enumerate(graph.edges) if i in idx]))

    # idx = [list(sp_Graph.edges).index(val) for val in max_res_nodes]
    # w_plot = np.append(sp_Graph.w[[i for i in range(len(sp_Graph.w)) if i not in idx]], sp_Graph.w[idx])

    # sp_Graph.fit(lamb = lamb, beta=beta)

    if(joint):
        obj = Joint_Objective(sp_Graph)
    else:
        obj = Objective(sp_Graph)

    tril_idx = np.tril_indices(sp_Graph.n_observed_nodes, k=-1)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[tril_idx]
    emp_dist = cov_to_dist(emp_cov)[tril_idx]

    # using code from supp fig 6 of feems-analysis
    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist, X)
    res = mod.fit()
    muhat, betahat = res.params

    if lrn is None:
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(2, 2, (1,2))
        v = Viz(ax, sp_Graph, edge_width=2.0, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=10, 
                obs_node_size=4.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_edges(use_weights=True)
        v.draw_obs_nodes(use_ids=False) 
        v.draw_edge_colorbar()
        # lre_idx = [list(sp_Graph.edges).index(val) for val in max_res_nodes]
        # paste correlation between the two weights
        # ax.text(0.5, 1.0, "cor={:.2f}".format(np.corrcoef(sp_Graph.w[~sp_Graph.lre_idx],w_true[~sp_Graph.lre_idx])[0,1]), transform=ax.transAxes)

        ax = fig.add_subplot(2, 2, 3)
        if len(sp_Graph.w)==len(w_true):
            ax.scatter(w_true, sp_Graph.w, color='black', alpha=0.8)
        else:
            ax.scatter(np.delete(w_true,list(sp_Graph.edges).index(lrn[0])), sp_Graph.w[~sp_Graph.lre_idx], color='black', alpha=0.8)
        ax.set_xlabel('true weights')
        ax.set_ylabel('estimated weights')
        ax.set_title('λ = {:.1f}, β = {:.1f}'.format(lamb, beta))
        ax.grid()
        
        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(fit_dist, emp_dist, marker=".", alpha=0.75, zorder=0, color="grey", s=3)
        x_ = np.linspace(np.min(fit_dist), np.max(fit_dist), 20)
        ax.plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=1)
        ax.text(0.8, 0.15, "$\lambda$={:.3}".format(lamb), transform=ax.transAxes)
        ax.text(0.8, 0.05, "R²={:.4f}".format(res.rsquared), transform=ax.transAxes)
        ax.set_ylabel("genetic distance")
        ax.set_xlabel("fitted distance")
    else:
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(2, 2, (1,2))
        v = Viz(ax, sp_Graph, edge_width=2.0, 
                edge_alpha=1, edge_zorder=100, sample_pt_size=10, 
                obs_node_size=4.5, sample_pt_color="black", 
                cbar_font_size=10)
        v.draw_edges(use_weights=True)
        v.draw_obs_nodes(use_ids=False) 
        v.draw_edge_colorbar()

        ax = fig.add_subplot(2, 2, 3)
        if len(sp_Graph.w)==len(w_true):
            ax.scatter(w_true, sp_Graph.w, color='black', alpha=0.8)
        else:
            ax.scatter(np.delete(w_true,list(sp_Graph.edges).index(lrn[0])), sp_Graph.w[~sp_Graph.lre_idx], color='black', alpha=0.8)
        ax.set_xlabel('true weights')
        ax.set_ylabel('estimated weights')
        ax.set_title('λ = {:.1f}, β = {:.1f}'.format(lamb, beta))
        ax.grid()

        ax = fig.add_subplot(2, 2, 4)
        ax.hist(sp_Graph.w[sp_Graph.lre_idx], color='grey', alpha=0.8)
        ax.set_xlabel('long range edge weights')

    return(None)

def sub_edge_get_ll(sp_Graph, old_edge, new_edge, lamb): 
    """Function to substitute edges into an already constructed graph object, this should make the process a whole lot faster. 
    derp...it does not, computing the gradient takes a chunk of the time versus assigning samples to nodes
    (Edges should be tuples of length 2) 
    """
    sp_Graph.remove_edge(*old_edge) 
    sp_Graph.add_edge(*new_edge)
    
    # sp_Graph.lre = [new_edge]
    # sp_Graph.lre_idx = np.array([val in sp_Graph.lre for val in list(sp_Graph.edges)])
    sp_Graph.Delta_q = nx.incidence_matrix(sp_Graph, oriented=True).T.tocsc()

    sp_Graph.adj_base = sp.sparse.triu(nx.adjacency_matrix(sp_Graph), k=1)

    sp_Graph.nnz_idx = sp_Graph.adj_base.nonzero()

    sp_Graph.Delta = sp_Graph._create_incidence_matrix()

    sp_Graph.diag_oper = sp_Graph._create_vect_matrix()

    sp_Graph._create_perm_diag_op() 

    sp_Graph.comp_grad_w()

    sp_Graph.fit(lamb = float(lamb), optimize_q='n-dim', lamb_q=1., alpha_q=1., verbose=False)
    obj = Joint_Objective(sp_Graph); obj.inv()
    return -obj.neg_log_lik()

def plot_residual_matrix(
    sp_Graph,
    lamb_cv,
    node_to_pop=None,
    pop_labs_file=None
):
    """Function to plot the residual matrix of the pairs of populations 
    """
    # TODO: finalize way to map samples to pops and pops to nodes

    # reading in file with sample and pop labels
    if pop_labs_file is not None:
        pop_labs_file = pd.read_csv()

    permuted_idx = query_node_attributes(sp_Graph, "permuted_idx")
    obs_perm_ids = permuted_idx[: sp_Graph.n_observed_nodes]

    tril_idx = np.tril_indices(sp_Graph.n_observed_nodes, k=-1)
    #sp_graph.fit(lamb=lamb_cv)
    obj = Objective(sp_Graph)
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[tril_idx]
    emp_dist = cov_to_dist(emp_cov)[tril_idx]

    X = sm.add_constant(fit_dist)
    mod = sm.OLS(emp_dist, X)
    res = mod.fit()
    
    resnode = np.zeros((sp_Graph.n_observed_nodes,sp_Graph.n_observed_nodes))
    resnode[np.tril_indices_from(resnode, k=-1)] = np.abs(res.resid)
    mask = np.zeros_like(resnode)
    mask[np.triu_indices_from(mask)] = True
    # with sns.axes_style("white"):
    #     fig = plt.figure(dpi=100)
    #     # try clustermap(col_cluster=False)
    #     ax = sns.heatmap(resnode, mask=mask, square=True,  cmap=sns.color_palette("crest", as_cmap=True), xticklabels=node_to_pop['pops'])
    #     plt.show()

    return(resnode)