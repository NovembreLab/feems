from __future__ import absolute_import, division, print_function

import sys

from copy import copy, deepcopy
import itertools as it
import networkx as nx
import numbers
import numpy as np
from scipy.linalg import pinvh
from scipy.optimize import fmin_l_bfgs_b, minimize
import scipy.sparse as sp
from scipy.stats import chi2, norm
import sksparse.cholmod as cholmod
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from .objective import Objective, loss_wrapper, neg_log_lik_w0_s2, comp_mats, interpolate_q
from .utils import cov_to_dist, benjamini_hochberg

class SpatialGraph(nx.Graph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True):
        """Represents the spatial network which the data is defined on and
        stores relevant matrices / performs linear algebra routines needed for
        the model and optimization. Inherits from the networkx Graph object.

        Args:
            genotypes (:obj:`numpy.ndarray`): genotypes for samples
            sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
            scale_snps (:obj:`Bool`): boolean to scale SNPs by SNP specific
                Binomial variance estimates
            c (:obj:`float`): float to store value of admix. prop. between two demes
        """
        # check inputs
        assert len(genotypes.shape) == 2
        assert len(sample_pos.shape) == 2
        assert np.all(~np.isnan(genotypes)), "no missing genotypes are allowed"
        assert np.all(~np.isinf(genotypes)), "non inf genotypes are allowed"
        assert (
            genotypes.shape[0] == sample_pos.shape[0]
        ), "genotypes and sample positions must be the same size"

        # remove invariant SNPs
        if np.sum(np.where(genotypes.sum(axis=0)==0)[0]) > 0 or np.sum(np.where(genotypes.sum(axis=0)==2*genotypes.shape[0])[0]) > 0:
            print('FEEMS requires polymorphic SNPs, but ID(s) {:g} were found to be invariant. '.format(list(np.where(genotypes.sum(axis=0)==0)[0]) + list(np.where(genotypes.sum(axis=0)==2*genotypes.shape[0])[0])))
            print('Running analyses by removing these SNPs from the genotype matrix...')
            genotypes = np.delete(genotypes,np.where(genotypes.sum(axis=0)==0)[0],1)
            genotypes = np.delete(genotypes,np.where(genotypes.sum(axis=0)==2*genotypes.shape[0])[0],1)

        # inherits from networkx Graph object -- changed this to new signature for python3
        print("Initializing graph...")
        super().__init__()
        self._init_graph(node_pos, edges)  # init graph

        # inputs
        self.sample_pos = sample_pos
        self.node_pos = node_pos
        self.scale_snps = scale_snps
        self.option = 'default'

        self.optimize_q = None
        
        print("Computing graph attributes...")
        # signed incidence_matrix
        self.Delta_q = nx.incidence_matrix(self, oriented=True).T.tocsc()

        # track nonzero edges upper triangular
        self.adj_base = sp.triu(nx.adjacency_matrix(self), k=1)
        self.nnz_idx = self.adj_base.nonzero()

        # adjacency matrix on the edges
        self.Delta = self._create_incidence_matrix()

        # vectorization operator on the edges
        self.diag_oper = self._create_vect_matrix()

        print("Assigning samples to nodes", end="...")
        self._assign_samples_to_nodes(sample_pos, node_pos)  # assn samples
        self._permute_nodes()  # permute nodes
        n_samples_per_node = query_node_attributes(self, "n_samples")
        permuted_idx = query_node_attributes(self, "permuted_idx")
        n_samps = n_samples_per_node[permuted_idx]
        self.n_samples_per_obs_node_permuted = n_samps[: self.n_observed_nodes]
        self._create_perm_diag_op()  # create perm operator
        self.factor = None  # sparse cholesky factorization of L11

        # initialize w
        self.w = np.ones(self.size())

        # compute gradient of the graph laplacian with respect to w (dL / dw)
        # this only needs to be done once
        self.comp_grad_w()

        # estimate allele frequencies at observed locations (in permuted order)
        self.genotypes = genotypes
        self._estimate_allele_frequencies()

        if scale_snps:
            self.mu = self.frequencies.mean(axis=0) / 2
            self.frequencies = self.frequencies / np.sqrt(self.mu * (1 - self.mu))

        # compute precision
        self.comp_precision(s2=1)

        # vector to store the kriging-interpolated q values
        self.q_prox = np.ones(len(self) - self.n_observed_nodes)

        # estimate sample covariance matrix
        self.S = self.frequencies @ self.frequencies.T / self.n_snps

        # creating an internal index for easier access
        self.perm_idx = query_node_attributes(self, "permuted_idx") 

        # container to store long-range edge attributes
        self.edge = []
        self.c = None

        print("done.")

    def _init_graph(self, node_pos, edges):
        """Initialize the graph and related graph objects

        Args:
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
        """
        self.add_nodes_from(np.arange(node_pos.shape[0]))
        self.add_edges_from((edges - 1).tolist())

        # add spatial coordinates to node attributes
        for i in range(len(self)):
            self.nodes[i]["idx"] = i
            self.nodes[i]["pos"] = node_pos[i, :]
            self.nodes[i]["n_samples"] = 0
            self.nodes[i]["sample_idx"] = []

    def _create_incidence_matrix(self):
        """Create a signed incidence matrix on the edges
        * note this is computed only once
        """
        data = np.array([], dtype=float)
        row_idx = np.array([], dtype=int)
        col_idx = np.array([], dtype=int)
        n_count = 0
        for i in range(self.size()):
            edge1 = np.array([self.nnz_idx[0][i], self.nnz_idx[1][i]])
            for j in range(i + 1, self.size()):
                edge2 = np.array([self.nnz_idx[0][j], self.nnz_idx[1][j]])
                if len(np.intersect1d(edge1, edge2)) > 0:
                    data = np.append(data, 1)
                    row_idx = np.append(row_idx, n_count)
                    col_idx = np.append(col_idx, i)

                    data = np.append(data, -1)
                    row_idx = np.append(row_idx, n_count)
                    col_idx = np.append(col_idx, j)

                    # increment
                    n_count += 1

        Delta = sp.csc_matrix(
            (data, (row_idx, col_idx)), shape=(int(len(data) / 2.0), self.size())
        )
        return Delta

    def _create_vect_matrix(self):
        """Construct matrix operators S so that S*vec(W) is the degree vector
        * note this is computed only once
        """
        row_idx = np.repeat(np.arange(len(self)), len(self))
        col_idx = np.array([], dtype=int)
        for ite, i in enumerate(range(len(self))):
            idx = np.arange(0, len(self) ** 2, len(self)) + ite
            col_idx = np.append(col_idx, idx)
        S = sp.csc_matrix(
            (np.ones(len(self) ** 2), (row_idx, col_idx)),
            shape=(len(self), len(self) ** 2),
        )
        return S

    def _assign_samples_to_nodes(self, sample_pos, node_pos):
        """Assigns each sample to a node on the graph by finding the closest
        node to that sample
        """
        n_samples = sample_pos.shape[0]
        assned_node_idx = np.zeros(n_samples, "int")
        for i in range(n_samples):
            dist = (sample_pos[i, :] - node_pos) ** 2
            idx = np.argmin(np.sum(dist, axis=1))
            assned_node_idx[i] = idx
            self.nodes[idx]["n_samples"] += 1
            self.nodes[idx]["sample_idx"].append(i)
        n_samples_per_node = query_node_attributes(self, "n_samples")
        self.n_observed_nodes = np.sum(n_samples_per_node != 0)
        self.assned_node_idx = assned_node_idx

    def _permute_nodes(self):
        """Permutes all graph matrices to start with the observed nodes first
        and then the unobserved nodes
        """
        # indicies of all nodes
        node_idx = query_node_attributes(self, "idx")
        n_samples_per_node = query_node_attributes(self, "n_samples")

        # set permuted node ids as node attribute
        ns = n_samples_per_node != 0
        s = n_samples_per_node == 0
        permuted_node_idx = np.concatenate([node_idx[ns], node_idx[s]])
        permuted_idx_dict = dict(zip(node_idx, permuted_node_idx))
        nx.set_node_attributes(self, permuted_idx_dict, "permuted_idx")

    def _create_perm_diag_op(self):
        """Creates permute diag operator"""
        # query permuted node ids
        permuted_node_idx = query_node_attributes(self, "permuted_idx")

        # construct adj matrix with permuted nodes
        row = permuted_node_idx.argsort()[self.nnz_idx[0]]
        col = permuted_node_idx.argsort()[self.nnz_idx[1]]
        self.nnz_idx_perm = (row, col)
        self.adj_perm = sp.coo_matrix(
            (np.ones(self.size()), (row, col)), shape=(len(self), len(self))
        )

        # permute diag operator
        vect_idx_r = row + len(self) * col
        vect_idx_c = col + len(self) * row
        self.P = self.diag_oper[:, vect_idx_r] + self.diag_oper[:, vect_idx_c]

    def _update_graph(self, basew, bases2):
        # self.option = 'default'
        
        self.w = basew; self.s2 = bases2

        self.comp_graph_laplacian(basew); self.comp_precision(bases2)
        
        obj = Objective(self)
        obj.inv(); obj.grad(reg=False); obj.Linv_diag = obj._comp_diag_pinv()

        Rmatdo = -2 * obj.Linv[self.n_observed_nodes:, :self.n_observed_nodes] + obj.Linv[:self.n_observed_nodes, :self.n_observed_nodes].diagonal() + obj.Linv_diag[self.n_observed_nodes:, np.newaxis]
        Rmatoo = -2*obj.Linv[:self.n_observed_nodes, :self.n_observed_nodes] + np.broadcast_to(np.diag(obj.Linv),(self.n_observed_nodes, self.n_observed_nodes)).T + np.broadcast_to(np.diag(obj.Linv), (self.n_observed_nodes, self.n_observed_nodes))
        
        self.q_prox = interpolate_q(1/self.q, Rmatdo, Rmatoo)
    
    def inv_triu(self, w, perm=True):
        """Take upper triangular vector as input and return symmetric weight
        sparse matrix
        """
        if perm:
            W = self.adj_perm.copy()
        else:
            W = self.adj_base.copy()
        W.data = w
        W = W + W.T
        return W.tocsc()

    def comp_graph_laplacian(self, weight, perm=True):
        """Computes the graph laplacian (note: this is computed each step of the
        optimization so needs to be fast)
        """
        if "array" in str(type(weight)) and weight.shape[0] == len(self):
            self.m = weight
            self.w = self.B @ self.m
            self.W = self.inv_triu(self.w, perm=perm)
        elif "array" in str(type(weight)):
            self.w = weight
            self.W = self.inv_triu(self.w, perm=perm)
        elif "matrix" in str(type(weight)):
            self.W = weight
        else:
            # TODO: decide to raise error here?
            print("inaccurate argument")
        W_rowsum = np.array(self.W.sum(axis=1)).reshape(-1)
        self.D = sp.diags(W_rowsum).tocsc()
        self.L = self.D - self.W
        self.L_block = {
            "oo": self.L[: self.n_observed_nodes, : self.n_observed_nodes],
            "dd": self.L[self.n_observed_nodes :, self.n_observed_nodes :],
            "do": self.L[self.n_observed_nodes :, : self.n_observed_nodes],
            "od": self.L[: self.n_observed_nodes, self.n_observed_nodes :],
        }

        if self.factor is None:
            # initialize the object if the cholesky factorization has not been
            # computed yet. This will perform the fill-in reducing permutation
            # and the cholesky factorization which is "slow" initially
            self.factor = cholmod.cholesky(self.L_block["dd"])
        else:
            # if it has been computed we can quickly update the factorization
            # by calling the cholesky method of factor which does not perform
            # the fill-in reducing permutation again because the sparsity
            # pattern of L11 is fixed throughout the algorithm
            self.factor = self.factor.cholesky(self.L_block["dd"])

    def comp_grad_w(self):
        """Computes the derivative of the graph laplacian with respect to the
        latent variables (dw / dm) note this is computed only once
        """
        # nonzero indexes
        idx = self.nnz_idx_perm

        # elements of mat
        data = 0.5 * np.ones(idx[0].shape[0] * 2)

        # row and columns indicies
        row = np.repeat(np.arange(idx[0].shape[0]), 2)
        col = np.ravel([idx[0], idx[1]], "F")

        # construct operator w = B*m
        sp_tup = (data, (row, col))
        self.B = sp.csc_matrix(sp_tup, shape=(idx[0].shape[0], len(self)))

    # ------------------------- Data -------------------------

    def _estimate_allele_frequencies(self):
        """Estimates allele frequencies by maximum likelihood on the observed
        nodes (in permuted order) of the spatial graph
        """
        self.n_snps = self.genotypes.shape[1]

        # create the data matrix of means
        self.frequencies = np.empty((self.n_observed_nodes, self.n_snps))

        # get indicies
        sample_idx = nx.get_node_attributes(self, "sample_idx")
        permuted_idx = query_node_attributes(self, "permuted_idx")
        observed_permuted_idx = permuted_idx[: self.n_observed_nodes]

        # loop of the observed nodes in order of the permuted nodes
        for i, node_id in enumerate(observed_permuted_idx):

            # find the samples assigned to the ith node
            s = sample_idx[node_id]

            # compute mean at each node
            allele_counts = np.mean(self.genotypes[s, :], axis=0) / 2 
            self.frequencies[i, :] = allele_counts

    def comp_precision(self, s2):
        """Computes the residual precision matrix"""
        o = self.n_observed_nodes
        self.s2 = s2
        if 'array' in str(type(s2)) and len(s2) > 1:
            self.q = self.n_samples_per_obs_node_permuted/self.s2[:o]
        elif 'array' in str(type(s2)) and len(s2) == 1:
            self.s2 = s2[0]
            self.q = self.n_samples_per_obs_node_permuted / self.s2
        else:
            self.q = self.n_samples_per_obs_node_permuted / self.s2
        
        self.q_diag = sp.diags(self.q).tocsc()
        self.q_inv_diag = sp.diags(1.0 / self.q).tocsc()
        self.q_inv_grad = -1.0 / self.n_samples_per_obs_node_permuted
        if 'array' in str(type(s2)) and len(s2) > 1:
            self.q_inv_grad = -sp.diags(1./self.n_samples_per_obs_node_permuted).tocsc()    
        else:
            self.q_inv_grad = -1./self.n_samples_per_obs_node_permuted   

    # ------------------------- Optimizers -------------------------

    def fit_null_model(self, verbose=True):
        """Estimates of the edge weights and residual variance
        under the model that all the edge weights have the same value
        """
        obj = Objective(self)
        res = minimize(neg_log_lik_w0_s2, [0.0, 0.0], method="L-BFGS-B", args=(obj))
        assert res.success is True, "did not converge"
        w0_hat = np.exp(res.x[0])
        s2_hat = np.exp(res.x[1])
        self.s2_hat = s2_hat
        self.w0 = w0_hat * np.ones(self.w.shape[0])
        self.s2 = s2_hat * np.ones(len(self))
        self.comp_precision(s2=s2_hat)

        # print update
        self.train_loss = neg_log_lik_w0_s2(np.r_[np.log(w0_hat), np.log(s2_hat)], obj)
        if verbose:
            sys.stdout.write(
                (
                    "constant-w/variance fit, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(res.nfev, self.train_loss)
            )

    def independent_fit(
        self, 
        outliers_df,
        lamb,
        lamb_q,
        optimize_q='n-dim',
        fdr=0.3,
        stop=None,
        top=0.05,
        exclude_boundary=True,
        maxls=50,
        m=10,
        factr=1e7,
        lb=-np.inf,
        ub=np.inf,
        maxiter=15000,
        search_area='all',
        opts=None
    ):
        """Function to iteratively fit a long range gene flow event to the graph until there are no more outliers (`alternate method`).
        Required:
            lamb (:obj:`float`): penalty strength on weights
            lamb_q (:obj:`float`): penalty strength on the residual variances
            
        Optional:
            optimize_q (:obj:'str'): indicator for method of optimizing residual variances (one of 'n-dim', '1-dim' or None)
            fdr (:obj:`float`): false-discovery rate of outlier edges 
            pval (:obj:`float`): p-value for assessing whether adding an admixture edge significantly increases log-likelihood over previous fit
            stop (:obj:`int`): number of admixture edges to add sequentially 
            exclude_boundary (:obj:`Bool`): whether to exclude boundary nodes from fitting procedure
            alpha (:obj:`float`): penalty strength on log weights
            alpha_q (:obj:`float`): penalty strength on log residual variances
            factr (:obj:`float`): tolerance for convergence 
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log weights
            ub (:obj:`int`): upper bound of log weights
            maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
            verbose (:obj:`Bool`): boolean to print summary of results

        Returns: 
            (:obj:`pandas.DataFrame`)
        """
        
        obj = Objective(self)
        obj.inv(); obj.grad(reg=False); obj.Linv_diag = obj._comp_diag_pinv()

        # storing the baseline weigths & s2
        usew = deepcopy(obj.sp_graph.w); uses2 = deepcopy(obj.sp_graph.s2)

        # dict storing all the results for plotting
        results = {}
        
        # passing in dummy variables just to initialize the procedure
        # TODO is this a valid dummy variable? see if you can compute using just FEEMS weights
        args = {'edge':[(self.perm_idx[0],self.perm_idx[1])], 'mode':'update'}
        nllnull = obj.eems_neg_log_lik(None, args)
        print('Log-likelihood of initial fit: {:.1f}\n'.format(nllnull))

        fit_cov, _, emp_cov = comp_mats(obj)
        fit_dist = cov_to_dist(fit_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]
        emp_dist = cov_to_dist(emp_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]

        results[0] = {'log-lik': -nllnull, 
                     'emp_dist': emp_dist,
                     'fit_dist': fit_dist,
                     'outliers_df': outliers_df,
                     'fdr': fdr}

        b, c = np.unique(outliers_df['dest.'], return_counts=True)
        dest = list(b[np.argsort(-c)])

        if stop is None:
            stop = len(dest)

        cnt = 1
        
        while cnt <= stop:
            print('\nFitting admixture edge to deme {:d}:'.format(dest[cnt-1]))
            self._update_graph(usew, uses2)
            
            # get the log-lik surface across the landscape
            if search_area=='radius':
                # picking the source deme with the lowest p-value
                df = self.calc_contour(destid=dest[cnt-1], search_area='radius', sourceid=outliers_df['source'].iloc[outliers_df['pval'].argmin()], opts=opts, exclude_boundary=exclude_boundary)
            else:
                df = self.calc_contour(destid=dest[cnt-1], search_area=search_area, exclude_boundary=exclude_boundary)
                
            
            joint_df = self.calc_joint_contour(contour_df=df, top=top, lamb=lamb, lamb_q=lamb_q, optimize_q=optimize_q, usew=usew, uses2=uses2, exclude_boundary=exclude_boundary)

            # replacing inf with nan
            joint_df = joint_df.replace([np.inf, -np.inf], np.nan)

            print('\n  MLE edge found from source {:d} to destination {:d} with strength {:.2f}'.format(joint_df['(source, dest.)'].iloc[np.nanargmax(joint_df['log-lik'])][0], dest[cnt-1], joint_df['admix. prop.'].iloc[np.nanargmax(joint_df['log-lik'])]))
            print('\n  Log-likelihood after adding MLE edge: {:.1f} (p-val = {:.2e})\n'.format(np.nanmax(joint_df['log-lik']),chi2.sf(2*(np.nanmax(joint_df['log-lik'])-nllnull),df=1)))

            args['edge'] = [joint_df['(source, dest.)'].iloc[np.nanargmax(joint_df['log-lik'])]]; args['mode'] = 'update'
            obj.eems_neg_log_lik([joint_df['admix. prop.'].iloc[np.nanargmax(joint_df['log-lik'])]], opts=args)

            res_dist = np.array(cov_to_dist(-0.5*args['delta'])[np.tril_indices(self.n_observed_nodes, k=-1)])
            
            results[cnt] = {'deme': dest[cnt-1], 
                           'contour_df': df,
                           'joint_contour_df': joint_df, 
                           'log-lik': np.nanmax(joint_df['log-lik']),
                           'mle_w': self.w,
                           'mle_s2': self.s2,
                           'fit_dist': res_dist,
                           'pval': chi2.sf(2*(-np.nanmax(joint_df['log-lik'])-nllnull), df=1)}
            cnt += 1

        print("Exiting independent fitting algorithm after adding {:d} edge(s).".format(cnt-1))
        
        return results
            
    def sequential_fit(
        self,
        outliers_df, 
        lamb,
        lamb_q,
        optimize_q='n-dim',
        fdr=0.3, 
        stop=5,
        pval=0.05,
        top=0.05,
        exclude_boundary=True,
        maxls=50,
        m=10,
        factr=1e7,
        lb=-np.inf,
        ub=np.inf,
        maxiter=15000,
        search_area='all',
        opts=None
    ):
        """Function to iteratively fit a long range gene flow event to the graph until there are no more outliers (`alternate method`).
        Required:
            outliers_df (:obj:`pandas.DataFrame`): outlier DataFrame as output by the sp_graph.extract_outliers() function
            lamb (:obj:`float`): penalty strength on weights
            lamb_q (:obj:`float`): penalty strength on the residual variances
            
        Optional:
            optimize_q (:obj:'str'): indicator for method of optimizing residual variances (one of 'n-dim', '1-dim' or None)
            fdr (:obj:`float`): false-discovery rate of outlier edges 
            pval (:obj:`float`): p-value for assessing whether adding an admixture edge significantly increases log-likelihood over previous fit
            stop (:obj:`int`): number of admixture edges to add sequentially 
            exclude_boundary (:obj:`Bool`): whether to exclude boundary nodes in fitting procedure
            alpha (:obj:`float`): penalty strength on log weights
            lamb_q (:obj:`float`): penalty strength on the residual variances
            alpha_q (:obj:`float`): penalty strength on log residual variances
            factr (:obj:`float`): tolerance for convergence
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log weights
            ub (:obj:`int`): upper bound of log weights
            maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
            verbose (:obj:`Bool`): boolean to print summary of results
            

        Returns: 
            (:obj:`pandas.DataFrame`)
        """

        # check inputs
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float, "lambda must be float"
        assert lamb_q >= 0.0, "lambda must be non-negative"
        assert type(lamb_q) == float, "lambda must be float"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"
        
        obj = Objective(self)
        obj.inv(); obj.grad(reg=False); obj.Linv_diag = obj._comp_diag_pinv()

        # resetting the long-range edge counter
        self.edge = []
        
        # dict storing all the results for plotting
        results = {}

        # store the deme id of each consecutive maximum outlier
        destid = []; nll = []

        # passing in dummy variables just to initialize the procedure
        args = {'edge':[], 'mode':'update'}
        nll.append(obj.eems_neg_log_lik(None , args))
        print('Log-likelihood of initial fit: {:.1f}\n'.format(-nll[-1]))
            
        # choice to pick the deme with the largest number of implicated outliers
        concat = pd.concat([outliers_df['dest.']]).value_counts()
        print('Deme ID and # of times it was implicated as an outlier:')
        print(concat.iloc[:5])

        # resolving ties if any...
        maxidx = outliers_df['dest.'].value_counts()[outliers_df['dest.'].value_counts() == outliers_df['dest.'].value_counts().max()].index.tolist()
        if len(maxidx) > 1:
            # TODO add the deme with a lower p-value first
            print("Multiple putative outlier demes ({:}) found, but only adding largest outlier deme {:d}".format(maxidx, maxidx[0]))
        destid.append(maxidx[0])

        fit_cov, _, emp_cov = comp_mats(obj)
        fit_dist = cov_to_dist(fit_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]
        emp_dist = cov_to_dist(emp_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]

        results[0] = {'log-lik': -nll[-1], 
                     'emp_dist': emp_dist,
                     'fit_dist': fit_dist,
                     'outliers_df': outliers_df,
                     'fdr': fdr}

        cnt = 1; keepgoing = True
        while keepgoing and cnt <= stop:
            print('\nFitting admixture edge to deme {:d}:'.format(destid[-1]))
            # fit the contour on the deme to get the log-lik surface across the landscape
            if search_area=='radius':
                # picking the source deme with the lowest p-value
                df = self.calc_contour(destid=int(destid[-1]), search_area='radius', sourceid=outliers_df['source'].iloc[outliers_df['pval'].argmin()], opts=opts, args=args, exclude_boundary=exclude_boundary)
            else:
                df = self.calc_contour(destid=int(destid[-1]), search_area=search_area, args=args, exclude_boundary=exclude_boundary)

            usew = deepcopy(obj.sp_graph.w); uses2 = deepcopy(obj.sp_graph.s2)
            joint_df = self.calc_joint_contour(contour_df=df, top=top, lamb=lamb, lamb_q=lamb_q, optimize_q=optimize_q, usew=usew, uses2=uses2, exclude_boundary=exclude_boundary)
            # print(obj.eems_neg_log_lik())

            # only change the last element
            self.edge[-1] = joint_df['(source, dest.)'].iloc[np.nanargmax(joint_df['log-lik'])]
            # save the whole array of c vals
            self.c = joint_df['prev. c'].iloc[np.nanargmax(joint_df['log-lik'])] + [joint_df['admix. prop.'].iloc[np.nanargmax(joint_df['log-lik'])]]

            nll.append(-np.nanmax(joint_df['log-lik']))
            print('\n  Log-likelihood after fitting deme {:d}: {:.1f}'.format(destid[-1], -nll[-1]))

            if chi2.sf(2*(nll[0]-nll[-1]), df=len(nll)-1) > pval: 
                print("Previous edge to deme {:d} did not significantly increase the log-likelihood at a p-value of {:g}\n".format(destid[-1],pval))
                keepgoing=False
                break
            else:
                print("Previous edge to deme {:d} significantly increased the log-likelihood.\n".format(destid[-1]))

            args['edge'] = self.edge; args['mode'] = 'update'
            obj.eems_neg_log_lik(self.c, args)
            res_dist = np.array(cov_to_dist(-0.5*args['delta'])[np.tril_indices(self.n_observed_nodes, k=-1)])

            # function to obtain outlier indices given two pairwise distances 
            outliers_df = self.extract_outliers(fdr=fdr, res_dist=res_dist, verbose=False)

            results[cnt] = {'deme': destid[-1], 
                           'contour_df': df,
                           'joint_contour_df': joint_df, 
                           'log-lik': -nll[-1],
                           'fit_dist': res_dist,
                           'mle_w': self.w,
                           'mle_s2': self.s2,
                           'outliers_df': outliers_df, 
                           # 'pval': chi2.sf(2*(nll[-2]-nll[-1]), df=1)}
                            'pval': chi2.sf(2*(nll[0]-nll[-1]), df=len(nll)-1)}

            if outliers_df is None:
                keepgoing = False
            else:
                maxidx = outliers_df['dest.'].value_counts()[outliers_df['dest.'].value_counts() == outliers_df['dest.'].value_counts().max()].index.tolist()
                print('Deme ID and # of times it was implicated as an outlier:')
                print(pd.concat([outliers_df['dest.']]).value_counts().iloc[:5])
                if len(maxidx) > 1:        
                    print("Multiple putative outlier demes ({:}) found, but only adding largest outlier deme {:d}".format(maxidx, maxidx[0]))
                destid.append(maxidx[0])

            cnt += 1

        print("Exiting sequential fitting algorithm after adding {:d} edge(s).".format(cnt-1))

        return results          
    
    def fit(
        self,
        lamb,
        w_init=None,
        s2_init=None,
        alpha=None,
        lamb_q=None, 
        alpha_q=None,
        optimize_q='n-dim',        
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=False,
        option='default',
        long_range_edges=None
    ):
        """Estimates the edge weights of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.

        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            lamb_q (:obj:`float`): penalty strength on the residual variances
            alpha_q (:obj:`float`): penalty strength on log residual variances
            factr (:obj:`float`): tolerance for convergence
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log weights
            ub (:obj:`int`): upper bound of log weights
            maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
            verbose (:obj:`Bool`): boolean to print summary of results

        Returns:
            None
        """
        # check inputs
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float, "lambda must be float"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"

        # creating a container to store these edges 
        if long_range_edges is not None:
            self.edge = long_range_edges

        self.optimize_q = optimize_q
        self.option = option

        if self.option == 'default':
            # init from null model if no init weights are provided
            if w_init is None and s2_init is None:
                # fit null model to estimate the residual variance and init weights
                self.fit_null_model(verbose=verbose)              
                w_init = self.w0
            else:
                # check initial edge weights
                assert w_init.shape == self.w.shape, (
                    "weights must have shape of edges"
                )
                assert np.all(w_init > 0.0), "weights must be non-negative"
                self.w0 = w_init
                self.comp_precision(s2=s2_init)

            # prefix alpha if not provided
            if alpha is None:
                alpha = 1.0 / self.w0.mean()
            else:
                assert type(alpha) == float, "alpha must be float"
                assert alpha >= 0.0, "alpha must be non-negative"

            if lamb_q is None:
                lamb_q = lamb
            if alpha_q is None:
                alpha_q = 1. / self.s2.mean()

            # run l-bfgs
            obj = Objective(self)
            obj.sp_graph.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            
            x0 = np.log(w_init)
            if obj.sp_graph.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            s2_init = np.array([self.s2]) if obj.sp_graph.optimize_q=="1-dim" else self.s2*np.ones(len(self))
            if obj.sp_graph.optimize_q is not None:
                x0 = np.r_[np.log(w_init), np.log(s2_init)]
            else:
                x0 = np.log(w_init)

            res = fmin_l_bfgs_b(
                func=loss_wrapper,
                x0=x0,
                args=[obj],
                factr=factr,
                m=m,
                maxls=maxls,
                maxiter=maxiter,
                approx_grad=False,
            )
        else: 
            if alpha is None:
                alpha = 1.0 / self.w.mean()
            else:
                assert type(alpha) == float, "alpha must be float"
                assert alpha >= 0.0, "alpha must be non-negative"

            if lamb_q is None:
                lamb_q = lamb
            if alpha_q is None:
                alpha_q = 1.0 / self.s2.mean()

            obj = Objective(self)
            obj.sp_graph.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            if obj.sp_graph.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q

            obj.inv(); obj.grad(reg=False)
            res = coordinate_descent(
                obj=obj,
                factr=factr,
                m=m,
                maxls=maxls,
                maxiter=maxiter,
                verbose=verbose
            )


        if res is not None:
            if obj.sp_graph.optimize_q is not None:
                self.w = np.exp(res[0][:self.size()])
                self.s2 = np.exp(res[0][self.size():])
                self.comp_graph_laplacian(self.w)
                self.comp_precision(s2=self.s2)
                
                obj.inv(); obj.grad(reg=False)
                obj.Linv_diag = obj._comp_diag_pinv()
    
                # interpolation scheme using Kriging
                Rmatdo = -2 * obj.Linv[self.n_observed_nodes:, :self.n_observed_nodes] + obj.Linv[:self.n_observed_nodes, :self.n_observed_nodes].diagonal() + obj.Linv_diag[self.n_observed_nodes:, np.newaxis]
                Rmatoo = -2*obj.Linv[:self.n_observed_nodes, :self.n_observed_nodes] + np.broadcast_to(np.diag(obj.Linv),(self.n_observed_nodes, self.n_observed_nodes)).T + np.broadcast_to(np.diag(obj.Linv), (self.n_observed_nodes, self.n_observed_nodes))
            
                self.q_prox = interpolate_q(1/self.q, Rmatdo, Rmatoo)
            else:    
                self.w = np.exp(res[0])
                
            # print update
            self.train_loss, _ = loss_wrapper(res[0], obj)
            if verbose:
                sys.stdout.write(
                    (
                        "lambda={:.3f}, "
                        "alpha={:.4f}, "
                        "converged in {} iterations, "
                        "train_loss={:.3f}\n"
                    ).format(lamb, alpha, res[2]["nit"], self.train_loss)
                ) 

    def extract_outliers(
        self, 
        fdr=0.25, 
        tol=2,
        res_dist=None, 
        verbose=False
    ):
        """Function to extract outlier deme pairs based on a FDR threshold specified by the user. 
        
        Required: 
            fdr (:obj:`float`): FDR control rate, a number between 0 & 1
            
        Returns:
            (:obj:`pandas.DataFrame`)
        """
        
        assert fdr>0 and fdr<1, "fdr should be a positive number between 0 and 1"

        obj = Objective(self)
        # computing pairwise covariance & distances between demes
        fit_cov, _, emp_cov = comp_mats(obj)
        emp_dist = cov_to_dist(emp_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]
        if res_dist is None:
            fit_dist = cov_to_dist(fit_cov)[np.tril_indices(self.n_observed_nodes, k=-1)]
        else: 
            fit_dist = deepcopy(res_dist)

        # print('Using a significance threshold of {:g}:\n'.format(pthresh))
        print('Using a FDR of {:g}: '.format(fdr), end='')
        ls = []; x, y = [], []
        acs = np.empty((2, self.n_snps, 2))
        # computing p-values (or z-values) for each pairwise comparison after mean centering
        pvals = norm.cdf(np.log(emp_dist)-np.log(fit_dist)-np.mean(np.log(emp_dist)-np.log(fit_dist)), 0, np.std(np.log(emp_dist)-np.log(fit_dist),ddof=1))
        
        bh = benjamini_hochberg(emp_dist, fit_dist, fdr=fdr)
                
        for k in np.where(bh)[0]:
            # code to convert single index to matrix indices
            x.append(np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1); y.append(int(k - 0.5*x[-1]*(x[-1]-1)))

            # Gi = self.genotypes[self.nodes[self.perm_idx[x[-1]]]['sample_idx'], :]
            # Gi = self.genotypes[self.nodes[self.perm_idx[x[-1]]]['sample_idx'], :].astype('int').T
            # Ga1 = np.zeros((Gi.shape[0], Gi.shape[1], 2), dtype=int)
            # Ga1[Gi == 1, 1] = 1
            # Ga1[Gi == 2] = 1

            # Gi = self.genotypes[self.nodes[self.perm_idx[y[-1]]]['sample_idx'], :].astype('int').T
            # Ga2 = np.zeros((Gi.shape[0], Gi.shape[1], 2), dtype=int)
            # Ga2[Gi == 1, 1] = 1
            # Ga2[Gi == 2] = 1

            # fst = allel.average_hudson_fst(allel.GenotypeArray(Ga1).count_alleles(), allel.GenotypeArray(Ga2).count_alleles(), blen=int(self.genotypes.shape[1]/10))[0]

            ls.append([self.perm_idx[x[-1]], self.perm_idx[y[-1]], tuple(self.nodes[self.perm_idx[x[-1]]]['pos'][::-1]), tuple(self.nodes[self.perm_idx[y[-1]]]['pos'][::-1]), pvals[k], emp_dist[k]-fit_dist[k]])

        rm = []
        newls = []
        for k in range(len(ls)):
            # checking the log-lik of fits with deme1 - deme2 to find the source & dest.
            resc = minimize(obj.eems_neg_log_lik, x0=np.random.random(), args={'edge':[(ls[k][0],ls[k][1])],'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)])
            rescopp = minimize(obj.eems_neg_log_lik, x0=np.random.random(), args={'edge':[(ls[k][1],ls[k][0])],'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)])
            
            if resc.x<1e-3 and rescopp.x<1e-3:
                rm.append(k)
            else:
                # approximately similar likelihood of either deme being destination 
                if np.abs(rescopp.fun - resc.fun) <= tol:
                    newls.append([self.perm_idx[y[k]], self.perm_idx[x[k]], tuple(self.nodes[self.perm_idx[y[k]]]['pos'][::-1]), tuple(self.nodes[self.perm_idx[x[k]]]['pos'][::-1]), pvals[k], emp_dist[k]-fit_dist[k]])
                else:
                    # if the "opposite" direction has a much higher log-likelihood then replace it entirely 
                    if rescopp.fun < resc.fun:
                        ls[k][0] = self.perm_idx[y[k]]
                        ls[k][1] = self.perm_idx[x[k]]

        ls += newls

        # print(np.array(ls), np.array(ls).shape)
        df = pd.DataFrame(ls, columns = ['source', 'dest.', 'source (lat., long.)', 'dest. (lat., long.)', 'pval', 'raw diff.'])

        if len(df)==0:
            print('No outliers found.')
            print('Consider raising the FDR threshold slightly.')
            return None
        else:
            print('{:d} outlier deme pairs found'.format(len(df)))
            if verbose:
                print(df.sort_values(by='pval').to_string(index=False))
                print('Putative destination demes (and # of times the deme appears as an outlier) experiencing admixture: ')
                print(df['dest.'].value_counts())
            else:
                b, c = np.unique(df['dest.'], return_counts=True)
                print('Putative destination demes: {}'.format(b[np.argsort(-c)]))
            return df.sort_values('pval', ascending=True)

    def calc_joint_contour(
        self,  
        lamb, 
        lamb_q,
        optimize_q='n-dim',
        contour_df=None,
        top=0.05, 
        destid=None, search_area='all', sourceid=None, opts=None, 
        exclude_boundary=True, 
        usew=None, uses2=None
    ):
        """Function to calculate admix. prop. values in a joint manner with weights w & deme-specific variance s2 (as opposed to just admix. prop. values in `calc_contour`).

        Required:
            contour_df (:obj:`pd.DataFrame`) : data frame containing the output from the function `calc_contour` 
            top (:obj:`float`) : how many top entries (based on log-lik) to consider for the joint fitting? (if top >= 1, then it is the number of top entries, but if top < 1 then it is the top percent of total entries to consider)
            NOTE: if the above two flags are specified, then none of the flags below need to be specified. 
            destid (:obj:`int`) : ID of the putative destination deme  
            The flag coverage is used to signifiy how large the contour should be:
                'all'    - include all demes from the entire graph
                'radius' - include all demes within a certain radius of a user-specified sampled source deme 
                    - sourceid : integer ID of a sampled deme as seen on a FEEMS map
                    - opts : integer specifying radius (as an `int`) around the sampled source deme
                'range'  - include all demes within a certain long. & lat. rectangle 
                    - opts : list of lists specifying long. & lat. limits (e.g., [[-120,-70],[25,50]] for contiguous USA)
                'custom' - specific array of deme ids
                    - opts : list of specific deme ids as index

        Returns:
            None
        """

        # TODO potentially pass in obj as an object (low priority)
        obj = Objective(self); obj.inv(); obj.grad(reg=False)
        obj.Linv_diag = obj._comp_diag_pinv()

        # TODO check if this scenario works BUT first check if this scenario is even necessary
        if contour_df is None:
            assert isinstance(lamb, (float,)) and lamb >= 0, "lamb must be a float >=0"
            assert isinstance(lamb_q, (float,)) and lamb_q >= 0, "lamb_q must be a float >= 0"

            assert isinstance(destid, (int,)), "destid must be an integer"
            try:
                destpid = np.where(self.perm_idx[:self.n_observed_nodes]==destid)[0][0]
            except:
                print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
                return None
            # creating a list of (source, dest.) pairings based on user-picked criteria
            if search_area == 'all':
                # including every possible node in graph as a putative source
                randedge = [(x,destid) for x in list(set(range(self.number_of_nodes()))-set([destid]+list(self.neighbors(destid))))]
            elif search_area == 'radius':
                assert isinstance(sourceid, (int,)), "sourceid must be an integer"
                assert isinstance(opts, (int,)) and opts > 0, "radius must be an integer >=1"
                
                neighs = [] 
                neighs = list(self.neighbors(sourceid)) + [sourceid]
    
                # including all nodes within a certain radius
                for _ in range(opts-1):
                    tempn = [list(self.neighbors(n1)) for n1 in neighs]
                    # dropping repeated nodes 
                    neighs = np.unique(list(it.chain(*tempn)))
    
                randedge = [(x,destid) for x in list(set(neighs)-set([destid]+list(self.neighbors(destid))))]
            elif search_area == 'range':
                assert len(opts) == 2, "limits must be list of length 2"
                # reverse coordinates if in Western and Southern hemispheres
                if opts[0][0] > opts[0][1]:
                    opts[0] = opts[0][::-1]
                elif opts[1][0] > opts[1][1]:
                    opts[1] = opts[1][::-1]
                elif opts[0][0] > opts[0][1] & opts[1][0] > opts[1][1]:
                    opts[0] = opts[0][::-1]
                    opts[1] = opts[1][::-1]          
                randedge = []
                for n in range(self.number_of_nodes()):
                    # checking for lat. & long. of all possible nodes in graph
                    if self.nodes[n]['pos'][0] > opts[0][0] and self.nodes[n]['pos'][0] < opts[0][1]:
                        if self.nodes[n]['pos'][1] > opts[1][0] and self.nodes[n]['pos'][1] < opts[1][1]:
                            randedge.append((n,destid))
    
                # remove tuple of dest -> dest ONLY if it is in randedge
                if (destid,destid) in randedge:
                    randedge.remove((destid,destid))
            elif search_area == 'custom':
                randedge = [(x,destid) for x in list(set(opts)-set([destid]+list(self.neighbors(destid))))]
    
            if exclude_boundary:
                randedge = [(e[0], e[1]) for e in randedge if sum(1 for _ in self.neighbors(e[0]))==6]

            # fit the baseline graph if no w or s2 is passed in 
            # baseline w and s2 to be stored in an object
            if usew is None:
                # self.fit(lamb=lamb, optimize_q=optimize_q, lamb_q=lamb_q, option='default', verbose=False);
                # obj.inv(); obj.grad(reg=False)
                usew = deepcopy(self.w); uses2 = deepcopy(self.s2) 
                baselinell = -obj.eems_neg_log_lik(None, {'mode':'compute'})
                # container for storing the MLE weights & s2
                mlew = deepcopy(usew); mles2 = deepcopy(uses2)
            else:
                self._update_graph(usew, uses2)
                baselinell = -obj.eems_neg_log_lik(None, {'mode':'compute'})
                # container for storing the MLE weights & s2
                mlew = deepcopy(usew); mles2 = deepcopy(uses2)            
            
            cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
            prevc = []

            curedge = deepcopy(self.edge)

            for ie, e in enumerate(randedge):

                print("\r\tOptimizing joint likelihood over {}/{} most likely demes in the graph".format(ie+1,len(randedge)), end="")
                    
                # initializing at baseline values
                self._update_graph(usew, uses2)

                try:
                    self.fit(lamb=lamb, optimize_q=optimize_q, lamb_q=lamb_q, long_range_edges=curedge.append(e), option='onlyc', verbose=False)
                    cest2[ie] = self.c[-1]
                    prevc.append(self.c[:-1])
                    llc2[ie] = -obj.eems_neg_log_lik(self.c, {'edge':curedge.append(e),'mode':'compute'})
                    # print(llc2[ie],randedge[ie])
                    # updating the MLE weights if the new log-lik is higher than the previous one (if not, keep the previous values)
                    if llc2[ie] != -np.inf:
                        if llc2[ie] > np.nanmax(np.append(llc2[:ie],[baselinell])):
                            mlew = deepcopy(self.w); mles2 = deepcopy(self.s2)
                except: 
                    # TODO check if this try block is actually required
                    try:
                        # initializing at baseline values
                        self._update_graph(mlew, mles2)

                        # optimizing over multiple edges
                        self.fit(lamb=lamb, optimize_q=optimize_q, lamb_q=lamb_q, long_range_edges=curedge.append(e), option='onlyc', verbose=False)
                        cest2[ie] = self.c[-1]
                        prevc.append(self.c[:-1])
                        llc2[ie] = -obj.eems_neg_log_lik(self.c, {'edge':curedge.append(e),'mode':'compute'})
                    
                        # updating the MLE weights if the new log-lik is higher than the previous one (if not, keep the previous values)
                        if llc2[ie] != -np.inf:
                            if llc2[ie] > np.nanmax(np.append(llc2[:ie],[baselinell])):
                                mlew = deepcopy(self.w); mles2 = deepcopy(self.s2)
                    except:
                        cest2[ie] = np.nan
                        llc2[ie] = np.nan
    
            print('...done!')

            df = pd.DataFrame(index=range(1,len(randedge)+1), columns=['(source, dest.)', 'admix. prop.', 'log-lik', 'scaled log-lik'])
            df['(source, dest.)'] = randedge; df['admix. prop.'] = cest2; df['log-lik'] = llc2
            df['prev. c'] = prevc
            
            # if MLE is found to be on the edge of the range specified by user then indicate that range should be extended
            if search_area == 'radius' or search_area == 'range':
                mles = df['(source, dest.)'].iloc[np.nanargmax(df['log-lik'])][0]
                if len(list(self.neighbors(mles))) < 6:
                    print("(Warning: MLE location of source found to be at the edge of the specified {:g}, consider increasing the `opts` to include a larger area.)".format(search_area))
    
            if np.sum(df['log-lik'].isna()) > 0.25*len(df):
                print("(Warning: log-likelihood could not be computed for ~{:.0f}% of demes. Try increasing value of lamb)".format(np.sum(df['log-lik'].isna())*100/len(df)))

            joint_contour_df = df
        else:
            assert isinstance(lamb, (float,)) and lamb >= 0, "lamb must be a float >=0"
            assert isinstance(lamb_q, (float,)) and lamb_q >= 0, "lamb_q must be a float >= 0"
            # get indices of the top hits
            if top<1:
                # treat as a percentage
                topidx = contour_df['log-lik'].nlargest(int(np.ceil(top * len(contour_df)))).index
            else: 
                # treat as a number 
                topidx = contour_df['log-lik'].nlargest(int(top)).index
            # print("Jointly optimizing likelihood over {:d} demes in the graph:\n".format(len(topidx)))

            if usew is None:                
                baselinell = -obj.eems_neg_log_lik(None, {'mode':'compute'})
                usew = deepcopy(self.w); uses2 = deepcopy(self.s2)
                mlew = deepcopy(usew); mles2 = deepcopy(uses2) 
            else:
                self._update_graph(usew, uses2)
                mlew = deepcopy(usew); mles2 = deepcopy(uses2) 
                baselinell = -obj.eems_neg_log_lik(None, {'mode':'compute'})
            
            # run the joint fitting scheme for each top hit
            joint_contour_df = contour_df.loc[topidx]; cnt = 1

            # add extra column to store previous c values
            joint_contour_df['prev. c'] = None

            curedge = deepcopy(self.edge)
            print(curedge)
            
            for i, row in joint_contour_df.iterrows():
                print("\r\tOptimizing joint likelihood over {}/{} most likely demes in the graph".format(cnt,len(topidx)), end="")

                # update counter (cos apparently iterrows() passes index back)
                cnt += 1
                
                # initializing at baseline values
                self._update_graph(usew, uses2)

                try:
                    # TODO verbose print message here?
                    self.fit(lamb=lamb, optimize_q=optimize_q, lamb_q=lamb_q, long_range_edges=curedge + [row['(source, dest.)']], option='onlyc', verbose=False)
                    
                    joint_contour_df.at[i, 'admix. prop.'] = self.c[-1]
                    # also track the estimates for c for pre-existing long-range edges
                    joint_contour_df.at[i, 'prev. c'] = list(self.c[:-1])
                    # TODO keep a rolling (hidden?) variable for the log-likelihood under each fit
                    joint_contour_df.at[i, 'log-lik'] = -obj.eems_neg_log_lik(self.c, {'edge':curedge + [row['(source, dest.)']],'mode':'compute'})

                    # updating the MLE weights if the new log-lik is higher than the previous one (if not, keep the previous values)
                    if joint_contour_df.index.get_loc(i) == 0:
                        mlew = deepcopy(self.w); mles2 = deepcopy(self.s2)
                    else:
                        if joint_contour_df.at[i, 'log-lik'] > np.nanmax(joint_contour_df['log-lik'].iloc[:joint_contour_df.index.get_loc(i)]):
                            mlew = deepcopy(self.w); mles2 = deepcopy(self.s2)
                except:  
                    joint_contour_df.at[i, 'admix. prop.'] = np.nan
                    joint_contour_df.at[i, 'log-lik'] = np.nan

            print("...done!")
    
            if np.sum(joint_contour_df['log-lik'].isna()) > 0.25*len(joint_contour_df):
                print("(Warning: log-likelihood could not be computed for ~{:.0f}% of demes. Try increasing value of lamb)".format(np.sum(joint_contour_df['log-lik'].isna())*100/len(joint_contour_df)))
                
        joint_contour_df['scaled log-lik'] = joint_contour_df['log-lik'] - np.nanmax(joint_contour_df['log-lik']) 

        # updating the graph with MLE weights so it does not need to be fit again
        self._update_graph(mlew, mles2)
        # print(obj.eems_neg_log_lik())

        # set bounds for c values
        joint_contour_df['admix. prop.'] = joint_contour_df['admix. prop.'].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)

        # only replace the edge if there is a single fit edge 
        if len(self.edge) == 1:
            self.edge = [joint_contour_df['(source, dest.)'].iloc[np.nanargmax(joint_contour_df['log-lik'])]]
            self.c = [joint_contour_df['admix. prop.'].iloc[np.nanargmax(joint_contour_df['log-lik'])]]

        # checking whether adding an extra admixture parameter improves model fit using a LRT
        joint_contour_df['pval'] = chi2.sf(2*(joint_contour_df['log-lik'] - baselinell), df=len(self.edge))
        
        return joint_contour_df

    def calc_contour(
        self, 
        destid, 
        search_area='all', 
        sourceid=None, 
        opts=None, 
        exclude_boundary=True, 
        args=None
    ):
        """
        Function to calculate admix. prop. values along with log-lik. values in a contour around the sampled source deme to capture uncertainty in the location of the source. 

        Required:
            destid (:obj:`int`) : ID of the putative destination deme             

        Optional:
            search_area (:obj:`str`): flag signifiy how large the search space should be for the source
                'all'    - include all demes from the entire graph
                'radius' - include all demes within a certain radius of a user-specified sampled source deme 
                    - sourceid : integer ID of a sampled deme as seen on a FEEMS map
                    - opts : integer specifying radius (as an `int`) around the sampled source deme
                'range'  - include all demes within a certain long. & lat. rectangle 
                    - opts : list of lists specifying long. & lat. limits (e.g., [[-120,-70],[25,50]] for contiguous USA)
                'custom' - specific array of deme ids
                    - opts : list of specific deme ids as index

        Returns: 
            (:obj:`pandas.DataFrame`)
        """
        obj = Objective(self)
        obj.inv(); obj.grad(reg=False); obj.Linv_diag = obj._comp_diag_pinv()

        # TODO include np.int 
        assert isinstance(destid, (numbers.Integral)), "destid must be an integer"

        try:
            destpid = np.where(self.perm_idx[:self.n_observed_nodes]==destid)[0][0] #-> 0:(o-1)
        except:
            print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
            return None

        # creating a list of (source, dest.) pairings based on user-picked criteria
        if search_area == 'all':
            # including every possible node in graph as a putative source
            randedge = [(x,destid) for x in list(set(range(self.number_of_nodes()))-set([destid]+list(self.neighbors(destid))))]
        elif search_area == 'radius':
            assert isinstance(sourceid, (int,)), "sourceid must be an integer"
            assert isinstance(opts, (int,)) and opts > 0, "radius must be an integer >=1"
            
            neighs = [] 
            neighs = list(self.neighbors(sourceid)) + [sourceid]

            # including all nodes within a certain radius
            for _ in range(opts-1):
                tempn = [list(self.neighbors(n1)) for n1 in neighs]
                # dropping repeated nodes 
                neighs = np.unique(list(it.chain(*tempn)))

            randedge = [(x,destid) for x in list(set(neighs)-set([destid]+list(self.neighbors(destid))))]
        elif search_area == 'range':
            assert len(opts) == 2, "limits must be list of length 2 (e.g., [[-120,-70],[25,50]])"
            # reverse coordinates if in Western and Southern hemispheres
            if opts[0][0] > opts[0][1]:
                opts[0] = opts[0][::-1]
            elif opts[1][0] > opts[1][1]:
                opts[1] = opts[1][::-1]
            elif opts[0][0] > opts[0][1] & opts[1][0] > opts[1][1]:
                opts[0] = opts[0][::-1]
                opts[1] = opts[1][::-1]          
            randedge = []
            for n in range(self.number_of_nodes()):
                # checking for lat. & long. of all possible nodes in graph
                if self.nodes[n]['pos'][0] > opts[0][0] and self.nodes[n]['pos'][0] < opts[0][1]:
                    if self.nodes[n]['pos'][1] > opts[1][0] and self.nodes[n]['pos'][1] < opts[1][1]:
                        randedge.append((n,destid))

            # remove tuple of dest -> dest ONLY if it is in randedge
            if (destid,destid) in randedge:
                randedge.remove((destid,destid))
        elif search_area == 'custom':
            randedge = [(x,destid) for x in list(set(opts)-set([destid]+list(self.neighbors(destid))))]

        # only include central demes (==6 neighbors), since demes on edge of range exhibit some boundary effects during estimation
        # randedge = list(it.compress(randedge,np.array([sum(1 for _ in self.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[i])) for i in list(set(range(self.number_of_nodes()))-set([destid]))])==6))
        if exclude_boundary:
            randedge = [(e[0], e[1]) for e in randedge if sum(1 for _ in self.neighbors(e[0]))==6]

        # just want to perturb it a bit instead of updating the entire matrix
        if args is None:
            args = {}
            args['mode'] = 'compute'
            # adding a dummy edge in since c=0 doesn't change any terms anyway
            args['delta'] = obj._compute_delta_matrix(None, {})
        else:
            args['mode'] = 'compute'
        
        # randpedge = []
        cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
        print("  Optimizing likelihood over {:d} demes in the graph".format(len(randedge)),end='...')
        checkpoints = {int(np.percentile(range(len(self.perm_idx)),25)): 25, int(np.percentile(range(len(self.perm_idx)),50)): 50, int(np.percentile(range(len(self.perm_idx)),75)): 75}
        for ie, e in enumerate(randedge):

            if ie in checkpoints:
                print('{:d}%'.format(checkpoints[ie]), end='...')
            
            # convert all sources to valid permuted ids (so observed demes should be b/w index 0 & o-1)
            # e2 = (np.where(self.perm_idx==e[0])[0][0], destpid) # -> contains the permuted ids, so 0:(o-1) is sampled (useful for indexing Linv & Lpinv)
            # randpedge.append((e[0],destid)) # -> contains the *un*permuted ids (useful for external viz)
            args['edge'] = [e]
            try:
                res = minimize(obj.eems_neg_log_lik, x0=np.random.random(), tol=1e-3, method='L-BFGS-B', args=args)
                cest2[ie] = res.x; llc2[ie] = res.fun
            except:
                cest2[ie] = np.nan; llc2[ie] = np.nan

        print('done!')
                
        df = pd.DataFrame(index=range(1,len(randedge)+1), columns=['(source, dest.)', 'admix. prop.', 'log-lik', 'scaled log-lik'])
        df['(source, dest.)'] = randedge; df['admix. prop.'] = cest2; df['log-lik'] = -llc2; df['scaled log-lik'] = df['log-lik']-np.nanmax(df['log-lik'])

        # if MLE is found to be on the edge of the range specified by user then indicate that range should be extended
        if search_area == 'radius' or search_area == 'range':
                mles = df['(source, dest.)'].iloc[np.nanargmax(df['log-lik'])][0]
                if len(list(self.neighbors(mles))) < 6:
                    print("  (Warning: MLE location of source found to be at the edge of the specified {}, consider increasing the `opts` to include a larger area and rerunning this command.)".format(search_area))

        df = df.replace([np.inf, -np.inf], np.nan)
        if np.sum(df['log-lik'].isna()) > 0.3*len(df):
            print("(Warning: log-likelihood could not be computed for ~{:.2f}% of demes)".format(np.sum(df['log-lik'].isna())*100/len(df)))

        df['admix. prop.'] = df['admix. prop.'].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)
        return df#.dropna() 

def coordinate_descent(
    obj, 
    factr=1e10, 
    m=10, 
    maxls=50, 
    maxiter=100,
    verbose=False
):
    """
    Minimize the negative log-likelihood iteratively with an admix. prop. c & (weights, s2) in a coordinate-descent manner until tolderance `atol` is reached. 
    """

    # flag to optimize admixture proportion
    optimc = True
    
    for bigiter in range(maxiter):
        
        # first fit admix. prop. c given the weights
        resc = minimize(obj.eems_neg_log_lik, x0=[0.1]*len(obj.sp_graph.edge), args={'edge':obj.sp_graph.edge,'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)]*len(obj.sp_graph.edge))

        if resc.status != 0:
            print('(Warning: admix. prop. optimization failed for deme {:d}, increase factr slightly or change optimization parameters)'.format(obj.sp_graph.edge[0][0]))
            return None

        if obj.sp_graph.c is not None:
            if len(obj.sp_graph.c) == len(resc.x):
                if np.allclose(resc.x, obj.sp_graph.c, atol=1e-3):
                    optimc = False

        obj.sp_graph.c = deepcopy(resc.x)

        if obj.sp_graph.optimize_q is not None:
            x0 = np.r_[np.log(obj.sp_graph.w), np.log(obj.sp_graph.s2)]
        else:
            x0 = np.log(obj.sp_graph.w)

        # then fit weights & s2 keeping c constant
        res = fmin_l_bfgs_b(
            func=loss_wrapper,
            x0=x0,
            args=[obj],
            factr=factr,
            m=m,
            maxls=maxls,
            maxiter=maxiter,
            approx_grad=False,
        )
        # print(res[2]['task'], res[2]['nit'], res[2]['warnflag'])
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge (increase maxiter or factr slightly)"
        if obj.sp_graph.optimize_q is not None:
            neww = np.exp(res[0][:obj.sp_graph.size()])
            news2 = np.exp(res[0][obj.sp_graph.size():])
            
            # difference in parameters for this step
            diffw = np.abs(np.exp(x0[:obj.sp_graph.size()]) - neww)
            diffs2 = np.abs(np.exp(x0[obj.sp_graph.size():]) - news2)
            # print(np.sum(diffw), np.sum(diffs2))
        else:
            neww = np.exp(res[0])
            news2 = obj.sp_graph.s2
            # difference in parameters for this step
            diffw = np.abs(np.exp(x0) - neww)
            diffs2 = [0]

        if np.allclose(diffw, np.zeros(len(diffw)), atol=100 * factr * np.finfo(float).eps) and np.allclose(diffs2, np.zeros(len(diffs2)), atol=100 * factr * np.finfo(float).eps) and not optimc:
            if verbose:
                print("Joint estimation converged in {:d} iterations!".format(bigiter+1))
            break

    return res

def query_node_attributes(graph, name):
    """Query the node attributes of a nx graph. This wraps get_node_attributes
    and returns an array of values for each node instead of the dict
    """
    d = nx.get_node_attributes(graph, name)
    arr = np.array(list(d.values()))
    return arr
