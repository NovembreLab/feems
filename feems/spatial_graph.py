from __future__ import absolute_import, division, print_function

import sys

import allel
from copy import deepcopy
import networkx as nx
import numpy as np
from scipy.linalg import pinvh
from scipy.optimize import fmin_l_bfgs_b, minimize
import scipy.sparse as sp
from scipy.stats import chi2
import sksparse.cholmod as cholmod
import pandas as pd

from .objective import Objective, loss_wrapper, neg_log_lik_w0_s2, comp_mats
from .utils import cov_to_dist

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

        # inherits from networkx Graph object -- changed this to new signature for python3
        print("initializing graph...")
        super().__init__()
        self._init_graph(node_pos, edges)  # init graph

        # inputs
        self.sample_pos = sample_pos
        self.node_pos = node_pos
        self.scale_snps = scale_snps
        self.option = 'default'

        self.optimize_q = None

        print("computing graph attributes...")
        # signed incidence_matrix
        self.Delta_q = nx.incidence_matrix(self, oriented=True).T.tocsc()

        # track nonzero edges upper triangular
        self.adj_base = sp.triu(nx.adjacency_matrix(self), k=1)
        self.nnz_idx = self.adj_base.nonzero()

        # adjacency matrix on the edges
        self.Delta = self._create_incidence_matrix()

        # vectorization operator on the edges
        self.diag_oper = self._create_vect_matrix()

        print("assigning samples to nodes", end="...")
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

        # estimate sample covariance matrix
        self.S = self.frequencies @ self.frequencies.T / self.n_snps

        # creating an internal index for easier access
        self.perm_idx = query_node_attributes(self, "permuted_idx") 

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
        """Computes the graph laplacian note this is computed each step of the
        optimization so needs to be fast
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

        Args:
            genotypes (:obj:`numpy.ndarray`): array of diploid genotypes with
                no missing data
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

    def sequential_fit(
        self, 
        fdr=0.3, 
        pval=0.05,
        stop=5,
        top=10,
        maxls=50,
        m=10,
        factr=1e7,
        lb=-1e10,
        ub=1e10,
        maxiter=15000,
        lamb=None,
        lamb_q=None,
        optimize_q='n-dim',
        search_area='all',
        opts=None
    ):
        """
        Function to iteratively fit a long range gene flow event to the graph until there are no more outliers (alternate method)
        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            optimize_q (:obj:'str'): indicator for method of optimizing residual variances (one of 'n-dim', '1-dim' or None)
            lamb_q (:obj:`float`): penalty strength on the residual variances
            alpha_q (:obj:`float`): penalty strength on log residual variances
            factr (:obj:`float`): tolerance for convergence
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log weights
            ub (:obj:`int`): upper bound of log weights
            maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
            verbose (:obj:`Bool`): boolean to print summary of results

            fdr (:obj:`float`): false-discovery rate of outlier edges 
            pval (:obj:`float`): p-value for assessing whether adding an admixture edge significantly increases log-likelihood over previous fit
            stop (:obj:`int`): number of admixture edges to add sequentially 
        """

        # check inputs
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float or type(lamb) == np.float64, "lambda must be float"
        assert lamb_q >= 0.0, "lambda must be non-negative"
        assert type(lamb_q) == float or type(lamb_q) == np.float64, "lambda must be float"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"
        
        # fit baseline graph if all weights are equal to 1
        if np.all(self.w == 1):
            self.fit(lamb = lamb, lamb_q = lamb_q, optimize_q = optimize_q) 

        obj = Objective(self)
        obj.inv()
        if not hasattr(obj, 'Lpinv'):
            obj.Lpinv = pinvh(obj.sp_graph.L.todense()) 
        obj.grad(reg=False)

        # dict storing all the results for plotting
        results = {}

        # store the deme id of each consecutive maximum outlier
        destid = []; nll = []

        # passing in dummy variables just to initialize the procedure
        args = {'edge':[(0,self.perm_idx[0])], 'mode':'update'}
        nll.append(obj.eems_neg_log_lik(0 , args))
        print('Log-likelihood of initial fit: {:.1f}\n'.format(-nll[-1]))

        # get the first round of outliers from the baseline fit
        outliers_df = obj.extract_outliers(fdr = fdr)

        if outliers_df is None:
            return None
            
        # choice to pick the deme with the largest number of implicated outliers
        concat = pd.concat([outliers_df['dest.']]).value_counts()
        print(concat.iloc[:5])

        # resolving ties if any...
        maxidx = outliers_df['dest.'].value_counts()[outliers_df['dest.'].value_counts() == outliers_df['dest.'].value_counts().max()].index.tolist()
        if len(maxidx) > 1:        
            print("Multiple putative outlier demes ({:}) found, but only adding deme {:d}".format(maxidx, maxidx[0]))
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
            print('\nFitting long range edge to deme {:d}...'.format(destid[-1]))
            # fit the contour on the deme to get the log-lik surface across the landscape
            if search_area=='radius':
                # picking the source deme with the lowest p-value
                df = obj.calc_contour(destid=int(destid[-1]), search_area='radius', sourceid=outliers_df['source'].iloc[outliers_df['pval'].argmin()], opts=opts, delta=args['delta'])
            else:
                df = obj.calc_contour(destid=int(destid[-1]), search_area=search_area, delta=args['delta'])
                
            usew = deepcopy(obj.sp_graph.w); uses2 = deepcopy(obj.sp_graph.s2)
            joint_df = obj.calc_joint_contour(df, top=top, lamb=lamb, lamb_q=lamb_q, optimize_q=optimize_q, usew=usew, uses2=uses2)
            # print(obj.eems_neg_log_lik())

            nll.append(-np.nanmax(joint_df['log-lik']))
            print('\nLog-likelihood after fitting deme {:d}: {:.1f}'.format(destid[-1], -nll[-1]))
            
            args['edge'] = [joint_df['(source, dest.)'].iloc[joint_df['log-lik'].argmax()]]; args['mode'] = 'update'
            obj.eems_neg_log_lik(c=joint_df['admix. prop.'].iloc[np.argmax(joint_df['log-lik'])], opts=args)
            # assert nll[-1] == obj.eems_neg_log_lik(c=joint_df['admix. prop.'].iloc[np.argmax(joint_df['log-lik'])], opts=opts), "difference in internal log-lik values (rerun the function)"
            # print('\nLog-likelihood after fitting deme {:d}: {:.1f}'.format(destid[-1], -nll[ -1]))

            if chi2.sf(2*(nll[-2]-nll[-1]), df=1) > pval: 
                print("Previous edge did not significantly increase the log-likelihood of the fit at a p-value of {:g}\n".format(pval))
                keepgoing=False
                break
            else:
                print("Previous edge to deme {:d} significantly increased the log-likelihood of the fit.\n".format(destid[-1]))

            res_dist = np.array(cov_to_dist(-0.5*args['delta'])[np.tril_indices(self.n_observed_nodes, k=-1)])

            # function to obtain outlier indices given two pairwise distances 
            outliers_df = obj.extract_outliers(fdr=fdr, res_dist=res_dist, verbose=False)

            results[cnt] = {'deme': destid[-1], 
                           'contour_df': df,
                           'joint_contour_df': joint_df, 
                           'log-lik': -nll[-1],
                           'fit_dist': res_dist,
                           'outliers_df': outliers_df, 
                           'pval': chi2.sf(2*(nll[-2]-nll[-1]), df=1)}

            if outliers_df is None:
                keepgoing = False
            else:
                maxidx = outliers_df['dest.'].value_counts()[outliers_df['dest.'].value_counts() == outliers_df['dest.'].value_counts().max()].index.tolist()
                print('Deme ID and # of times it was implicated as an outlier:')
                print(pd.concat([outliers_df['dest.']]).value_counts().iloc[:5])
                if len(maxidx) > 1:        
                    print("Multiple putative outlier demes ({:}) found, but only adding deme {:d}".format(maxidx, maxidx[0]))
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
        verbose=True,
        option='default',
        long_range_edges=[(0,1)]
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
        """
        # check inputs
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float or type(lamb) == np.float64, "lambda must be float"
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
        self.edge = long_range_edges

        self.c = np.random.random(len(self.edge))

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
            s2_init = self.s2 if obj.sp_graph.optimize_q=="1-dim" else self.s2*np.ones(len(self))
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
                alpha_q = 1. / self.s2.mean()

            obj = Objective(self)
            obj.sp_graph.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            if obj.sp_graph.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
                
            obj.inv(); obj.Lpinv = pinvh(self.L.todense()); 
            obj.grad(reg=False)
            res = coordinate_descent(
                obj=obj,
                factr=factr,
                m=m,
                maxls=maxls,
                maxiter=maxiter,
                verbose=verbose
            )

        if obj.sp_graph.optimize_q is not None:
            self.w = np.exp(res[0][:self.size()])
            self.s2 = np.exp(res[0][self.size():])
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

def coordinate_descent(
    obj, 
    factr=1e7, 
    m=10, 
    maxls=50, 
    maxiter=100,
    atol=1e-3,
    verbose=False
):
    """
    Minimize the negative log-likelihood iteratively with an admix. prop. c value & refit the new weights based on that until tolerance is reached. 
    """
    
    # obj.sp_graph.edge = edge
    # obj.sp_graph.option = 'onlyc'

    # flag to optimize admixture proportion
    optimc = True
    
    for bigiter in range(maxiter):
        if verbose:
            print(bigiter,end='...')
        
        # first fit admix. prop. c given the weights
        resc = minimize(obj.eems_neg_log_lik, x0=np.random.random(), args={'edge':obj.sp_graph.edge,'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)])
        # print(resc.x)
        # resc = minimize(obj.neg_log_lik_c, x0=np.log10(self.sp_graph.c/(1-self.sp_graph.c)), bounds=[(-3,3)], method='L-BFGS-B', args={'lre':obj.sp_graph.edge,'mode':'sampled'})
        # print(resc.x, obj.sp_graph.c)
        if resc.status != 0:
            print('Warning: admix. prop. optimization failed (increase atol or factr slightly)')
            return None
        if np.allclose(resc.x, obj.sp_graph.c, atol=atol):
            optimc = False

        obj.sp_graph.c = deepcopy(resc.x)
        # print(resc.x)
        # obj.sp_graph.c = deepcopy(10**resc.x/(1+10**resc.x))

        if obj.sp_graph.optimize_q is not None:
            x0 = np.r_[np.log(obj.sp_graph.w), np.log(obj.sp_graph.s2)]
        else:
            x0 = np.log(obj.sp_graph.w)

        # then fit weights & s2 keeping c constant
        res = fmin_l_bfgs_b(
            func=loss_wrapper,
            x0=x0,
            # bounds=[(-1e10,1e10) for _ in x0], #-> setting bounds on how far the value can be perturbed (produces Singular matrix errors so leaving it blank)
            args=[obj],
            factr=factr,
            m=m,
            maxls=maxls,
            maxiter=maxiter,
            approx_grad=False,
        )
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

        if np.allclose(diffw, np.zeros(len(diffw)), atol=1e-8) and np.allclose(diffs2, np.zeros(len(diffs2)), atol=1e-8) and not optimc:
            if verbose:
                print("joint estimation converged in {:d} iterations!".format(bigiter+1))
            break

    return res


def query_node_attributes(graph, name):
    """Query the node attributes of a nx graph. This wraps get_node_attributes
    and returns an array of values for each node instead of the dict
    """
    d = nx.get_node_attributes(graph, name)
    arr = np.array(list(d.values()))
    return arr
