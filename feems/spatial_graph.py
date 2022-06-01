from __future__ import absolute_import, division, print_function

import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
from scipy.optimize import fmin_l_bfgs_b, minimize

from .objective import Objective, loss_wrapper, neg_log_lik_w0_s2


class SpatialGraph(nx.Graph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True, 
                long_range_edges=[(0,0)]):
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
            long_range_edges (:obj:`list`): list of 2-tuples with pairs of nodes
                Default is a self pointer to the first node
        """
        # check inputs
        assert len(genotypes.shape) == 2
        assert len(sample_pos.shape) == 2
        assert np.all(~np.isnan(genotypes)), "no missing genotypes are allowed"
        assert np.all(~np.isinf(genotypes)), "non inf genotypes are allowed"
        assert (
            genotypes.shape[0] == sample_pos.shape[0]
        ), "genotypes and sample positions must be the same size"
        assert type(long_range_edges) == list or long_range_edges is None, "long_range_edges should be a list of 2-tuples"

        # inherits from networkx Graph object -- changed this to new signature for python3
        super().__init__()
        self._init_graph(node_pos, edges)  # init graph

        # inputs
        self.sample_pos = sample_pos
        self.node_pos = node_pos
        self.scale_snps = scale_snps

        # creating a container to store these edges 
        self.lre = long_range_edges
        # mask for indices of edges in lre
        self.lre_idx = np.array([val in self.lre for val in list(self.edges)])

        # signed incidence_matrix
        self.Delta_q = nx.incidence_matrix(self, oriented=True).T.tocsc()

        # track nonzero edges upper triangular
        self.adj_base = sp.triu(nx.adjacency_matrix(self), k=1)
        self.nnz_idx = self.adj_base.nonzero()

        # adjacency matrix on the edges
        self.Delta = self._create_incidence_matrix()

        # vectorization operator on the edges
        self.diag_oper = self._create_vect_matrix()

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

        # initialize c (admixture proportions)
        # currently only one lre (otherwise need a vector here)
        self.c = 0.9995 # np.repeat(0.5, np.sum(self.lre_idx))

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
        data = np.array([], dtype=np.float)
        row_idx = np.array([], dtype=np.int)
        col_idx = np.array([], dtype=np.int)
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
        col_idx = np.array([], dtype=np.int)
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
            allele_counts = np.mean(self.genotypes[s, :], axis=0)
            self.frequencies[i, :] = allele_counts

    def comp_precision(self, s2):
        """Computes the residual precision matrix"""
        self.s2 = s2
        self.q = self.n_samples_per_obs_node_permuted / self.s2
        self.q_diag = sp.diags(self.q).tocsc()
        self.q_inv_diag = sp.diags(1.0 / self.q).tocsc()
        self.q_inv_grad = -1.0 / self.n_samples_per_obs_node_permuted

    # ------------------------- Optimizers -------------------------

    def fit_null_model(self, verbose=True):
        """Estimates of the edge weights and residual variance
        under the model that all the edge weights have the same value
        """
        obj = Objective(self)
        res = minimize(neg_log_lik_w0_s2, [0.0, 0.0], method="Nelder-Mead", args=(obj))
        assert res.success is True, "did not converge"
        w0_hat = np.exp(res.x[0])
        s2_hat = np.exp(res.x[1])
        self.w0 = w0_hat * np.ones(self.w.shape[0])
        self.s2 = s2_hat
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

    def fit(
        self,
        lamb,
        w_init=None,
        s2_init=None,
        alpha=None,
        beta=0.0, 
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
    ):
        """Estimates the edge weights of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.

        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            beta (:obj:`float`): penalty strength for long range weights
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
        assert type(lamb) == float, "lambda must be float"
        assert beta >= 0.0, "beta must be non-negative"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"

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

        # run l-bfgs
        obj = Objective(self)
        obj.lamb = lamb
        obj.alpha = alpha
        obj.beta = beta
        
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
            bounds=[(lb, ub) for _ in range(x0.shape[0])],
        )
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge"
        self.w = np.exp(res[0])

        # print update
        self.train_loss, _ = loss_wrapper(res[0], obj)
        if verbose:
            sys.stdout.write(
                (
                    "lambda={:.5f}, "
                    "alpha={:.5f}, "
                    "beta={:.5f}, "
                    "converged in {} iterations, "
                    "train_loss={:.5f}\n"
                ).format(lamb, alpha, beta, res[2]["nit"], self.train_loss)
            )


def query_node_attributes(graph, name):
    """Query the node attributes of a nx graph. This wraps get_node_attributes
    and returns an array of values for each node instead of the dict
    """
    d = nx.get_node_attributes(graph, name)
    arr = np.array(list(d.values()))
    return arr
