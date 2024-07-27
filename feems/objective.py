from __future__ import absolute_import, division, print_function

import allel
from copy import deepcopy
import itertools as it
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import det, pinvh
from scipy.optimize import fmin_l_bfgs_b, minimize
import scipy.sparse as sp
from scipy.stats import wishart, norm, chi2
import statsmodels.api as sm

from .utils import cov_to_dist, benjamini_hochberg, get_outlier_idx

class Objective(object):
    def __init__(self, sp_graph):
        """Evaluations and gradient of the feems objective function

        Args:
            sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
        """
        # spatial graph
        self.sp_graph = sp_graph

        # reg params
        self.lamb = None
        self.alpha = None
        self.lamb_q = None
        self.alpha_q = None

        self.nll = 0.0

        self.C = np.vstack((-np.ones(self.sp_graph.n_observed_nodes-1), np.eye(self.sp_graph.n_observed_nodes-1))).T
        
        # genetic distance matrix
        self.sp_graph.D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        self.CDCt = self.C @ self.sp_graph.D @ self.C.T

    def _rank_one_solver(self, B):
        """Solver for linear system (L_{d-o,d-o} + ones/d) * X = B using rank
        ones update equation
        """
        # dims
        d = len(self.sp_graph)
        o = self.sp_graph.n_observed_nodes

        # vector of ones with size d-o
        ones = np.ones(d - o)

        # sparse cholesky factorization
        # solve the systems
        # L_block{dd}\B
        # TODO: how to handle when B is sparse
        U = self.sp_graph.factor(B)

        # L_block{dd}\ones
        v = self.sp_graph.factor(ones)

        # denominator
        denom = d + np.sum(v)
        X = U - np.outer(v, v @ B) / denom

        return (X, v, denom)

    def _solve_lap_sys(self):
        """Solve (L_{d-o,d-o} + ones/d) * X = L_{d-o,o} + ones/d using rank one
        solver
        """
        o = self.sp_graph.n_observed_nodes
        d = len(self.sp_graph)

        # set B = L_{d-o,o}
        B = self.sp_graph.L_block["do"]

        # solve (L_{d-o,d-o} + ones/d) \ B
        self.lap_sol, v, denom = self._rank_one_solver(B.toarray())

        # compute rank one update for vector of ones
        ones = np.ones(o)
        self.lap_sol += np.outer(v, ones) * (1.0 / d - np.sum(v) / (d * denom))

    def _comp_mat_block_inv(self):
        """Computes matrix block inversion formula"""
        d = len(self.sp_graph)
        o = self.sp_graph.n_observed_nodes

        # multiply L_{o,d-o} by solution of lap-system
        A = self.sp_graph.L_block["od"] @ self.lap_sol

        # multiply one matrix by solution of lap-system
        B = np.outer(np.ones(o), self.lap_sol.sum(axis=0)) / d

        # sum up with L_{o,o} and one matrix 
        ## Eqn 16 (pg. 23)
        self.L_double_inv = self.sp_graph.L_block["oo"].toarray() + 1.0 / d - A - B

    def _comp_inv_lap(self, B=None):
        """Computes submatrices of inverse of lap"""
        if B is None:
            B = np.eye(self.sp_graph.n_observed_nodes)

        # inverse of graph laplacian
        # compute o-by-o submatrix of inverse of lap
        self.Linv_block = {}
        self.Linv_block["oo"] = np.linalg.solve(self.L_double_inv, B)
        # compute (d-o)-by-o submatrix of inverse of lap
        self.Linv_block["do"] = -self.lap_sol @ self.Linv_block["oo"]

        # stack the submatrices
        self.Linv = np.vstack((self.Linv_block["oo"], self.Linv_block["do"]))

    def _comp_inv_cov(self, B=None):
        """Computes inverse of the covariance matrix"""
        # helper
        A = (
            -self.sp_graph.q_inv_diag.toarray()
            - (self.sp_graph.q_inv_diag @ self.L_double_inv) @ self.sp_graph.q_inv_diag
        )
        if B is None:
            B = np.eye(self.sp_graph.n_observed_nodes)

        # solve o-by-o linear system to get X
        self.X = np.linalg.solve(A, B)

        # inverse covariance matrix
        self.inv_cov = self.X + np.diag(self.sp_graph.q)
        self.inv_cov_sum = self.inv_cov.sum(axis=0)
        self.denom = self.inv_cov_sum.sum()

    def _comp_grad_obj(self):
        """Computes the gradient of the objective function with respect to the
        latent variables dLoss / dL
        """
        # compute inverses
        self._comp_inv_lap()

        self.comp_B = self.inv_cov - (1.0 / self.denom) * np.outer(
            self.inv_cov_sum, self.inv_cov_sum
        )
        self.comp_A = self.comp_B @ self.sp_graph.S @ self.comp_B
        M = self.comp_A - self.comp_B
        self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

        # grads
        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        self.grad_obj = gradD - gradW

        # grads for d diag(Jq^-1) / dq
        if self.sp_graph.optimize_q == 'n-dim':
            self.grad_obj_q = np.zeros(len(self.sp_graph))
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)                   
        elif self.sp_graph.optimize_q == '1-dim':
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

    def _comp_grad_obj_c(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c) with respect to the latent variables dLoss / dL
        """

        # compute inverses
        self._comp_inv_lap()

        if not hasattr(self, 'Lpinv'):
            self.Lpinv = pinvh(self.sp_graph.L.todense())

        sid = np.where(self.sp_graph.perm_idx == self.sp_graph.edge[0][0])[0][0]
        did = np.where(self.sp_graph.perm_idx == self.sp_graph.edge[0][1])[0][0]

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        resmat = Rmat + (Q1mat + Q1mat.T) - 2*self.sp_graph.q_inv_diag

        if sid<self.sp_graph.n_observed_nodes:
            resmat[sid,did] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*Rmat[sid,did] + (1+self.sp_graph.c)/self.sp_graph.q[sid] + (1-self.sp_graph.c)/self.sp_graph.q[did]
            resmat[did,sid] = resmat[sid,did]

            for i in set(range(self.sp_graph.n_observed_nodes))-set([sid,did]):
                resmat[i,did] = (1-self.sp_graph.c)*Rmat[i,did] + self.sp_graph.c*Rmat[i,sid] + 0.5*(self.sp_graph.c**2-self.sp_graph.c)*Rmat[sid,did] + 1/self.sp_graph.q[i] + (1-self.sp_graph.c)/self.sp_graph.q[did] + self.sp_graph.c/self.sp_graph.q[sid]
                resmat[did,i] = resmat[i,did]
        else:
            neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[sid]))
            # finds the neighboring deme that has samples
            neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

            R1d = -2*self.Lpinv[sid,did] + self.Lpinv[sid,sid] + self.Lpinv[did,did]
            R1 = np.array(-2*self.Lpinv[:self.sp_graph.n_observed_nodes,sid].T + np.diag(self.Linv) + self.Lpinv[sid,sid])

            # apply this formula only to neighboring sampled demes
            # TODO: check if I need this here? what happens if I don't have this? (test in the case when the true deme is unsampled but next to a sampled deme)
            # (anecdotally, there were discontinuities around a sampled deme in the log-lik surface)
            for n in neighs:
                # convert back to appropriate indexing excluding the unsampled demes
                s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==n][0]
                # (1+c)q_s gives an overestimate of the c value (slide 61) ->  keeping it at 1-c
                resmat[s,did] = Rmat[s,did] + 0.5*(self.sp_graph.c**2-self.sp_graph.c)*R1d + (1-self.sp_graph.c)/self.sp_graph.q[s] + (1+self.sp_graph.c)/self.sp_graph.q[did]
                resmat[did,s] = resmat[s,did]

            rsm = np.mean(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            rsd = np.std(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            ## smaller coefficients inside the exp make the log-lik more peaked around the maximum value
            qprox = np.dot(1/self.sp_graph.q, 1/R1*np.exp(-np.abs(rsm-R1)/rsd)/np.sum(1/R1*np.exp(-np.abs(rsm-R1)/rsd)))

            ## id
            for i in set(range(self.sp_graph.n_observed_nodes))-set([sid,did]+neighs):
                Ri1 = -2*self.Lpinv[i,sid] + self.Lpinv[i,i] + self.Lpinv[sid,sid]
                resmat[i,did] = (1-self.sp_graph.c)*(Rmat[i,did]) + self.sp_graph.c*Ri1 + 0.5*(self.sp_graph.c**2-self.sp_graph.c)*R1d + 1/self.sp_graph.q[i] + (1-self.sp_graph.c)/self.sp_graph.q[did] + self.sp_graph.c*qprox
                resmat[did,i] = resmat[i,did]
        
        # convert distance matrix to covariance matrix (using code from rwc package in Hanks & Hooten 2013)
        rwsm = np.mean(resmat, 0).reshape(-1,1) @ np.ones(resmat.shape[0]).reshape(1,-1)
        clsm = np.ones(resmat.shape[0]).reshape(-1, 1) @ np.mean(resmat, 1).reshape(1,-1)
        Sigma = 0.5 * (-resmat + rwsm + clsm - np.sum(resmat)/resmat.shape[0]**2)
        # Eqn 18 in Marcus et al 2021 
        CRCt = np.linalg.inv(self.C @ Sigma @ self.C.T) 
        Pi1 = Sigma @ self.C.T @ CRCt @ self.C
        siginv = np.linalg.inv(Sigma)
        if self.sp_graph.optimize_q == 'n-dim':
            # Eqn 12 from Marcus et al 2021 (but is equivalent to Eqn 18 I think?)
            # M = self.C.T @ (CRCt @ (self.C @ self.sp_graph.S @ self.C.T) @ CRCt - CRCt) @ self.C
            M = siginv @ Pi1 @ self.sp_graph.S @ siginv @ Pi1 - siginv @ Pi1
            # plt.imshow(M); plt.colorbar(); plt.show()
        else:
            self.comp_B = self.inv_cov - (1.0 / self.denom) * np.outer(
                self.inv_cov_sum, self.inv_cov_sum
            )
            self.comp_A = self.comp_B @ self.sp_graph.S @ self.comp_B
            M = self.comp_A - self.comp_B
            
        self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        self.grad_obj = np.ravel(gradD - gradW)
        
        # grads for d diag(Jq^-1) / dq
        if self.sp_graph.optimize_q == 'n-dim':
            self.grad_obj_q = np.zeros(len(self.sp_graph))
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)        
        else:
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

    def _comp_grad_reg(self):
        """Computes gradient"""
        lamb = self.lamb
        alpha = self.alpha

        # avoid overflow in exp
        # term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
        # term_1 = alpha * self.sp_graph.w + np.log(term_0)
        # term_2 = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term_1)
        # self.grad_pen = term_2 * (alpha / term_0)
        term = alpha * self.sp_graph.w + np.log(
            1 - np.exp(-alpha * self.sp_graph.w)
        )  # avoid overflow in exp
        self.grad_pen = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term)
        self.grad_pen = self.grad_pen * (alpha / (1 - np.exp(-alpha * self.sp_graph.w))) 

        if self.sp_graph.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
            
            term = alpha_q * self.sp_graph.s2 + np.log(
                1 - np.exp(-alpha_q * self.sp_graph.s2)
            )
            self.grad_pen_q = self.sp_graph.Delta_q.T @ self.sp_graph.Delta_q @ (lamb_q * term)
            self.grad_pen_q = self.grad_pen_q * (alpha_q / (1 - np.exp(-alpha_q * self.sp_graph.s2)))

    def inv(self):
        """Computes relevant inverses for gradient computations"""
        # compute inverses
        self._solve_lap_sys()
        self._comp_mat_block_inv()
        self._comp_inv_cov()

    def grad(self, reg=True):
        """Computes relevent gradients the objective"""
        # compute derivatives
        if self.sp_graph.option == 'default':
            self._comp_grad_obj()
        elif self.sp_graph.option == 'onlyc':
            self._comp_grad_obj_c()

        if reg is True:
            self._comp_grad_reg()

    def neg_log_lik(self):
        """Evaluate the negative log-likelihood function given the current
        params
        """

        o = self.sp_graph.n_observed_nodes
        self.trA = self.sp_graph.S @ self.inv_cov

        # trace
        self.trB = self.inv_cov_sum @ self.trA.sum(axis=1)
        self.tr = np.trace(self.trA) - self.trB / self.denom

        # det
        # E = self.X + np.diag(self.sp_graph.q)
        # self.det = np.linalg.det(self.inv_cov) * o / self.denom
        # VS: made a change here to accommodate larger data sets (was leading to overflow)
        self.logdet = np.linalg.slogdet(self.inv_cov)[1]

        # negative log-likelihood
        # nll = self.sp_graph.n_snps * (self.tr - np.log(self.det))
        nll = self.sp_graph.n_snps * (self.tr - self.logdet - np.log(o/self.denom))

        return nll

    # def loss(self):
    #     """Evaluate the loss function given the current params"""
    #     lamb = self.lamb
    #     alpha = self.alpha

    #     lik = self.neg_log_lik()

    #     term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
    #     term_1 = alpha * self.sp_graph.w + np.log(term_0)
    #     pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2

    #     # loss
    #     loss = lik + pen 
    #     return loss

    def _update_graph(self, basew, bases2):
        self.sp_graph.option = 'default'
        
        self.sp_graph.w = basew; self.sp_graph.s2 = bases2
        
        self.sp_graph.comp_graph_laplacian(basew); self.sp_graph.comp_precision(bases2)
        self.inv(); self.grad(reg=False)
        self.Lpinv = pinvh(self.sp_graph.L.todense())

    def loss(self):
        """Evaluate the loss function given the current params"""
        lamb = self.lamb
        alpha = self.alpha

        if self.sp_graph.option == 'default':
            lik = self.neg_log_lik()
        else:
            lik = self.eems_neg_log_lik(self.sp_graph.c, opts={'mode':'compute','edge':self.sp_graph.edge})

        term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
        term_1 = alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
                
        if self.sp_graph.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen += 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2  

        # loss
        loss = lik + pen
        return loss 

    def eems_neg_log_lik(self, c=None, opts=None):
        # could also just pass in c=0 to get the log-lik of the current delta matrix
        
        # lre passed in as permuted_idx
        if opts is not None:
            sid = np.where(self.sp_graph.perm_idx == opts['edge'][0][0])[0][0]
            did = np.where(self.sp_graph.perm_idx == opts['edge'][0][1])[0][0]
            assert did < self.sp_graph.n_observed_nodes, "ensure that the destination is a sampled deme (check ID from the map or from output of extract_outliers"
            opts['lre'] = [(sid,did)]
            # print(opts['lre'])
        else:
            opts = {}
            # if no edge is passed in, just use a dummy index with c=0
            opts['lre'] = [(0,1)]
            
        if c is not None:
            if opts['mode'] != 'update':
                dd = self._compute_delta_matrix(c, opts)
                # print(np.where(np.isnan(dd)))
                nll = -wishart.logpdf(-self.sp_graph.n_snps*self.CDCt, self.sp_graph.n_snps, -self.C @ dd @ self.C.T)
            else:
                opts['delta'] = self._compute_delta_matrix(c, opts)
                nll = -wishart.logpdf(-self.sp_graph.n_snps*self.CDCt, self.sp_graph.n_snps, -self.C @ opts['delta'] @ self.C.T)
        else:
            dd = self._compute_delta_matrix(0, opts)
            nll = -wishart.logpdf(-self.sp_graph.n_snps*self.CDCt, self.sp_graph.n_snps, -self.C @ dd @ self.C.T)
                   
        return nll
    
    def _compute_delta_matrix(self, c, opts):
        """
        Compute a new delta matrix given a previous delta matrix as a perturbation from a single long range gene flow event OR create a new delta matrix from resmat
        """

        if not hasattr(self, 'Linv'):
            self.inv(); self.grad(reg=False)
        
        if not hasattr(self, 'Lpinv'):
            self.Lpinv = pinvh(self.sp_graph.L.todense())

        # print(self.sp_graph.w[:10])

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) 
        Q1mat = np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes))
        ## both variations below gives not posdef errors with 6x6
        # Q1mat = 0.5*(1-self.Linv.diagonal()).reshape(-1,1) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) 
        # Q1mat = self.sp_graph.number_of_nodes()/2*np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        
        if 'delta' in opts:
            resmat = np.copy(opts['delta'])
        else:
            resmat = Rmat + (Q1mat + Q1mat.T) - 2*self.sp_graph.q_inv_diag 

        if opts['lre'][0][0] < self.sp_graph.n_observed_nodes:
            # resmat[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+c)/self.sp_graph.q[opts['lre'][0][0]] + (1-c)/self.sp_graph.q[opts['lre'][0][1]]
            resmat[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+c)*Q1mat[opts['lre'][0][0],opts['lre'][0][0]] + (1-c)*Q1mat[opts['lre'][0][1],opts['lre'][0][1]]
            # resmat[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+c)*(1/self.sp_graph.q[opts['lre'][0][0]] + self.Linv[opts['lre'][0][0],opts['lre'][0][0]]) + (1-c)*(1/self.sp_graph.q[opts['lre'][0][1]] + self.Linv[opts['lre'][0][1],opts['lre'][0][1]])
            resmat[opts['lre'][0][1],opts['lre'][0][0]] = resmat[opts['lre'][0][0],opts['lre'][0][1]]

            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]):
                # resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,opts['lre'][0][0]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[opts['lre'][0][0]]
                resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,opts['lre'][0][0]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)*Q1mat[opts['lre'][0][1],opts['lre'][0][1]] + c*Q1mat[opts['lre'][0][0],opts['lre'][0][0]]
                # resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,opts['lre'][0][0]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1/self.sp_graph.q[i] + self.Linv[i,i]) + (1-c)*(1/self.sp_graph.q[opts['lre'][0][1]] + self.Linv[opts['lre'][0][1],opts['lre'][0][1]]) + c*(1/self.sp_graph.q[opts['lre'][0][0]] + self.Linv[opts['lre'][0][0],opts['lre'][0][0]])
                resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]
        else:
            ## picking the 3 closest sampled demes (using the old approach)
            # gets the 6 neighboring demes
            # neighs = []
            neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[opts['lre'][0][0]]))
            # finds the neighboring deme that has samples
            neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

            R1d = -2*self.Lpinv[opts['lre'][0][0],opts['lre'][0][1]] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]] + self.Lpinv[opts['lre'][0][1],opts['lre'][0][1]]
            R1 = np.array(-2*self.Lpinv[:self.sp_graph.n_observed_nodes,opts['lre'][0][0]].T + np.diag(self.Linv) + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]])

            # apply this formula only to neighboring sampled demes
            # TODO: check if I need this here? what happens if I don't have this? (test in the case when the true deme is unsampled but next to a sampled deme)
            # (anecdotally, there was discontinuities around a sampled deme in the log-lik surface)
            for n in neighs:
                # convert back to appropriate indexing excluding the unsampled demes
                s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==n][0]
                # (1+c)q_s gives an overestimate of the c value (slide 61) ->  keeping it at 1-c
                resmat[s,opts['lre'][0][1]] = Rmat[s,opts['lre'][0][1]] + 0.5*(c**2-c)*R1d + (1-c)/self.sp_graph.q[s] + (1+c)/self.sp_graph.q[opts['lre'][0][1]]
                resmat[opts['lre'][0][1],s] = resmat[s,opts['lre'][0][1]]

            # find the closest sampled deme to 1 (this is just the proxy source, use q from here but do not model this as the source)
            # proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])
            # proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])[:1]
            # proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set(self.sp_graph.perm_idx[:self.sp_graph.n_observed_nodes])-set([opts['lre'][0][0]])])[:self.sp_graph.n_observed_nodes]
            # qprox = np.dot(1/self.sp_graph.q[proxs], (1/R1[0,proxs]*np.exp(-2/R1[0,proxs]))/np.sum(1/R1[0,proxs]*np.exp(-2/R1[0,proxs])))
            # qprox = np.dot(1/self.sp_graph.q[proxs], (R1[0,proxs]*np.exp(-2*R1[0,proxs]))/np.sum(R1[0,proxs]*np.exp(-2*R1[0,proxs])))
            rsm = np.mean(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            rsd = np.std(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            # qprox = np.dot(1/self.sp_graph.q[proxs], 1/R1[0,proxs]*np.exp(-0.5*np.abs(rsm-R1[0,proxs])/rsd)/np.sum(1/R1[0,proxs]*np.exp(-0.5*np.abs(rsm-R1[0,proxs])/rsd)))
            ## smaller coefficients inside the exp make the log-lik more peaked around the maximum value
            qprox = np.dot(1/self.sp_graph.q, 1/R1*np.exp(-np.abs(rsm-R1)/rsd)/np.sum(1/R1*np.exp(-np.abs(rsm-R1)/rsd)))

            ## id
            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]+neighs):
                Ri1 = -2*self.Lpinv[i,opts['lre'][0][0]] + self.Lpinv[i,i] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
                # resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Ri1 + 0.5*(c**2-c)*R1d + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[proxs]
                resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Ri1 + 0.5*(c**2-c)*R1d + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c*qprox
                resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]

            ## if picking the closest sampled deme as a proxy (treating the unsampled deme with the same equations as above)
            ## (this gives a very smooth loglik contour and produces same results as doing topidx=1)
            # R1 = -2*self.Lpinv[:self.sp_graph.n_observed_nodes,opts['lre'][0][0]].T + np.diag(self.Linv) + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
            # one = np.argmin(R1)

            # resmat[one,opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[one,opts['lre'][0][1]] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + (1+c)/self.sp_graph.q[one]*np.exp(-2*R1[0,one]) 
            # resmat[one,opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[one,opts['lre'][0][1]] + (1+c)*(1/self.sp_graph.q[one] + self.Linv[one,one]) + (1-c)*(1/self.sp_graph.q[opts['lre'][0][1]] + self.Linv[opts['lre'][0][1],opts['lre'][0][1]])
            # resmat[opts['lre'][0][1],one] = resmat[one,opts['lre'][0][1]]
            # for i in set(range(self.sp_graph.n_observed_nodes))-set([one,opts['lre'][0][1]]):
            #     # resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,one] + 0.5*(c**2-c)*Rmat[one,opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[one]*np.exp(-2*R1[0,one])
            #     resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,one] + 0.5*(c**2-c)*Rmat[one,opts['lre'][0][1]] + (1/self.sp_graph.q[i] + self.Linv[i,i]) + (1-c)*(1/self.sp_graph.q[opts['lre'][0][1]] + self.Linv[opts['lre'][0][1],opts['lre'][0][1]]) + c*(1/self.sp_graph.q[one] + self.Linv[one,one])
            #     resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]

            # ## picking the n closest demes and doing a weighted average
            # R1 = -2*self.Lpinv[:self.sp_graph.n_observed_nodes,opts['lre'][0][0]].T + np.diag(self.Linv) + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
            # topidx = np.ravel(np.argsort(R1)[0,:1]).tolist()
            # q1 = np.dot(1/self.sp_graph.q[topidx], (1/R1[0,topidx].T)/np.sum(1/R1[0,topidx]))
            # # q1 = 1/self.sp_graph.q[topidx]*np.exp(-2*R1[0,topidx])
            
            # for it in topidx:
            #     ## doing this gives me a very low log-lik for neighboring unsampled deme compared to sampled deme 
            #     # resmat[it,opts['lre'][0][1]] = Rmat[it,opts['lre'][0][1]] + 0.5*(c**2-c)*R1[0,opts['lre'][0][1]] + (1+c)/self.sp_graph.q[opts['lre'][0][1]] - c*q1 + 1/self.sp_graph.q[it]
            #     resmat[it,opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[it,opts['lre'][0][1]] + (1+c)/self.sp_graph.q[it] + (1-c)/self.sp_graph.q[opts['lre'][0][1]]
            #     resmat[opts['lre'][0][1],it] = resmat[it,opts['lre'][0][1]]                
            # for i in set(range(self.sp_graph.n_observed_nodes))-set(topidx+[opts['lre'][0][1]]):
            #     resmat[i,opts['lre'][0][1]] = (1-c)*Rmat[i,opts['lre'][0][1]] + c*R1[0,i] + 0.5*(c**2-c)*R1[0,opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c*q1
            #     resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]

        # # basically, smaller the coefficient in front of CRCt, the more downward biased the admix. prop. estimates 
        # nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        return np.array(resmat)


def neg_log_lik_w0_s2(z, obj):
    """Computes negative log likelihood for a constant w and residual variance"""
    theta = np.exp(z)
    obj.lamb = 0.0
    obj.alpha = 1.0
    
    obj.sp_graph.w = theta[0] * np.ones(obj.sp_graph.size())
    obj.sp_graph.comp_graph_laplacian(obj.sp_graph.w)
    obj.sp_graph.comp_precision(s2=theta[1])
    obj.inv()
    nll = obj.neg_log_lik()
    
    return nll


def loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w,q) which returns the loss and gradient"""                
    n_edges = obj.sp_graph.size()
    if obj.sp_graph.optimize_q is not None:
        theta = np.exp(z)
        theta0 = theta[:n_edges]
        obj.sp_graph.comp_graph_laplacian(theta0)
        # if obj.optimize_q is not None:
        theta1 = theta[n_edges:]
        obj.sp_graph.comp_precision(s2=theta1)
    else:
        theta = np.exp(z)
        obj.sp_graph.comp_graph_laplacian(theta)
    obj.inv()
    obj.grad()     

    # loss / grad
    loss = obj.loss()
    if obj.sp_graph.optimize_q is None:
        grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    elif obj.sp_graph.optimize_q == 'n-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2 + obj.grad_pen_q * obj.sp_graph.s2
    else:
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2    

    return (loss, grad)

# def loss_wrapper(z, obj):
#     """Wrapper function to optimize z=log(w) which returns the loss and gradient"""
#     theta = np.exp(z)
#     obj.sp_graph.comp_graph_laplacian(theta)
#     obj.inv()
#     obj.grad()

#     #  s / grad
#     loss = obj.loss()
#     grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
#     return (loss, grad)

def comp_mats(obj):
    """Compute fitted covariance matrix and its inverse & empirical convariance matrix"""
    obj.inv()
    obj.grad(reg=False)
    sp_graph = obj.sp_graph
    d = len(sp_graph)
    fit_cov = obj.Linv_block['oo'] - 1/d + sp_graph.q_inv_diag.toarray()
    
    inv_cov_sum0 = obj.inv_cov.sum(axis=0)
    inv_cov_sum1 = obj.inv_cov.sum()
    inv_cov = obj.inv_cov + np.outer(inv_cov_sum0, inv_cov_sum0) / (d - inv_cov_sum1)    
    
    assert np.allclose(inv_cov, np.linalg.inv(fit_cov)) == True, "fit_cov must be inverse of inv_cov"
    
    n_snps = sp_graph.n_snps
    
    # VS: changing code here to run even when scale_snps=False
    if hasattr(obj.sp_graph.q, 'mu'):
        frequencies_ns = sp_graph.frequencies * np.sqrt(sp_graph.mu*(1-sp_graph.mu))
        mu0 = frequencies_ns.mean(axis=0) / 2 # compute mean of allele frequencies in the original scale
        mu = 2*mu0 / np.sqrt(sp_graph.mu*(1-sp_graph.mu))
        frequencies_centered = sp_graph.frequencies - mu
    else:
        frequencies_centered = sp_graph.frequencies

    emp_cov = frequencies_centered @ frequencies_centered.T / n_snps
    
    return fit_cov, inv_cov, emp_cov
    