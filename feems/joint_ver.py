import sys
import allel
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from scipy.stats import wishart, norm, chi2
import statsmodels.api as sm
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, minimize
import itertools as it

from .spatial_graph import SpatialGraph, query_node_attributes
from .objective import Objective, neg_log_lik_w0_s2, comp_mats
from .utils import cov_to_dist, benjamini_hochberg, get_outlier_idx

class FEEMSmix_Objective(Objective): 
    def __init__(self, sp_graph, option='default'):
        """Inherit from the feems object Objective and overwrite some methods for evaluations 
        and gradient of feems objective when residual variance is estimated jointly with edge weights
        Args:
            sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
            option (string): indicating whether weights & admixture proportion are jointly estimated
        """
        super().__init__(sp_graph=sp_graph)   
        
        # indicator whether optimizing residual variance jointly with edge weights
        # None : residual variance is kept fixed at Nelder-Mead value
        # 1-dim : single residual variance is estimated across all demes 
        # n-dim : deme-specific residual variances are estimated for each deme
        
        ## reg params for residual variance
        self.lamb_q = None
        self.alpha_q = None

        self.nll = 0.0

        self.C = np.vstack((-np.ones(self.sp_graph.n_observed_nodes-1), np.eye(self.sp_graph.n_observed_nodes-1))).T

        # really important to have the right indices for any work
        self.perm_idx = query_node_attributes(self.sp_graph, "permuted_idx")

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
            self.Lpinv = np.linalg.pinv(self.sp_graph.L.T.todense())

        sid = np.where(self.perm_idx == self.sp_graph.edge[0][0])[0][0]
        did = np.where(self.perm_idx == self.sp_graph.edge[0][1])[0][0]

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
            qprox = np.dot(1/self.sp_graph.q, 1/R1[0,:]*np.exp(-np.abs(rsm-R1[0,:])/rsd)/np.sum(1/R1[0,:]*np.exp(-np.abs(rsm-R1[0,:])/rsd)))

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

    # def _comp_grad_obj_c_t(self):
    #     """Computes the gradient of the objective function (now defined with admix. prop. c & admix. time t) with respect to the latent variables dLoss / dL
    #     """

    #     # compute inverses
    #     self._comp_inv_lap()

    #     Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv)[:self.sp_graph.n_observed_nodes],(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv)[:self.sp_graph.n_observed_nodes],(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
    #     Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
    #     Tstar = Rmat + (Q1mat + Q1mat.T); Tstar[np.diag_indices_from(Tstar)] = 0

    #     dT0 = np.zeros_like(Tstar)

    #     for ie, _ in enumerate(self.sp_graph.edge):
    #         for i in list(set(range(dT0.shape[0]))-set([did])):
    #             dT0[i,did] = self.sp_graph.c[ie]*Tstar[i,self.sp_graph.edge[ie][0]] - self.sp_graph.c[ie]*Tstar[i,did] 
    #             dT0[self.sp_graph.edge[ie][1],i] = dT0[i,self.sp_graph.edge[ie][1]]
    #         dT0[self.sp_graph.edge[ie][1],self.sp_graph.edge[ie][1]] = self.sp_graph.c[ie]*Tstar[self.sp_graph.edge[ie][0],self.sp_graph.edge[ie][0]] - self.sp_graph.c[ie]*Tstar[self.sp_graph.edge[ie][1],self.sp_graph.edge[ie][1]] 
    #         #self.sp_graph.c[ie]**2*Tstar[self.sp_graph.edge[ie][0],self.sp_graph.edge[ie][0]] + 2*self.sp_graph.c[ie]*(1-self.sp_graph.c[ie])*Tstar[self.sp_graph.edge[ie][0],self.sp_graph.edge[ie][1]] + (self.sp_graph.c[ie]**2-2*self.sp_graph.c[ie])*Tstar[self.sp_graph.edge[ie][1],self.sp_graph.edge[ie][1]] 
            
    #     dTt = -self.sp_graph.t[ie]*(np.diag(1/self.sp_graph.q)@np.diag(np.diag(dT0)) + self.Linv@dT0 + dT0@self.Linv) + dT0

    #     resmat = Tstar + dTt; resmat[np.diag_indices_from(resmat)] = 0

    #     CRCt = np.linalg.inv(-0.5*self.C @ resmat @ self.C.T)
    #     if self.optimize_q == 'n-dim':
    #         M = self.C.T @ (CRCt @ (self.C @ self.sp_graph.S @ self.C.T) @ CRCt - CRCt) @ self.C
    #     else:
    #         self.comp_B = self.inv_cov - (1.0 / self.denom) * np.outer(
    #             self.inv_cov_sum, self.inv_cov_sum
    #         )
    #         self.comp_A = self.comp_B @ self.sp_graph.S @ self.comp_B
    #         M = self.comp_A - self.comp_B

    #     self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

    #     gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
    #     gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
    #     self.grad_obj = np.ravel(gradD - gradW)
        
    #     # grads for d diag(Jq^-1) / dq
    #     if self.optimize_q == 'n-dim':
    #         self.grad_obj_q = np.zeros(len(self.sp_graph))
    #         self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)        
    #     else:
    #         self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

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

    def _update_graph(self, basew, bases2):
        self.sp_graph.option = 'default'
        # TODO check the dimensions here to make sure that s2 assignment is correct for s2 
        self.sp_graph.w = basew; self.sp_graph.s2 = bases2
        self.sp_graph.comp_graph_laplacian(basew); self.sp_graph.comp_precision(bases2)
        self.inv(); self.grad(reg=False)
        self.Lpinv = np.linalg.pinv(self.sp_graph.L.todense())
        # print('{:.1f}'.format(self.eems_neg_log_lik()))
        
    def loss(self):
        """Evaluate the loss function given the current params"""
        lamb = self.lamb
        alpha = self.alpha

        if self.sp_graph.option == 'default':
            lik = self.neg_log_lik()
        # elif self.sp_graph.option == 'onlyc':
        else:
            # lik = self.neg_log_lik_c(self.sp_graph.c, opts={'mode':'sampled','lre':self.sp_graph.edge})
            lik = self.eems_neg_log_lik(self.sp_graph.c, opts={'mode':'compute','edge':self.sp_graph.edge})
        # else:
        #     lik = self.neg_log_lik_c_t([self.sp_graph.c, self.sp_graph.t], opts={'lre':self.sp_graph.edge})
        #     print([self.sp_graph.c, self.sp_graph.t])

        term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
        term_1 = alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
        # print(pen)
                
        if self.sp_graph.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen += 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2  
            # print(str(pen)+"here") 

        # loss
        loss = lik + pen
        return loss 

    def joint_eems_neg_log_lik(self, x0, opts):
        """Evaluate the joint negative log-likelihood for the given weights, s2 and admix. prop. c, but taking as input all three sets of parameters
        (prototype function: only works with option 'n-dim')
        """

        n_edges = self.sp_graph.size()
        self.sp_graph.comp_graph_laplacian(np.exp(x0[:n_edges]))
        self.sp_graph.comp_precision(s2=np.exp(x0[n_edges:-1]))
        self.inv(); #self.grad(reg=False)

        # lre passed in as permuted_idx
        sid = np.where(self.perm_idx == opts['edge'][0][0])[0][0]
        did = np.where(self.perm_idx == opts['edge'][0][1])[0][0]
        opts['lre'] = [(sid,did)]

        if sid<self.sp_graph.n_observed_nodes:
            self.Lpinv = np.linalg.pinv(self.sp_graph.L.T.todense())
        if did>self.sp_graph.n_observed_nodes:
            print("Enter a valid destination deme ID")
            return 
            
        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        opts['delta'] = self._compute_delta_matrix(x0[-1], opts)
        nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ opts['delta'] @ self.C.T)
        
        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = self.alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * self.lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2

        term_0 = 1.0 - np.exp(-self.alpha_q * self.sp_graph.s2)
        term_1 = self.alpha_q * self.sp_graph.s2 + np.log(term_0)
        pen += 0.5 * self.lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2

        return nll + pen

    def eems_neg_log_lik(self, c=None, opts=None):
        # could also just pass in c=0 to get the log-lik of the current delta matrix
        
        # lre passed in as permuted_idx
        if opts is not None:
            sid = np.where(self.perm_idx == opts['edge'][0][0])[0][0]
            did = np.where(self.perm_idx == opts['edge'][0][1])[0][0]
            assert did < self.sp_graph.n_observed_nodes, "ensure that the destination is a sampled deme (check ID from the map or from output of extract_outliers"
            opts['lre'] = [(sid,did)]
            # print(opts['lre'])
        else:
            opts = {}
            # if no edge is passed in, just use a dummy index with c=0
            opts['lre'] = [(0,1)]

        # TODO make D into an internal variable for sp_graph
        # TODO also make C D C^T an internal variable (unnecessary computations)
        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        if c is not None:
            # TODO is it better to make this a True/False flag instead?
            if opts['mode'] != 'update':
                dd = self._compute_delta_matrix(c, opts)
                # print(np.where(np.isnan(dd)))
                nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ dd @ self.C.T)
            else:
                opts['delta'] = self._compute_delta_matrix(c, opts)
                nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ opts['delta'] @ self.C.T)
        else:
            dd = self._compute_delta_matrix(0, opts)
            nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ dd @ self.C.T)
                   
        return nll

    # def eems_neg_log_lik_sigma(self, x, opts):
    #     "Function to estimate c AND \sigma^\star jointly (latter tends to be \approx 1 for most cases, so dropping it)"
        
    #     # lre passed in as permuted_idx
    #     sid = np.where(self.perm_idx == opts['edge'][0][0])[0][0]
    #     did = np.where(self.perm_idx == opts['edge'][0][1])[0][0]
    #     opts['lre'] = [(sid,did)]

    #     D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

    #     # TODO is it better to make this a True/False flag instead?
    #     if opts['mode'] != 'update':
    #         dd = self.compute_delta_matrix(x[0], opts)
    #         nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -np.exp(x[1])*self.C @ dd @ self.C.T)
    #     else:
    #         opts['delta'] = self.compute_delta_matrix(x[0], opts)
    #         nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -np.exp(x[1])*self.C @ opts['delta'] @ self.C.T)
        
    #     return nll

    # TODO include an underscore at the start of the function name (internal use only)
    def _compute_delta_matrix(self, c, opts):
        """
        Compute a new delta matrix given a previous delta matrix as a perturbation from a single long range gene flow event OR create a new delta matrix from resmat
        """

        if not hasattr(self, 'Linv'):
            self.inv(); self.grad(reg=False)
        
        if not hasattr(self, 'Lpinv'):
            self.Lpinv = np.linalg.pinv(self.sp_graph.L.T.todense())

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
            # proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set(self.perm_idx[:self.sp_graph.n_observed_nodes])-set([opts['lre'][0][0]])])[:self.sp_graph.n_observed_nodes]
            # qprox = np.dot(1/self.sp_graph.q[proxs], (1/R1[0,proxs]*np.exp(-2/R1[0,proxs]))/np.sum(1/R1[0,proxs]*np.exp(-2/R1[0,proxs])))
            # qprox = np.dot(1/self.sp_graph.q[proxs], (R1[0,proxs]*np.exp(-2*R1[0,proxs]))/np.sum(R1[0,proxs]*np.exp(-2*R1[0,proxs])))
            rsm = np.mean(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            rsd = np.std(Rmat[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)])
            # qprox = np.dot(1/self.sp_graph.q[proxs], 1/R1[0,proxs]*np.exp(-0.5*np.abs(rsm-R1[0,proxs])/rsd)/np.sum(1/R1[0,proxs]*np.exp(-0.5*np.abs(rsm-R1[0,proxs])/rsd)))
            ## smaller coefficients inside the exp make the log-lik more peaked around the maximum value
            qprox = np.dot(1/self.sp_graph.q, 1/R1[0,:]*np.exp(-np.abs(rsm-R1[0,:])/rsd)/np.sum(1/R1[0,:]*np.exp(-np.abs(rsm-R1[0,:])/rsd)))

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

        # TODO: use a function to compute the negative log-likelihood given the two matrices (instead of doing it in a single function?)
        # D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S
        # # basically, smaller the coefficient in front of CRCt, the more downward biased the admix. prop. estimates 
        # # nll = -wishart.logpdf(2*self.sp_graph.n_snps*self.C @ self.sp_graph.S @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)
        # nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        return resmat
        
    def extract_outliers(self, fdr=0.25, res_dist=None, verbose=False):
        """Function to extract outlier deme pairs based on a FDR threshold specified by the user (default: 0.25)"""
        assert fdr>0 and fdr<1, "fdr should be a positive number between 0 and 1"

        # computing pairwise covariance & distances between demes
        fit_cov, _, emp_cov = comp_mats(self)
        emp_dist = cov_to_dist(emp_cov)[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)]
        if res_dist is None:
            fit_dist = cov_to_dist(fit_cov)[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)]
        else: 
            fit_dist = res_dist

        # print('Using a significance threshold of {:g}:\n'.format(pthresh))
        print('Using a FDR of {:g}:\n'.format(fdr))
        ls = []; x, y = [], []
        acs = np.empty((2, self.sp_graph.n_snps, 2))
        # computing p-values (or z-values) for each pairwise comparison after mean centering
        pvals = norm.cdf(np.log(emp_dist)-np.log(fit_dist)-np.mean(np.log(emp_dist)-np.log(fit_dist)), 0, np.std(np.log(emp_dist)-np.log(fit_dist)))
        # med = np.median(np.log(emp_dist)-np.log(fit_dist))
        # mad = 1.4826*np.median(np.abs(np.log(emp_dist)-np.log(fit_dist) - med))
        # mvals = (np.log(emp_dist)-np.log(fit_dist) - med)/mad

        bh = benjamini_hochberg(emp_dist, fit_dist, fdr=fdr)
        
        # for k in np.where(pvals < pthresh)[0]:
        # for k in np.where(mvals < -mthresh)[0]:
        for k in np.where(bh)[0]:
            # code to convert single index to matrix indices
            x.append(np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1); y.append(int(k - 0.5*x[-1]*(x[-1]-1)))

            # Gi = self.sp_graph.genotypes[self.sp_graph.nodes[self.perm_idx[x[-1]]]['sample_idx'], :]
            Gi = self.sp_graph.genotypes[self.sp_graph.nodes[self.perm_idx[x[-1]]]['sample_idx'], :].astype('int').T
            Ga1 = np.zeros((Gi.shape[0], Gi.shape[1], 2), dtype=int)
            Ga1[Gi == 1, 1] = 1
            Ga1[Gi == 2] = 1

            Gi = self.sp_graph.genotypes[self.sp_graph.nodes[self.perm_idx[y[-1]]]['sample_idx'], :].astype('int').T
            Ga2 = np.zeros((Gi.shape[0], Gi.shape[1], 2), dtype=int)
            Ga2[Gi == 1, 1] = 1
            Ga2[Gi == 2] = 1

            fst = allel.average_hudson_fst(allel.GenotypeArray(Ga1).count_alleles(), allel.GenotypeArray(Ga2).count_alleles(), blen=int(self.sp_graph.genotypes.shape[1]/10))[0]

            ls.append([self.perm_idx[x[-1]], self.perm_idx[y[-1]], tuple(self.sp_graph.nodes[self.perm_idx[x[-1]]]['pos'][::-1]), tuple(self.sp_graph.nodes[self.perm_idx[y[-1]]]['pos'][::-1]), pvals[k], emp_dist[k]-fit_dist[k], fst])

        rm = []
        newls = []
        for k in range(len(ls)):
            # checking the log-lik of fits with deme1 - deme2 to find the source & dest. deme
            resc = minimize(self.eems_neg_log_lik, x0=np.random.random(), args={'edge':[(ls[k][0],ls[k][1])],'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)], tol=1e-3)
            rescopp = minimize(self.eems_neg_log_lik, x0=np.random.random(), args={'edge':[(ls[k][1],ls[k][0])],'mode':'compute'}, method='L-BFGS-B', bounds=[(0,1)], tol=1e-3)
            # resc = minimize(self.neg_log_lik_c, x0=6*np.random.random()-3, args={'lre':[(x[k],y[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(-3,3)])
            # rescopp = minimize(self.neg_log_lik_c, x0=6*np.random.random()-3, args={'lre':[(y[k],x[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(-3,3)])
            if resc.x<1e-3 and rescopp.x<1e-3:
                rm.append(k)
            else:
                # approximately similar likelihood of either deme being destination 
                if np.abs(rescopp.fun - resc.fun) <= 5:
                    newls.append([self.perm_idx[y[k]], self.perm_idx[x[k]], tuple(self.sp_graph.nodes[self.perm_idx[y[k]]]['pos'][::-1]), tuple(self.sp_graph.nodes[self.perm_idx[x[k]]]['pos'][::-1]), pvals[k], emp_dist[k]-fit_dist[k], ls[k][-1]])
                else:
                    # if the "opposite" direction has a much higher log-likelihood then replace it entirely 
                    if rescopp.fun < resc.fun:
                        ls[k][0] = self.perm_idx[y[k]]
                        ls[k][1] = self.perm_idx[x[k]]

        ls += newls

        # removing demes that have estimated admix. prop. \approx 0 
        for i in sorted(rm, reverse=True):
            del ls[i]
        
        df = pd.DataFrame(ls, columns = ['source', 'dest.', 'source (lat., long.)', 'dest. (lat., long.)', 'pval', 'raw diff.', 'Fst'])

        if len(df)==0:
            print('No outliers found.')
            print('Consider raising the significance threshold slightly.')
            return None
        else:
            print('{:d} outlier deme pairs found'.format(len(df)))
            if verbose:
                print(df.sort_values(by='pval').to_string(index=False))
                print('\nPutative destination demes (and # of times the deme appears as an outlier) experiencing admixture:')
                print(df['dest.'].value_counts())
            else:
                print('\nPutative destination demes:{}'.format(np.unique(df['dest.'])))
            return df.sort_values('pval', ascending=True)
    
    def calc_contour(self, destid, search_area='all', sourceid=None, opts=None, exclude_boundary=True, delta=None):
        """
        Function to calculate admix. prop. values along with log-lik. values in a contour around the sampled source deme to capture uncertainty in the location of the source. 
        destid 
        The flag search_area is used to signifiy how large the contour should be:
            'all'    - include all demes (sampled & unsampled) from the entire graph
            'radius' - include all demes within a certain radius of a sampled source deme 
                - 'opts' : integer specifying radius (as an `int`) around the sampled source deme
            'range'  - include all demes within a certain long. & lat. rectangle 
                - 'opts' : list of lists specifying long. & lat. limits (e.g., [[-120,-70],[25,50]] for contiguous USA)
            'custom' - specific array of deme ids
                - 'opts' : list of specific deme ids as index
        """
        assert isinstance(destid, (int, np.integer)), "destid must be an integer"

        try:
            destpid = np.where(self.perm_idx[:self.sp_graph.n_observed_nodes]==destid)[0][0] #-> 0:(o-1)
        except:
            print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
            return None

        # creating a list of (source, dest.) pairings based on user-picked criteria
        if search_area == 'all':
            # including every possible node in graph as a putative source
            randedge = [(x,destid) for x in list(set(range(self.sp_graph.number_of_nodes()))-set([destid]))]
        elif search_area == 'radius':
            assert isinstance(sourceid, (int, np.integer)), "sourceid must be an integer"
            assert isinstance(opts, (int, np.integer)) and opts > 0, "radius must be an integer >=1"
            
            neighs = [] 
            neighs = list(self.sp_graph.neighbors(sourceid)) + [sourceid]

            # including all nodes within a certain radius
            for _ in range(opts-1):
                tempn = [list(self.sp_graph.neighbors(n1)) for n1 in neighs]
                # dropping repeated nodes 
                neighs = np.unique(list(it.chain(*tempn)))

            randedge = [(x,destid) for x in list(set(neighs)-set([destid]))]
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
            for n in range(self.sp_graph.number_of_nodes()):
                # checking for lat. & long. of all possible nodes in graph
                if self.sp_graph.nodes[n]['pos'][0] > opts[0][0] and self.sp_graph.nodes[n]['pos'][0] < opts[0][1]:
                    if self.sp_graph.nodes[n]['pos'][1] > opts[1][0] and self.sp_graph.nodes[n]['pos'][1] < opts[1][1]:
                        randedge.append((n,destid))

            # remove tuple of dest -> dest ONLY if it is in randedge
            if (destid,destid) in randedge:
                randedge.remove((destid,destid))
        elif search_area == 'custom':
            randedge = [(x,destid) for x in list(set(opts)-set([destid]))]

        # only include central demes (==6 neighbors), since demes on edge of range exhibit some boundary effects during estimation
        # randedge = list(it.compress(randedge,np.array([sum(1 for _ in self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[i])) for i in list(set(range(self.sp_graph.number_of_nodes()))-set([destid]))])==6))
        if exclude_boundary:
            randedge = [(e[0], e[1]) for e in randedge if sum(1 for _ in self.sp_graph.neighbors(e[0]))==6]

        if not hasattr(self, 'Lpinv'):
            self.Lpinv = np.linalg.pinv(self.sp_graph.L.T.todense())

        # just want to perturb it a bit instead of updating the entire matrix
        args = {}
        args['mode'] = 'compute'
        if 'delta' in args:
            args['delta'] = np.copy(delta)
        else:
            # adding a dummy edge in since c=0 doesn't change any terms anyway
            args['delta'] = self._compute_delta_matrix(0, {'lre':[(0, 2)], 'mode':'compute'})
        
        # randpedge = []
        cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
        print("Optimizing likelihood over {:d} demes in the graph".format(len(randedge)),end='...')
        checkpoints = {int(np.percentile(range(len(self.perm_idx)),25)): 25, int(np.percentile(range(len(self.perm_idx)),50)): 50, int(np.percentile(range(len(self.perm_idx)),75)): 75}
        for ie, e in enumerate(randedge):

            if ie in checkpoints:
                print('{:d}%'.format(checkpoints[ie]), end='...')
            
            # convert all sources to valid permuted ids (so observed demes should be b/w index 0 & o-1)
            # e2 = (np.where(self.perm_idx==e[0])[0][0], destpid) # -> contains the permuted ids, so 0:(o-1) is sampled (useful for indexing Linv & Lpinv)
            # randpedge.append((e[0],destid)) # -> contains the *un*permuted ids (useful for external viz)
            args['edge'] = [e]
            try:
                res = minimize(self.eems_neg_log_lik, x0 = 0.1, bounds=[(0,1)], tol=1e-2, method='L-BFGS-B', args=args)
                cest2[ie] = res.x; llc2[ie] = res.fun
            except:
                cest2[ie] = np.nan; llc2[ie] = np.nan

        print('done!')
        ## TODO: if MLE is found to be on the edge of the range specified by user then indicate that range should be extended
        df = pd.DataFrame(index=range(1,len(randedge)+1), columns=['(source, dest.)', 'admix. prop.', 'log-lik', 'scaled log-lik'])
        df['(source, dest.)'] = randedge; df['admix. prop.'] = cest2; df['log-lik'] = -llc2; df['scaled log-lik'] = df['log-lik']-np.nanmax(df['log-lik'])

        if np.sum(df['log-lik'].isna()) > 0.3*len(df):
            print("(warning: log-likelihood could not be computed for ~{:.2f}% of demes)".format(np.sum(df['log-lik'].isna())*100/len(df)))
            
        return df#.dropna()


    def calc_joint_contour(self, contour_df=None, top=0.01, lamb=None, lamb_q=None, destid=None, search_area='all', sourceid=None, opts=None, exclude_boundary=True, usew=None, uses2=None):
        """
        Function to calculate admix. prop. values in a joint manner with weights w & deme-specific variance s2 (as opposed to just admix. prop. values in `calc_contour`).
        contour_df (:obj:`pd.DataFrame`) : data frame containing the output from the function `calc_contour` 
        top (:obj:`float`) : how many top entries (based on log-lik) to consider for the joint fitting? (if top >= 1, then it is the number of top entries, but if top < 1 then it is the top percent of total entries to consider)
        NOTE: if the above two flags are specified, then none of the flags below need to be specified. 
        destid (:obj:`int`) : 
        The flag coverage is used to signifiy how large the contour should be:
            'all'    - include all demes (sampled & unsampled) from the entire graph
            'radius' - include all demes within a certain radius of a sampled source deme 
                - 'opts' : integer specifying radius (as an `int`) around the sampled source deme
            'range'  - include all demes within a certain long. & lat. rectangle 
                - 'opts' : list of lists specifying long. & lat. limits (e.g., [[-120,-70],[25,50]] for contiguous USA)
            'custom' - specific array of deme ids
                - 'opts' : list of specific deme ids as index
        """
        if contour_df is None:
            assert isinstance(lamb, (float, np.float)) and lamb >= 0, "lamb must be a float >=0"
            assert isinstance(lamb_q, (float, np.float)) and lamb_q >= 0, "lamb_q must be a float >= 0"

            assert isinstance(destid, (int, np.integer)), "destid must be an integer"
            try:
                destpid = np.where(self.perm_idx[:self.sp_graph.n_observed_nodes]==destid)[0][0]
            except:
                print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
                return None
            # creating a list of (source, dest.) pairings based on user-picked criteria
            if search_area == 'all':
                # including every possible node in graph as a putative source
                randedge = [(x,destid) for x in list(set(range(self.sp_graph.number_of_nodes()))-set([destid]))]
            elif search_area == 'radius':
                assert isinstance(sourceid, (int, np.integer)), "sourceid must be an integer"
                assert isinstance(opts, (int, np.integer)) and opts > 0, "radius must be an integer >=1"
                
                neighs = [] 
                neighs = list(self.sp_graph.neighbors(sourceid)) + [sourceid]
    
                # including all nodes within a certain radius
                for _ in range(opts-1):
                    tempn = [list(self.sp_graph.neighbors(n1)) for n1 in neighs]
                    # dropping repeated nodes 
                    neighs = np.unique(list(it.chain(*tempn)))
    
                randedge = [(x,destid) for x in list(set(neighs)-set([destid]))]
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
                for n in range(self.sp_graph.number_of_nodes()):
                    # checking for lat. & long. of all possible nodes in graph
                    if self.sp_graph.nodes[n]['pos'][0] > opts[0][0] and self.sp_graph.nodes[n]['pos'][0] < opts[0][1]:
                        if self.sp_graph.nodes[n]['pos'][1] > opts[1][0] and self.sp_graph.nodes[n]['pos'][1] < opts[1][1]:
                            randedge.append((n,destid))
    
                # remove tuple of dest -> dest ONLY if it is in randedge
                if (destid,destid) in randedge:
                    randedge.remove((destid,destid))
            elif search_area == 'custom':
                randedge = [(x,destid) for x in list(set(opts)-set([destid]))]
    
            if exclude_boundary:
                randedge = [(e[0], e[1]) for e in randedge if sum(1 for _ in self.sp_graph.neighbors(e[0]))==6]

            if not hasattr(self, 'Lpinv'):
                self.Lpinv = np.linalg.pinv(self.sp_graph.L.T.todense())

            # fit the baseline graph if no w or s2 is passed in 
            # baseline w and s2 to be stored in an object
            if usew is None:
                self.sp_graph.fit(lamb=lamb, optimize_q='n-dim', lamb_q=lamb_q, option='default', verbose=False);
                self.inv(); self.grad(reg=False)
                baselinell = -self.eems_neg_log_lik()
                usew = deepcopy(self.sp_graph.w); uses2 = deepcopy(self.sp_graph.s2) 
                # container for storing the MLE weights & s2
                mlew = deepcopy(usew); mles2 = deepcopy(uses2)
            else:
                self._update_graph(usew, uses2)
                # container for storing the MLE weights & s2
                mlew = deepcopy(usew); mles2 = deepcopy(uses2)            
            
            cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
            print("Optimizing likelihood over {:d} demes in the graph".format(len(randedge)),end='...')
            checkpoints = {int(np.percentile(range(len(self.perm_idx)),25)): 25, int(np.percentile(range(len(self.perm_idx)),50)): 50, int(np.percentile(range(len(self.perm_idx)),75)): 75}
            
            for ie, e in enumerate(randedge):

                if ie in checkpoints:
                    print('{:d}%'.format(checkpoints[ie]), end='...')
                    
                # initializing at baseline values
                self._update_graph(usew, uses2)

                ## TODO CHANGE THIS
                # using a try/except here as some configs lead to not pos def errors in simulations
                try:
                    self.sp_graph.fit(lamb=lamb, optimize_q='n-dim', lamb_q=lamb_q, long_range_edges=[e], option='onlyc', verbose=False)
                    cest2[ie] = self.sp_graph.c
                    llc2[ie] = -self.eems_neg_log_lik(self.sp_graph.c, {'edge':self.sp_graph.edge,'mode':'compute'})
                    # updating the MLE weights if the new log-lik is higher than the previous one (if not, keep the previous values)
                    if llc2[ie] > np.max(llc2[:ie]):
                        mlew = deepcopy(self.sp_graph.w); mles2 = deepcopy(self.sp_graph.s2)
                except: 
                    self._update_graph(usew, uses2)

                    # using a larger lambda value to avoid not pos def error from Cholesky decompoisition
                    try:
                        self.sp_graph.fit(lamb=lamb*10, optimize_q='n-dim', lamb_q=lamb_q*10, long_range_edges=[e], option='onlyc', verbose=False)
                        cest2[ie] = self.sp_graph.c
                        llc2[ie] = -self.eems_neg_log_lik(self.sp_graph.c, {'edge':self.sp_graph.edge,'mode':'compute'})
                        if llc2[ie] > np.max(llc2[:ie]):
                            mlew = deepcopy(self.sp_graph.w); mles2 = deepcopy(self.sp_graph.s2)
                    except: 
                        cest2[ie] = np.nan
                        llc2[ie] = np.nan
    
            print('done!')

            ## TODO: if MLE is found to be on the edge of the range specified by user then indicate that range should be extended
            df = pd.DataFrame(index=range(1,len(randedge)+1), columns=['(source, dest.)', 'admix. prop.', 'log-lik', 'scaled log-lik'])
            df['(source, dest.)'] = randedge; df['admix. prop.'] = cest2; df['log-lik'] = llc2
    
            if np.sum(df['log-lik'].isna()) > 0.25*len(df):
                print("(warning: log-likelihood could not be computed for ~{:.0f}% of demes)".format(np.sum(df['log-lik'].isna())*100/len(df)))

            joint_contour_df = df
        else:
            assert isinstance(lamb, (float, np.float)) and lamb >= 0, "lamb must be a float >=0"
            assert isinstance(lamb_q, (float, np.float)) and lamb_q >= 0, "lamb_q must be a float >= 0"
            # get indices of the top hits
            if top<1:
                # treat as a percentage
                topidx = contour_df['log-lik'].nlargest(int(top * len(contour_df))).index
            else: 
                # treat as a number 
                topidx = contour_df['log-lik'].nlargest(int(top)).index
            print("Jointly optimizing likelihood over {:d} demes in the graph...".format(len(topidx)))

            if usew is None:
                # baseline w and s2 to be stored in an object
                self.sp_graph.fit(lamb=lamb, optimize_q='n-dim', lamb_q=lamb_q, option='default', verbose=False);
                self.inv(); self.grad(reg=False)
                baselinell = -self.eems_neg_log_lik()
                usew = deepcopy(self.sp_graph.w); uses2 = deepcopy(self.sp_graph.s2)
                mlew = deepcopy(usew); mles2 = deepcopy(uses2) 
            else:
                self._update_graph(usew, uses2)
                mlew = deepcopy(usew); mles2 = deepcopy(uses2) 
            
            # run the joint fitting scheme for each top hit
            joint_contour_df = contour_df.loc[topidx]
            for i, row in joint_contour_df.iterrows():
                # print(row['(source, dest.)'])
                # initializing at baseline values
                # TODO make these baseline w & s2 as private variables? or even better, create a function that takes as input the w & s2 and updates the sp_graph object?
                self._update_graph(usew, uses2)

                # using a try/except here as some configs lead to not pos def errors in simulations
                try:
                    self.sp_graph.fit(lamb=lamb, optimize_q='n-dim', lamb_q=lamb_q, long_range_edges=[row['(source, dest.)']], option='onlyc', verbose=True)
                    joint_contour_df.at[i, 'admix. prop.'] = self.sp_graph.c
                    joint_contour_df.at[i, 'log-lik'] = -self.eems_neg_log_lik(self.sp_graph.c, {'edge':self.sp_graph.edge,'mode':'compute'})
                    # updating the MLE weights if the new log-lik is higher than the previous one (if not, keep the previous values)
                    if joint_contour_df.at[i, 'log-lik'] > np.max(joint_contour_df['log-lik'].loc[:i]):
                        mlew = deepcopy(self.sp_graph.w); mles2 = deepcopy(self.sp_graph.s2)
                except: 
                    self._update_graph(usew, uses2)

                    # using a larger lambda value to avoid not pos def error from Cholesky decompoisition
                    try:
                        self.sp_graph.fit(lamb=lamb*10, optimize_q='n-dim', lamb_q=lamb_q*10, long_range_edges=[row['(source, dest.)']], option='onlyc', verbose=True)
                        joint_contour_df.at[i, 'admix. prop.'] = self.sp_graph.c
                        joint_contour_df.at[i, 'log-lik'] = -self.eems_neg_log_lik(self.sp_graph.c, {'edge':self.sp_graph.edge,'mode':'compute'})
                        if joint_contour_df.at[i, 'log-lik'] > np.max(joint_contour_df['log-lik'].loc[:i]):
                            mlew = deepcopy(self.sp_graph.w); mles2 = deepcopy(self.sp_graph.s2)
                    except: 
                        joint_contour_df.at[i, 'admix. prop.'] = np.nan
                        joint_contour_df.at[i, 'log-lik'] = np.nan

        joint_contour_df['scaled log-lik'] = joint_contour_df['log-lik'] - np.nanmax(joint_contour_df['log-lik']) 

        # updating the graph with MLE weights so it does not need to be fit again
        # print(mlew[:10], self.sp_graph.w[:10])
        self._update_graph(mlew, mles2)
        # print(self.eems_neg_log_lik())

        # # checking whether adding an extra admixture parameter improves model fit using a LRT
        # joint_contour_df['pval'] = chi2.sf(2*(joint_contour_df['log-lik'] - baselinell), df=1)
        
        return joint_contour_df                
              
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

def coordinate_descent(obj, factr=1e7, m=10, maxls=50, maxiter=100, verbose=False):
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
            # TODO print a more explanatory message here
            print('Warning: admix. prop. optimization failed (consider increasing factr)')
            return None
        if np.allclose(resc.x, obj.sp_graph.c, atol=1e-3):
            optimc = False

        obj.sp_graph.c = deepcopy(resc.x)
        # print(resc.x)
        # obj.sp_graph.c = deepcopy(10**resc.x/(1+10**resc.x))

        ## TODO need an option here for 1-dim (currently only have n-dim and None options)
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

class FEEMSmix_SpatialGraph(SpatialGraph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True,):
        """Inherit from the feems object SpatialGraph and overwrite some methods for 
        estimation of edge weights and residual variance jointly
        """               
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps,)
        
    # ------------------------- Data -------------------------        
        
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
        # TODO check whether FEEMSmix matters here (prob want to just keep Objective)
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
                    "train_loss={:.3f}\n"
                ).format(res.nfev, self.train_loss)
            )  

    def sequential_fit(
        self, 
        fdr=0.25, 
        pval=0.05,
        stop=5,
        maxls=50,
        m=10,
        factr=1e7,
        lb=-1e10,
        ub=1e10,
        maxiter=15000,
        lamb=None,
        lamb_q=None,
    ):
        """
        Function to iteratively fit a long range gene flow event to the graph until there are no more outliers (alternate method)
        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            optimize_q (:obj:'str'): indicator whether optimizing residual variances (one of 'n-dim', '1-dim' or None)
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
        
        # TODO: include code here to fit the baseline FEEMS plot as well (with the same options)
    

        obj = FEEMSmix_Objective(self)
        # TODO: is there a faster way to calculate this pinv?
        obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.T.todense()); 
        obj.grad(reg=False)

        # dict storing all the results for plotting
        results = {}

        # store the deme id of each consecutive maximum outlier
        destid = []; nll = []

        # passing in dummy variables just to initialize the procedure
        opts = {'edge':[(0,obj.perm_idx[0])], 'mode':'update'}
        nll.append(obj.eems_neg_log_lik(0 , opts))
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

        # X = sm.add_constant(fit_dist)
        # mod = sm.OLS(emp_dist, X)
        # res = mod.fit()
        # muhat, betahat = res.params
        # plt.plot(fit_dist, emp_dist, 'o', color='grey', alpha=0.1, markersize=5)
        # plt.axline((np.min(fit_dist), muhat+np.min(fit_dist)), slope=betahat, color='orange', ls='--', lw=3) 
        # plt.grid(); plt.box(False); plt.xlabel('fit distance (baseline FEEMS)'); plt.ylabel('empirical distance')
        # # plt.plot(res_dist.T[mvals<-3], emp_dist[mvals<-3], 'k+', alpha=0.8, markersize=8)
        # plt.text(np.mean(fit_dist), np.mean(emp_dist)*0.2, "R²={:.3f}".format(np.around(res.rsquared_adj,3)), size='x-large'); plt.show()

        cnt = 1; keepgoing = True
        while keepgoing and cnt <= stop:
            print('\nFitting long range edge to deme {:d}...'.format(destid[-1]))
            # fit the contour on the deme to get the log-lik surface across the landscape
            # TODO: include the options in the function definition above
            df = obj.calc_contour(destid=int(destid[-1]), search_area='all', delta=opts['delta'])
            usew = deepcopy(obj.sp_graph.w); uses2 = deepcopy(obj.sp_graph.s2)
            joint_df = obj.calc_joint_contour(df, top=10, lamb=lamb, lamb_q=lamb_q, usew=usew, uses2=uses2)
            # print(obj.eems_neg_log_lik())

            # TODO check whether the log-lik is being updated to be the joint MLE instead of the point MLE (w & s2 seems to be updated) -> as the next edge should be built on the weights estimated from the previous edge? 

            nll.append(-np.nanmax(joint_df['log-lik']))
            print('\nLog-likelihood after fitting deme {:d}: {:.1f}'.format(destid[-1], -nll[-1]))
            
            opts['edge'] = [joint_df['(source, dest.)'].iloc[joint_df['log-lik'].argmax()]]; opts['mode'] = 'update'
            obj.eems_neg_log_lik(c=joint_df['admix. prop.'].iloc[np.argmax(joint_df['log-lik'])], opts=opts)
            # assert nll[-1] == obj.eems_neg_log_lik(c=joint_df['admix. prop.'].iloc[np.argmax(joint_df['log-lik'])], opts=opts), "difference in internal log-lik values (rerun the function)"
            # print('\nLog-likelihood after fitting deme {:d}: {:.1f}'.format(destid[-1], -nll[ -1]))

            if chi2.sf(2*(nll[-2]-nll[-1]), df=1) > pval: 
                print("Previous edge did not significantly increase the log-likelihood of the fit at a p-value of {:g}\n".format(pval))
                keepgoing=False
                break
            else:
                print("Previous edge to deme {:d} significantly increased the log-likelihood of the fit.\n".format(destid[-1]))

            res_dist = np.array(cov_to_dist(-0.5*opts['delta'])[np.tril_indices(self.n_observed_nodes, k=-1)])

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
        lb=-1e10,
        ub=1e10,
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
            optimize_q (:obj:'str'): indicator whether optimizing residual variances ('n-dim', '1-dim', None)
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
        # assert optimize_q is not None, "optimize_q should be one of '1-dim' or 'n-dim'"

        # creating a container to store these edges 
        self.edge = long_range_edges
        # mask for indices of edges in lre
        # self.lre_idx = np.array([val in self.lre for val in list(self.edges)])

        self.c = np.random.random(len(self.edge))
        self.t = np.array([0.04]) # -> should this just be zeros? it serves as init for optimizer and 0 might not be a good starting point...

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
            obj = FEEMSmix_Objective(self)
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

            obj = FEEMSmix_Objective(self)
            obj.sp_graph.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            if obj.sp_graph.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            # TODO: is there a faster way to calculate this pinv? ALSO does this need to be calculated here?
            obj.inv(); obj.Lpinv = np.linalg.pinv(self.L.T.todense()); 
            obj.grad(reg=False)
            res = coordinate_descent(
                obj=obj,
                factr=factr,
                m=m,
                maxls=maxls,
                maxiter=maxiter,
                verbose=verbose
            )

        # if maxiter >= 100:
        #     assert res[2]["warnflag"] == 0, "did not converge"
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
