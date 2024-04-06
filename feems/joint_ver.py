import sys
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from scipy.stats import wishart
from scipy.stats import norm
import networkx as nx
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b, minimize
import itertools as it

from .spatial_graph import SpatialGraph, query_node_attributes
from .objective import Objective, neg_log_lik_w0_s2, comp_mats
from .utils import cov_to_dist, mean_pairwise_differences_between

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
        self.optimize_q = 'n-dim'
        
        ## reg params for residual variance
        self.lamb_q = None
        self.alpha_q = None

        self.nll = 0.0

        self.C = np.vstack((-np.ones(self.sp_graph.n_observed_nodes-1),np.eye(self.sp_graph.n_observed_nodes-1))).T

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
        if self.optimize_q == 'n-dim':
            self.grad_obj_q = np.zeros(len(self.sp_graph))
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)                   
        elif self.optimize_q == '1-dim':
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

    def _comp_grad_obj_c(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c) with respect to the latent variables dLoss / dL
        """

        # compute inverses
        self._comp_inv_lap()

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        resmat = Rmat + (Q1mat + Q1mat.T) - 2*self.sp_graph.q_inv_diag

        for ie, _ in enumerate(self.sp_graph.lre):
            if(self.sp_graph.lre[ie][0]<self.sp_graph.n_observed_nodes and self.sp_graph.lre[ie][1]<self.sp_graph.n_observed_nodes):
                resmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] = (0.5*self.sp_graph.c[ie]**2-1.5*self.sp_graph.c[ie]+1)*Rmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] + (1+self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][0]] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][1]]
                resmat[self.sp_graph.lre[ie][1],self.sp_graph.lre[ie][0]] = resmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]]

                for i in set(range(self.sp_graph.n_observed_nodes))-set([self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]]):
                    resmat[i,self.sp_graph.lre[ie][1]] = (1-self.sp_graph.c[ie])*Rmat[i,self.sp_graph.lre[ie][1]] + self.sp_graph.c[ie]*Rmat[i,self.sp_graph.lre[ie][0]] + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*Rmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] + 1/self.sp_graph.q[i] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][1]] + self.sp_graph.c[ie]/self.sp_graph.q[self.sp_graph.lre[ie][0]]
                    resmat[self.sp_graph.lre[ie][1],i] = resmat[i,self.sp_graph.lre[ie][1]]
            else:
                #TODO: annotate this in more detail
                neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[self.sp_graph.lre[0][0]]))
                neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

                R1d = -2*self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][1]] + self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][0]] + self.Lpinv[self.sp_graph.lre[0][1],self.sp_graph.lre[0][1]]

                for s in neighs:
                    # convert back to appropriate indexing excluding the unsampled demes
                    s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==s][0]
                    resmat[s,self.sp_graph.lre[0][1]] = Rmat[s,self.sp_graph.lre[0][1]] + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*R1d + (1-self.sp_graph.c[ie])/self.sp_graph.q[s] + (1+self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[0][1]]
                    resmat[self.sp_graph.lre[0][1],s] = resmat[s,self.sp_graph.lre[0][1]]

                proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=self.sp_graph.lre[0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([self.sp_graph.lre[0][0]])])
                for i in set(range(self.sp_graph.n_observed_nodes))-set([self.sp_graph.lre[0][0],self.sp_graph.lre[0][1]]+neighs):
                    Ri1 = -2*self.Lpinv[i,self.sp_graph.lre[0][0]] + self.Lpinv[i,i] + self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][0]]
                    resmat[i,self.sp_graph.lre[0][1]] = (1-self.sp_graph.c[ie])*(Rmat[i,self.sp_graph.lre[0][1]]) + self.sp_graph.c[ie]*Ri1 + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*R1d + 1/self.sp_graph.q[i] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[0][1]] + self.sp_graph.c[ie]/self.sp_graph.q[proxs]
                    resmat[self.sp_graph.lre[0][1],i] = resmat[i,self.sp_graph.lre[0][1]]

        CRCt = np.linalg.inv(-0.5*self.C @ resmat @ self.C.T)
        if self.optimize_q == 'n-dim':
            M = self.C.T @ (CRCt @ (self.C @ self.sp_graph.S @ self.C.T) @ CRCt - CRCt) @ self.C
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
        if self.optimize_q == 'n-dim':
            self.grad_obj_q = np.zeros(len(self.sp_graph))
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)        
        else:
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

    def _comp_grad_obj_c_t(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c & admix. time t) with respect to the latent variables dLoss / dL
        """

        # compute inverses
        self._comp_inv_lap()

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        Tstar = Rmat + (Q1mat + Q1mat.T); Tstar[np.diag_indices_from(Tstar)] = 0

        dT0 = np.zeros_like(Tstar)

        for ie, _ in enumerate(self.sp_graph.lre):
            for i in list(set(range(dT0.shape[0]))-set([self.sp_graph.lre[ie][1]])):
                dT0[i,self.sp_graph.lre[ie][1]] = self.sp_graph.c[ie]*Tstar[i,self.sp_graph.lre[ie][0]] - self.sp_graph.c[ie]*Tstar[i,self.sp_graph.lre[ie][1]] 
                dT0[self.sp_graph.lre[ie][1],i] = dT0[i,self.sp_graph.lre[ie][1]]
            dT0[self.sp_graph.lre[ie][1],self.sp_graph.lre[ie][1]] = self.sp_graph.c[ie]*Tstar[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][0]] - self.sp_graph.c[ie]*Tstar[self.sp_graph.lre[ie][1],self.sp_graph.lre[ie][1]] 
            #self.sp_graph.c[ie]**2*Tstar[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][0]] + 2*self.sp_graph.c[ie]*(1-self.sp_graph.c[ie])*Tstar[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] + (self.sp_graph.c[ie]**2-2*self.sp_graph.c[ie])*Tstar[self.sp_graph.lre[ie][1],self.sp_graph.lre[ie][1]] 
            
        dTt = -self.sp_graph.t[ie]*(np.diag(1/self.sp_graph.q)@np.diag(np.diag(dT0)) + self.Linv@dT0 + dT0@self.Linv) + dT0

        resmat = Tstar + dTt; resmat[np.diag_indices_from(resmat)] = 0

        CRCt = np.linalg.inv(-0.5*self.C @ resmat @ self.C.T)
        if self.optimize_q == 'n-dim':
            M = self.C.T @ (CRCt @ (self.C @ self.sp_graph.S @ self.C.T) @ CRCt - CRCt) @ self.C
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
        if self.optimize_q == 'n-dim':
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
        
        if self.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
            
            term = alpha_q * self.sp_graph.s2 + np.log(
                1 - np.exp(-alpha_q * self.sp_graph.s2)
            )
            self.grad_pen_q = self.sp_graph.Delta_q.T @ self.sp_graph.Delta_q @ (lamb_q * term)
            self.grad_pen_q = self.grad_pen_q * (alpha_q / (1 - np.exp(-alpha_q * self.sp_graph.s2)))

    def loss(self):
        """Evaluate the loss function given the current params"""
        lamb = self.lamb
        alpha = self.alpha

        if self.sp_graph.option == 'default':
            lik = self.neg_log_lik()
        elif self.sp_graph.option == 'onlyc':
            # lik = self.neg_log_lik_c(self.sp_graph.c, opts={'mode':'sampled','lre':self.sp_graph.lre})
            lik = self.neg_log_lik_c(self.sp_graph.c, opts={'mode':'sampled','lre':self.sp_graph.lre})
        else:
            lik = self.neg_log_lik_c_t([self.sp_graph.c, self.sp_graph.t], opts={'lre':self.sp_graph.lre})
            print([self.sp_graph.c, self.sp_graph.t])

        term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
        term_1 = alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
        # print(pen)
                
        if self.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen += 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2  
            # print(str(pen)+"here") 

        # loss
        loss = lik + pen
        return loss 

    def neg_log_lik_c_t(self, ct, opts):
        """Evaluate the full negative log-likelihood for the given weights AND admix. prop. c + admix. time t
        """

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        # Rmat = -2*2/self.sp_graph.number_of_nodes()*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(2/self.sp_graph.number_of_nodes()*np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(2/self.sp_graph.number_of_nodes()*np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) 
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        # Q1mat = self.sp_graph.number_of_nodes()/2*np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        Tstar = Rmat + (Q1mat + Q1mat.T); Tstar[np.diag_indices(self.sp_graph.n_observed_nodes)] = 0 

        dT0 = np.zeros_like(Tstar)

        # dT0[opts['lre'][0][0],opts['lre'][0][1]] = ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 2*ct[0]*Q1mat[opts['lre'][0][0],opts['lre'][0][0]]
        # dT0[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*ct[0]**2-1.5*ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+ct[0])*self.Linv[opts['lre'][0][0],opts['lre'][0][0]] + ct[0]/self.sp_graph.q[opts['lre'][0][0]] + (1-ct[0])*self.Linv[opts['lre'][0][1],opts['lre'][0][1]] - ct[0]/self.sp_graph.q[opts['lre'][0][1]]
        # # dT0[opts['lre'][0][0],opts['lre'][0][1]] = ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 2*ct[0]*(1/self.sp_graph.q[opts['lre'][0][0]] + self.Linv[opts['lre'][0][0],opts['lre'][0][0]])
        # dT0[opts['lre'][0][1],opts['lre'][0][0]] = dT0[opts['lre'][0][0],opts['lre'][0][1]]
        # for i in list(set(range(dT0.shape[0]))-set([opts['lre'][0][0],opts['lre'][0][1]])):
        #     dT0[i,opts['lre'][0][1]] = ct[0]*Tstar[i,opts['lre'][0][0]] - ct[0]*Tstar[i,opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]]; dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]
            # dT0[i,opts['lre'][0][1]] = -ct[0]*Rmat[i,opts['lre'][0][1]] + ct[0]*Rmat[i,opts['lre'][0][0]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + self.Linv[i,i] - ct[0]/self.sp_graph.q[opts['lre'][0][1]] + (1-ct[0])*self.Linv[opts['lre'][0][1],opts['lre'][0][1]] + ct[0]*(1/self.sp_graph.q[opts['lre'][0][0]] + self.Linv[opts['lre'][0][0],opts['lre'][0][0]]); dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]
        # ct[0]**2*Tstar[opts['lre'][0][0],opts['lre'][0][0]] + 2*ct[0]*(1-ct[0])*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + (ct[0]**2-2*ct[0])*Tstar[opts['lre'][0][1],opts['lre'][0][1]]
        #ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - ct[0]*Tstar[opts['lre'][0][1],opts['lre'][0][1]]  -> has infs in array?  

        ## changing diag(1/q) to diag(1/q)+diag(Linv) does not change the estimates...
        # dTt = -ct[1]*(np.diag(1/self.sp_graph.q)@np.diag(np.diag(dT0)) + self.Linv@dT0 + dT0@self.Linv) + dT0
        # dTt = -ct[1]/(1+ct[1])*(np.diag(1/self.sp_graph.q)@np.diag(np.diag(Tstar+dT0)) + self.Linv@(Tstar+dT0) + (Tstar+dT0)@self.Linv + np.ones((self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes))) + Tstar+dT0

        # resmat = Tstar+dTt; resmat[np.diag_indices_from(resmat)] = 0

        if opts['mode']=='sampled':
            # dT0[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*ct[0]**2-1.5*ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + ct[0]/self.sp_graph.q[opts['lre'][0][0]] - ct[0]/self.sp_graph.q[opts['lre'][0][1]]
            dT0[opts['lre'][0][0],opts['lre'][0][1]] = ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 2*ct[0]*Q1mat[opts['lre'][0][0],opts['lre'][0][0]]
            dT0[opts['lre'][0][1],opts['lre'][0][0]] = dT0[opts['lre'][0][0],opts['lre'][0][1]]
            for i in list(set(range(dT0.shape[0]))-set([opts['lre'][0][0],opts['lre'][0][1]])):
                dT0[i,opts['lre'][0][1]] = ct[0]*Tstar[i,opts['lre'][0][0]] - ct[0]*Tstar[i,opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]]; dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]

            T0 = Tstar+dT0

            ## changing diag(1/q) to diag(1/q)+diag(Linv) does not change the estimates...
            resmat = T0 + ct[1]*(-np.diag(1/self.sp_graph.q)@np.diag(np.diag(T0)) - self.sp_graph.L@T0 - T0@self.sp_graph.L.T + np.ones((self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes))) 
            # resmat = T0 + ct[1]*(-np.diag(1/self.sp_graph.q)@np.diag(np.diag(T0)) - self.Linv@T0 - T0@self.Linv + np.ones((self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)))

        else:
            Tstar = Rmat + (Q1mat + Q1mat.T); Tstar[np.diag_indices(self.sp_graph.n_observed_nodes)] = 0 

            dT0 = np.zeros_like(Tstar)
            # gets the 6 neighboring demes
            neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[opts['lre'][0][0]]))
            # finds the neighboring deme that has samples
            neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

            R1d = -2*self.Lpinv[opts['lre'][0][0],opts['lre'][0][1]] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]] + self.Lpinv[opts['lre'][0][1],opts['lre'][0][1]]
            R1 = -2*self.Lpinv[:self.sp_graph.n_observed_nodes,opts['lre'][0][0]].T + np.diag(self.Linv) + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]

            # apply this formula only to neighboring sampled demes
            for n in neighs:
                # convert back to appropriate indexing excluding the unsampled demes
                s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==n][0]
                dT0[s,opts['lre'][0][1]] = ct[0]*Tstar[s,s] - ct[0]*Tstar[s,opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[s,opts['lre'][0][1]] + 2*ct[0]*Q1mat[s,s]
                dT0[opts['lre'][0][1],s] = dT0[s,opts['lre'][0][1]]
                # resmat[s,opts['lre'][0][1]] = Rmat[s,opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*R1d + (1-ct[0])/self.sp_graph.q[s] + (1+ct[0])/self.sp_graph.q[opts['lre'][0][1]]
                # resmat[opts['lre'][0][1],s] = resmat[s,opts['lre'][0][1]]

            # find the closest sampled deme to 1 (this is just the proxy source, use q from here but do not model this as the source)
            # proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])
            proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])[:3]
            qprox = np.dot(1/self.sp_graph.q[proxs], (1/R1[0,proxs].T)/np.sum(1/R1[0,proxs]))
            # print(proxs)
            ## id
            ## this needs a more alert mind and quiet morning (not one in which I've had <6 hours of sleep)
            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]+neighs):
                Ri1 = -2*self.Lpinv[i,opts['lre'][0][0]] + self.Lpinv[i,i] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
                resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Ri1 + 0.5*(c**2-c)*R1d + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c*qprox
                resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]
                dT0[i,opts['lre'][0][1]] = ct[0]*Tstar[i,opts['lre'][0][0]] - ct[0]*Tstar[i,opts['lre'][0][1]] + 0.5*(ct[0]**2-ct[0])*Rmat[opts['lre'][0][0],opts['lre'][0][1]]
                dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]



        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        return nll

    def joint_neg_log_lik_c_t(self, x0, opts):
        """Evaluate the joint negative log-likelihood for the given weights, s2 and admix. prop. c & admix. time t, but taking as input all four sets of parameters
        (prototype function: only works with option 'n-dim')
        """

        n_edges = self.sp_graph.size()
        self.sp_graph.comp_graph_laplacian(np.exp(x0[:n_edges]))
        self.sp_graph.comp_precision(s2=np.exp(x0[n_edges:-2]))
        self.inv()

        c = x0[-2]
        t = x0[-1]/(1+x0[-1])

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) 
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) 
        Tstar = Rmat + (Q1mat + Q1mat.T); Tstar[np.diag_indices(self.sp_graph.n_observed_nodes)] = 0 

        dT0 = np.zeros_like(Tstar)

        dT0[opts['lre'][0][0],opts['lre'][0][1]] = c*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - c*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 2*c/self.sp_graph.q[opts['lre'][0][0]]; dT0[opts['lre'][0][1],opts['lre'][0][0]] = dT0[opts['lre'][0][0],opts['lre'][0][1]]
        for i in list(set(range(dT0.shape[0]))-set([opts['lre'][0][0],opts['lre'][0][1]])):
            # dT0[i,opts['lre'][0][1]] = c*Tstar[i,opts['lre'][0][0]] - c*Tstar[i,opts['lre'][0][1]]; dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]
            dT0[i,opts['lre'][0][1]] = c*Tstar[i,opts['lre'][0][0]] - c*Tstar[i,opts['lre'][0][1]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]]; dT0[opts['lre'][0][1],i] = dT0[i,opts['lre'][0][1]]
        dT0[opts['lre'][0][1],opts['lre'][0][1]] = c*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - c*Tstar[opts['lre'][0][1],opts['lre'][0][1]]  
        #ct[0]**2*Tstar[opts['lre'][0][0],opts['lre'][0][0]] + 2*ct[0]*(1-ct[0])*Tstar[opts['lre'][0][0],opts['lre'][0][1]] + (ct[0]**2-2*ct[0])*Tstar[opts['lre'][0][1],opts['lre'][0][1]]
        #ct[0]*Tstar[opts['lre'][0][0],opts['lre'][0][0]] - ct[0]*Tstar[opts['lre'][0][1],opts['lre'][0][1]]  -> has infs in array?  

        # dTt = -t*(np.diag(1/self.sp_graph.q)@np.diag(np.diag(dT0)) + self.Linv@dT0 + dT0@self.Linv) + dT0

        resmat = Tstar+dTt; resmat[np.diag_indices_from(resmat)] = 0

        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        # nll = -wishart.logpdf(2*self.sp_graph.n_snps*self.C @ self.sp_graph.S @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)
        nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = self.alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * self.lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2

        term_0 = 1.0 - np.exp(-self.alpha_q * self.sp_graph.s2)
        term_1 = self.alpha_q * self.sp_graph.s2 + np.log(term_0)
        pen += 0.5 * self.lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2

        return nll + pen

    def joint_neg_log_lik_c(self, x0, opts):
        """Evaluate the joint negative log-likelihood for the given weights, s2 and admix. prop. c, but taking as input all three sets of parameters
        (prototype function: only works with option 'n-dim')
        """

        n_edges = self.sp_graph.size()
        self.sp_graph.comp_graph_laplacian(np.exp(x0[:n_edges]))
        self.sp_graph.comp_precision(s2=np.exp(x0[n_edges:-1]))
        self.inv(); #self.grad(reg=False)

        c = x0[-1]
        # print(c)

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) 
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) 
        resmat = Rmat + (Q1mat + Q1mat.T); resmat[np.diag_indices(self.sp_graph.n_observed_nodes)] = 0 

        resmat[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+c)/self.sp_graph.q[opts['lre'][0][0]] + (1-c)/self.sp_graph.q[opts['lre'][0][1]]
        resmat[opts['lre'][0][1],opts['lre'][0][0]] = resmat[opts['lre'][0][0],opts['lre'][0][1]]

        for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]):
            resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,opts['lre'][0][0]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[opts['lre'][0][0]]
            resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]

        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S

        nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = self.alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * self.lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2

        term_0 = 1.0 - np.exp(-self.alpha_q * self.sp_graph.s2)
        term_1 = self.alpha_q * self.sp_graph.s2 + np.log(term_0)
        pen += 0.5 * self.lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2

        return nll + pen

    def neg_log_lik_c(self, c, opts):
        """Evaluate the full negative log-likelihood for the given weights & admix. prop. c
        (changing function to only take in single long-range edge with a flag for 'sampled' vs 'unsampled' source)
        """

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        # Rmat = -2*2/self.sp_graph.number_of_nodes()*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(2/self.sp_graph.number_of_nodes()*np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(2/self.sp_graph.number_of_nodes()*np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) 
        Q1mat = np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) #np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        ## both variations below gives not posdef errors with 6x6
        # Q1mat = 0.5*(1-self.Linv.diagonal()).reshape(-1,1) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1) 
        # Q1mat = self.sp_graph.number_of_nodes()/2*np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        resmat = Rmat + (Q1mat + Q1mat.T); resmat[np.diag_indices(self.sp_graph.n_observed_nodes)] = 0 

        if opts['mode']=='sampled':
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
            # TODO: this is very slow ->  see if you can speed up by using permuted_idx vector
            # gets the 6 neighboring demes
            neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[opts['lre'][0][0]]))
            # finds the neighboring deme that has samples
            neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

            R1d = -2*self.Lpinv[opts['lre'][0][0],opts['lre'][0][1]] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]] + self.Lpinv[opts['lre'][0][1],opts['lre'][0][1]]
            R1 = np.array(-2*self.Lpinv[:self.sp_graph.n_observed_nodes,opts['lre'][0][0]].T + np.diag(self.Linv) + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]])

            # apply this formula only to neighboring sampled demes
            for n in neighs:
                # convert back to appropriate indexing excluding the unsampled demes
                s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==n][0]
                # (1+c)q_s gives an overestimate of the c value (slide 61) ->  keeping it at 1-c
                resmat[s,opts['lre'][0][1]] = Rmat[s,opts['lre'][0][1]] + 0.5*(c**2-c)*R1d + (1-c)/self.sp_graph.q[s] + (1+c)/self.sp_graph.q[opts['lre'][0][1]]
                resmat[opts['lre'][0][1],s] = resmat[s,opts['lre'][0][1]]

            # find the closest sampled deme to 1 (this is just the proxy source, use q from here but do not model this as the source)
            # proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])
            # proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])[:1]
            ## TODO: cache entries for a specific unsampled node here (so rerunning on a new destination will be much faster)
            proxs = np.argsort([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set(self.perm_idx[:self.sp_graph.n_observed_nodes])-set([opts['lre'][0][0]])])[:self.sp_graph.n_observed_nodes]
            # qprox = np.dot(1/self.sp_graph.q[proxs], (1/R1[0,proxs].T)/np.sum(1/R1[0,proxs]))
            qprox = np.dot(1/self.sp_graph.q[proxs], (R1[0,proxs]*np.exp(-2*R1[0,proxs]))/np.sum(R1[0,proxs]*np.exp(-2*R1[0,proxs])))
            ## id
            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]+neighs):
                Ri1 = -2*self.Lpinv[i,opts['lre'][0][0]] + self.Lpinv[i,i] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
                # resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Ri1 + 0.5*(c**2-c)*R1d + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts  ['lre'][0][1]] + c/self.sp_graph.q[proxs]
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

        D = np.ones(self.sp_graph.n_observed_nodes).reshape(-1,1) @ np.diag(self.sp_graph.S).reshape(1,-1) + np.diag(self.sp_graph.S).reshape(-1,1) @ np.ones(self.sp_graph.n_observed_nodes).reshape(1,-1) - 2*self.sp_graph.S
        # basically, smaller the coefficient in front of CRCt, the more downward biased the admix. prop. estimates 
        # nll = -wishart.logpdf(2*self.sp_graph.n_snps*self.C @ self.sp_graph.S @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)
        nll = -wishart.logpdf(-self.sp_graph.n_snps*self.C @ D @ self.C.T, self.sp_graph.n_snps, -self.C @ resmat @ self.C.T)

        return nll
    
    def extract_outliers(self, pthresh=0.01, verbose=False):
        """Function to extract outlier deme pairs based on a p-value threshold specified by the user (default: 0.01)"""
        assert pthresh>0, "pthresh should be a positive number"

        # computing pairwise covariance & distances between demes
        fit_cov, _, emp_cov = comp_mats(self)
        fit_dist = cov_to_dist(fit_cov)[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)]
        emp_dist = cov_to_dist(emp_cov)[np.tril_indices(self.sp_graph.n_observed_nodes, k=-1)]

        self.perm_idx = query_node_attributes(self.sp_graph, "permuted_idx")

        print('Using a significance threhsold of {:g}:\n'.format(pthresh))
        ls = []; x, y = [], []
        acs = np.empty((2, self.sp_graph.n_snps, 2))
        # computing p-values (or z-values) for each pairwise comparison after mean centering
        pvals = norm.cdf(np.log(emp_dist)-np.log(fit_dist)-np.mean(np.log(emp_dist)-np.log(fit_dist)), 0, np.std(np.log(emp_dist)-np.log(fit_dist)))
        for k in np.where(pvals < pthresh)[0]:
            # code to convert single index to matrix indices
            x.append(np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1); y.append(int(k - 0.5*x[-1]*(x[-1]-1)))

            Gi = self.sp_graph.genotypes[self.sp_graph.nodes[self.perm_idx[x[-1]]]['sample_idx'], :]
            acs[0, :, 0] = np.sum(Gi, axis=0)
            acs[0, :, 1] = (2 * Gi.shape[0]) - np.sum(Gi, axis=0)

            Gi = self.sp_graph.genotypes[self.sp_graph.nodes[self.perm_idx[y[-1]]]['sample_idx'], :]
            acs[1, :, 0] = np.sum(Gi, axis=0)
            acs[1, :, 1] = (2 * Gi.shape[0]) - np.sum(Gi, axis=0)

            fst = mean_pairwise_differences_between(acs[0, :, :].astype(np.int32), acs[1, :, :].astype(np.int32))

            ls.append([self.perm_idx[x[-1]], self.perm_idx[y[-1]], tuple(self.sp_graph.nodes[x[-1]]['pos'][::-1]), tuple(self.sp_graph.nodes[y[-1]]['pos'][::-1]), pvals[k], emp_dist[k]-fit_dist[k], (self.sp_graph.nodes[self.perm_idx[x[-1]]]['n_samples'], self.sp_graph.nodes[self.perm_idx[y[-1]]]['n_samples']), fst])

        rm = []
        for k in range(len(ls)):
            # checking the log-lik of fits with deme1 - deme2 to find the source & dest. deme
            resc = minimize(self.neg_log_lik_c, x0=np.random.random(), args={'lre':[(x[k],y[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(0,1)], tol=1e-3)
            rescopp = minimize(self.neg_log_lik_c, x0=np.random.random(), args={'lre':[(y[k],x[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(0,1)], tol=1e-3)
            # resc = minimize(self.neg_log_lik_c, x0=6*np.random.random()-3, args={'lre':[(x[k],y[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(-3,3)])
            # rescopp = minimize(self.neg_log_lik_c, x0=6*np.random.random()-3, args={'lre':[(y[k],x[k])],'mode':'sampled'}, method='L-BFGS-B', bounds=[(-3,3)])
            if resc.x<1e-3 and rescopp.x<1e-3:
                rm.append(k)
            else:
                if rescopp.fun < resc.fun:
                    ls[k][0] = self.perm_idx[y[k]]
                    ls[k][1] = self.perm_idx[x[k]]

        # removing demes that have estimated admix. prop. ≈ 0 
        for i in sorted(rm, reverse=True):
            del ls[i]
        
        df = pd.DataFrame(ls, columns = ['source', 'dest.', 'source (lat., long.)', 'dest. (lat., long.)', 'pval', 'raw diff.', '# of samples (source, dest.)', 'Fst'])
        # TODO: check (lat., long.) values in df (seems to be wrong in the afroeurasia dataset)

        if len(df)==0:
            print('No outliers found.')
            print('Consider raising the significance threshold slightly.')
        else:
            print('{:d} outlier deme pairs found'.format(len(df)))
            if verbose:
                print(df.sort_values(by='pval').to_string(index=False))
            else:
                print(df.sort_values(by='pval').iloc[:6,:5].to_string(index=False))

            print('\nPutative destination demes (and # of times the deme appears as an outlier) experiencing admixture:')
            if verbose:
                print(df['dest.'].value_counts())
            else:
                print(df['dest.'].value_counts().head(5))

        return df
    
    def calc_contour(self, destid, search_area='all', sourceid=None, opts=None, exclude_boundary=True):
        """
        Function to calculate admix. prop. values along with log-lik. values in a contour around the sampled source deme to capture uncertainty in the location of the source. 
        The flag coverage is used to signifiy how large the contour should be:
            'all'    - include all demes (sampled & unsampled) from the entire graph
            'radius' - include all demes within a certain radius of a sampled source deme (sourceid output from extract_outliers)
                - 'opts' : integer specifying radius around the sampled source deme
            'range'  - include all demes within a certain long. & lat. rectangle 
                - 'opts' : list of lists specifying long. & lat. limits (e.g., [[-120,-70],[25,50]] for contiguous USA)
            'custom' - specific array of deme ids
                - 'opts' : list of specific deme ids
        """
        assert type(destid) == int, "destid must be an integer"

        try:
            destpid = np.where(self.perm_idx[:self.sp_graph.n_observed_nodes]==destid)[0][0] #-> 0:(o-1)
        except:
            print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')

        # creating a list of (source, dest.) pairings based on user-picked criteria
        if search_area == 'all':
            # including every possible node in graph as a putative source
            randedge = [(x,destid) for x in list(set(range(self.sp_graph.number_of_nodes()))-set([destid]))]
        elif search_area == 'radius':
            assert type(sourceid) == int, "sourceid must be an integer"
            assert type(opts) == int and opts > 0, "radius must be an integer >=1"
            try: 
                sourcepid = np.where(self.perm_idx[:self.sp_graph.n_observed_nodes]==sourceid)[0][0] #-> 0:(o-1)
            except: 
                print('invalid ID for source deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
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

        # TODO: check if Lpinv has been calculated, if not then calculate it
        if not hasattr(self, 'Lpinv'):
            self.Lpinv = np.linalg.pinv(self.sp_graph.L.todense())
        randpedge = []
        cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
        print("Optimizing likelihood over {:d} demes in the graph...".format(len(randedge)))
        for ie, e in enumerate(randedge):
            # TODO: put a progress bar here
            # TODO: int wrapping doesn't work very well here
            if int(ie*100/len(randedge))%25 == 0 and ie/len(randedge) < 1:
                print('{:d}%'.format(int(ie*100/len(randedge))), end='...')
            # convert all sources to valid permuted ids (so observed demes should be b/w index 0 & o-1)
            e2 = (np.where(self.perm_idx==e[0])[0][0], destpid) # -> contains the permuted ids, so 0:(o-1) is sampled (useful for indexing Linv & Lpinv)
            # randpedge.append((e[0],destid)) # -> contains the *un*permuted ids (useful for external viz)
            if e2[0]<self.sp_graph.n_observed_nodes:
                try:
                    res = minimize(self.neg_log_lik_c, x0=0.1, bounds=[(0,1)], tol=1e-2, method='L-BFGS-B', args={'lre':[e2],'mode':'sampled'})
                    # res = minimize(self.neg_log_lik_c, x0=np.log10(self.sp_graph.c/(1-self.sp_graph.c)), bounds=[(-3,3)], method='L-BFGS-B', args={'lre':[e2],'mode':'sampled'})
                    cest2[ie] = res.x; llc2[ie] = res.fun
                    # cest2[ie] = 10**res.x/(1+10**res.x); llc2[ie] = res.fun
                except:
                    cest2[ie] = np.nan; llc2[ie] = np.nan
            else:
                try:
                    res = minimize(self.neg_log_lik_c, x0=0.1, bounds=[(0,1)], tol=1e-2, method='L-BFGS-B', args={'lre':[e2],'mode':'unsampled'})
                    cest2[ie] = res.x; llc2[ie] = res.fun
                except:
                    print(e2)
                    cest2[ie] = np.nan; llc2[ie] = np.nan
        ## TODO: if MLE is found to be on the edge of the range specified by user then indicate that range should be extended
        ## TODO: if MLE admix. prop. < 1e-4 then just assign 0 
        ## TODO: if there are any nan values, drop it from df (and print warning message but nothing to be alarmed about unless 10%? of values dropped)
        df = pd.DataFrame(index=range(1,len(randedge)+1), columns=['(source, dest.)', 'admix. prop.', 'log-lik', 'scaled log-lik'])
        df['(source, dest.)'] = randedge; df['admix. prop.'] = cest2; df['log-lik'] = -llc2; df['scaled log-lik'] = df['log-lik']-np.nanmax(df['log-lik'])

        return df
              
def loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w,q) which returns the loss and gradient"""                
    n_edges = obj.sp_graph.size()
    if obj.optimize_q is not None:
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
    if obj.optimize_q is None:
        grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    ## VS: not penalizing q (or s2), only penalizing w
    elif obj.optimize_q == 'n-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2 + obj.grad_pen_q * obj.sp_graph.s2
    else:
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2    
    # if obj.optimize_q is not None:
    #     grad = np.zeros_like(theta)
    #     grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    #     grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2
    # else:
    #     grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w

    return (loss, grad)

def coordinate_descent(obj, factr=1e7, m=10, maxls=50, maxiter=50, verbose=False):
    """
    Minimize the negative log-likelihood iteratively with an admix. prop. c value & refit the new weights based on that until tolerance is reached. We assume the best-fit weights are passed in to this function as x0. 
    """
    if obj.sp_graph.option=='onlyc':
        # typically don't need 10 iterations, but set as a way to ensure convergence
        for bigiter in range(10):
            optimc = True
            print(bigiter,end='...')
            # first fit admix. prop. c
            if optimc: # stop it from overly optimizing over c
                resc = minimize(obj.neg_log_lik_c, x0=obj.sp_graph.c, args={'lre':obj.sp_graph.lre,'mode':'sampled'}, method='L-BFGS-B', bounds=[(0,1)])
                # resc = minimize(obj.neg_log_lik_c, x0=np.log10(self.sp_graph.c/(1-self.sp_graph.c)), bounds=[(-3,3)], method='L-BFGS-B', args={'lre':obj.sp_graph.lre,'mode':'sampled'})
                if resc.status != 0:
                    print('Warning: admix. prop. optimization failed')
                if np.min(np.abs(resc.x - obj.sp_graph.c)) > 1e-3:
                    optimc = False
                    if bigiter > 10:
                        break
                obj.sp_graph.c = deepcopy(resc.x)
                # obj.sp_graph.c = deepcopy(10**resc.x/(1+10**resc.x))

            if obj.optimize_q is not None:
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
            if maxiter >= 100:
                assert res[2]["warnflag"] == 0, "did not converge (increase maxiter)"
            if obj.optimize_q is not None:
                neww = np.exp(res[0][:obj.sp_graph.size()])
                news2 = np.exp(res[0][obj.sp_graph.size():])
            else:
                neww = np.exp(res[0])
                news2 = obj.sp_graph.s2
            # print('length of s2:',len(news2))

            if np.allclose(obj.sp_graph.w, neww, atol=1e-3) and np.allclose(obj.sp_graph.s2, news2, atol=1e-3):
                if verbose:
                    print('admix. prop. estimation converged in {} iterations!'.format(bigiter+1))
                break
            else: # update weights and s2 & continue
                obj.sp_graph.w = deepcopy(neww)
                obj.sp_graph.s2 = deepcopy(news2)
                obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.todense()); obj.grad(reg=False)
    elif obj.sp_graph.option=='bothct': # optimize over c & t
        for bigiter in range(20):
            optimct = True
            print(bigiter,end='...')
            # first fit admix. prop. c & admix. time t
            if optimct: # stop it from overly optimizing over c & t (should it be a while loop?)
                #TODO: optimize this upper bound for t as higher values lead to bad fits
                resct = minimize(obj.neg_log_lik_c_t, x0=[0.1,0.01], args={'lre':obj.sp_graph.lre}, method='L-BFGS-B', bounds=[(0,0.9),(0,0.1)])
                if resct.status != 0:
                    print('Warning: admix. prop. & admix. time optimization failed')
                if np.min(np.abs(resct.x[0] - obj.sp_graph.c)) > 1e-3 and np.min(np.abs(resct.x[1] - obj.sp_graph.t)) > 1e-4:
                    optimct = False
                    if bigiter > 20:
                        break
                obj.sp_graph.c = np.array([resct.x[0]])
                obj.sp_graph.t = np.array([resct.x[1]])

            if obj.optimize_q == 'n-dim':
                x0 = np.r_[np.log(obj.sp_graph.w), np.log(obj.sp_graph.s2)]
            elif obj.optimize_q == '1-dim':
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
            if maxiter >= 100:
                assert res[2]["warnflag"] == 0, "did not converge (increase maxiter)"
            if obj.optimize_q == 'n-dim':
                neww = np.exp(res[0][:obj.sp_graph.size()])
                news2 = np.exp(res[0][obj.sp_graph.size():])
            elif obj.optimize_q == '1-dim':
                neww = np.exp(res[0])
                news2 = obj.sp_graph.s2
            # print('length of s2:',len(news2))

            if np.allclose(obj.sp_graph.w, neww, atol=1e-6) and np.allclose(obj.sp_graph.s2, news2, atol=1e-6):
                if verbose:
                    print('admix. prop. estimation converged in {} iterations!'.format(bigiter+1))
                break
            else: # update weights and s2 & continue
                obj.sp_graph.w = deepcopy(neww)
                obj.sp_graph.s2 = deepcopy(news2)
                obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.todense()); obj.grad(reg=False)

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
        obj = FEEMSmix_Objective(self)
        res = minimize(neg_log_lik_w0_s2, [0.0, 0.0], method="Nelder-Mead", args=(obj))
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
        long_range_edges=[(0,0)]
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
        # assert optimize_q is not None, "optimize_q should be one of '1-dim' or 'n-dim'"

        # creating a container to store these edges 
        self.lre = long_range_edges
        # mask for indices of edges in lre
        # self.lre_idx = np.array([val in self.lre for val in list(self.edges)])

        self.c = np.random.random(len(self.lre))
        self.t = np.array([0.04]) # -> should this just be zeros? it serves as init for optimizer and 0 might not be a good starting point...

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
            obj.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            x0 = np.log(w_init)
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            s2_init = self.s2 if obj.optimize_q=="1-dim" else self.s2*np.ones(len(self))
            if obj.optimize_q is not None:
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
            obj.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            #TODO: just need the diagonals (is there an easy way to calculate this?)
            obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.todense()); 
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
        if obj.optimize_q is not None:
            self.w = np.exp(res[0][:self.size()])
            self.s2 = np.exp(res[0][self.size():])
        else:    
            self.w = np.exp(res[0])
            if self.option == 'onlyc':
                self.c = res[0][-len(self.lre):]
            elif self.option == 'bothct':
                self.c = res[0][(-len(self.lre)+1):-len(self.lre)]
                self.t = res[0][-len(self.lre):]

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