import sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import wishart
from scipy.optimize import fmin_l_bfgs_b, minimize, minimize_scalar
import matplotlib.pyplot as plt

from .spatial_graph import SpatialGraph
from .objective import Objective, neg_log_lik_w0_s2

class Joint_Objective(Objective): 
    def __init__(self, sp_graph, option='default'):
        """Inherit from the feems object Objective and overwrite some methods for evaluations 
        and gradient of feems objective when residual variance is estimated jointly with edge weights
        Args:
            sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
            option (string): indicating whether weights & admixture proportion are jointly estimated
        """
        super().__init__(sp_graph=sp_graph)   
        
        # indicator whether optimizing residual variance jointly with edge weights
        # None  : residual variance is holding fixed
        # 1-dim : single residual variance is estimated across all nodes
        # n-dim : node-specific residual variances are estimated 
        self.optimize_q = None
        
        # reg params for residual variance
        self.lamb_q = None
        self.alpha_q = None
        self.option = option

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
            
        if self.optimize_q == '1-dim':
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

    def _comp_grad_obj_joint(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c) with respect to the latent variables dLoss / dL
        ** does not work for cases with missing nodes i.e., all demes need to contain data ** 
        """

        # compute inverses
        self._comp_inv_lap()
        
        C = np.vstack((-np.ones(self.sp_graph.n_observed_nodes-1),np.eye(self.sp_graph.n_observed_nodes-1))).T

        lrn = self.sp_graph.lre
        
        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = (np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(1/self.sp_graph.q,(1,-1))).T
        resmat = -0.5*(Rmat + (Q1mat + Q1mat.T) - 2*np.diag(1/self.sp_graph.q))
        
        resmat[lrn[0][0],lrn[0][1]] = -0.5*((0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*Rmat[lrn[0][0],lrn[0][1]] + (1+self.sp_graph.c)/self.sp_graph.q[lrn[0][0]] + (1-self.sp_graph.c)/self.sp_graph.q[lrn[0][1]])
        resmat[lrn[0][1],lrn[0][0]] = resmat[lrn[0][0],lrn[0][1]]

        ## id
        for i in set(range(self.sp_graph.n_observed_nodes))-set([lrn[0][0],lrn[0][1]]):
            resmat[i,lrn[0][1]] = -0.5*((1-self.sp_graph.c)*(Rmat[i,lrn[0][1]]) + self.sp_graph.c*Rmat[i,lrn[0][0]] + 0.5*(self.sp_graph.c**2-self.sp_graph.c)*Rmat[lrn[0][0],lrn[0][1]] + 1/self.sp_graph.q[i] + (1-self.sp_graph.c)/self.sp_graph.q[lrn[0][1]] + self.sp_graph.c/self.sp_graph.q[lrn[0][0]])
            resmat[lrn[0][1],i] = resmat[i,lrn[0][1]]

        M = C.T@(np.linalg.inv(C@resmat@C.T)@(C@self.sp_graph.S@C.T)@np.linalg.inv(C@resmat@C.T)-np.linalg.inv(C@resmat@C.T))@C

        self.grad_obj_L = self.sp_graph.n_snps * M 

        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        self.grad_obj = gradD - gradW
    
        ## (1 x d2)
        # delldsig = self.sp_graph.n_snps * (self.Linv @ C.T@(np.linalg.inv(C@resmat@C.T)@(C@self.sp_graph.S@C.T)@np.linalg.inv(C@resmat@C.T)-np.linalg.inv(C@resmat@C.T))@C @ self.Linv.T).reshape(1,-1)

        # ## (d2 x d2) 
        dsigdL = np.zeros((self.sp_graph.n_observed_nodes**2,self.sp_graph.n_observed_nodes**2))
        idx = np.arange(self.sp_graph.n_observed_nodes**2).reshape(-1,self.sp_graph.n_observed_nodes).T[np.tril_indices(self.sp_graph.n_observed_nodes,k=-1)]   
        for I in idx:
            i = I//self.sp_graph.n_observed_nodes; j = I%self.sp_graph.n_observed_nodes
            dsigdL[I,self.sp_graph.n_observed_nodes*i+j] = 2*self.Linv[i,j]**2
            dsigdL[I,self.sp_graph.n_observed_nodes*j+i] = 2*self.Linv[i,j]**2

            dsigdL[I,self.sp_graph.n_observed_nodes*i+i] = -self.Linv[i,i]**2
            dsigdL[I,self.sp_graph.n_observed_nodes*j+j] = -self.Linv[j,j]**2

            It = self.sp_graph.n_observed_nodes*j + i
            dsigdL[It,self.sp_graph.n_observed_nodes*i+j] = 2*self.Linv[j,i]**2
            dsigdL[It,self.sp_graph.n_observed_nodes*j+i] = 2*self.Linv[i,j]**2

            dsigdL[It,self.sp_graph.n_observed_nodes*i+i] = -self.Linv[i,i]**2
            dsigdL[It,self.sp_graph.n_observed_nodes*j+j] = -self.Linv[j,j]**2

        s = self.sp_graph.lre[0][0]; d = self.sp_graph.lre[0][1]
        I = self.sp_graph.n_observed_nodes*d + s 
        dsigdL[I,self.sp_graph.n_observed_nodes*s+d] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(2*self.Linv[s,d]**2)
        dsigdL[I,self.sp_graph.n_observed_nodes*d+s] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(2*self.Linv[d,s]**2)

        dsigdL[I,self.sp_graph.n_observed_nodes*s+s] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(-self.Linv[s,s]**2)  
        dsigdL[I,self.sp_graph.n_observed_nodes*d+d] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(-self.Linv[d,d]**2) 

        # do it for transpose too
        I = self.sp_graph.n_observed_nodes*s + d 
        dsigdL[I,self.sp_graph.n_observed_nodes*s+d] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(2*self.Linv[s,d]**2)
        dsigdL[I,self.sp_graph.n_observed_nodes*d+s] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(2*self.Linv[d,s]**2)

        dsigdL[I,self.sp_graph.n_observed_nodes*s+s] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(-self.Linv[s,s]**2)  
        dsigdL[I,self.sp_graph.n_observed_nodes*d+d] = (0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*(-self.Linv[d,d]**2) 

        for i in set(range(self.sp_graph.n_observed_nodes))-{s,d}:
            I = self.sp_graph.n_observed_nodes*d + i
            dsigdL[I,self.sp_graph.n_observed_nodes*i+d] = (1-self.sp_graph.c)*(2*self.Linv[i,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*d+i] = (1-self.sp_graph.c)*(2*self.Linv[i,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*s+s] = 0.5*(self.sp_graph.c**2+self.sp_graph.c)*(-self.Linv[s,s]**2)

            dsigdL[I,self.sp_graph.n_observed_nodes*d+d] = 0.5*(self.sp_graph.c**2-3*self.sp_graph.c+2)*(-self.Linv[d,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*s+d] = (self.sp_graph.c**2-self.sp_graph.c)*(self.Linv[s,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*d+s] = (self.sp_graph.c**2-self.sp_graph.c)*(self.Linv[s,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*i+i] = -self.Linv[i,i]**2

            I = self.sp_graph.n_observed_nodes*i + d
            dsigdL[I,self.sp_graph.n_observed_nodes*i+d] = (1-self.sp_graph.c)*(2*self.Linv[i,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*d+i] = (1-self.sp_graph.c)*(2*self.Linv[i,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*s+s] = 0.5*(self.sp_graph.c**2+self.sp_graph.c)*(-self.Linv[s,s]**2)

            dsigdL[I,self.sp_graph.n_observed_nodes*d+d] = 0.5*(self.sp_graph.c**2-3*self.sp_graph.c+2)*(-self.Linv[d,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*s+d] = (self.sp_graph.c**2-self.sp_graph.c)*(self.Linv[s,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*d+s] = (self.sp_graph.c**2-self.sp_graph.c)*(self.Linv[s,d]**2)
            dsigdL[I,self.sp_graph.n_observed_nodes*i+i] = -self.Linv[i,i]**2
            
        ## (d2 x e)
        dLdw = np.zeros((self.sp_graph.n_observed_nodes**2,len(self.sp_graph.edges)))
        for I in range(len(self.sp_graph.edges)):
            Ii = list(self.sp_graph.edges)[I][0]; Ij = list(self.sp_graph.edges)[I][1]

            dLdw[self.sp_graph.n_observed_nodes*Ii + Ij, I] = -1
            dLdw[self.sp_graph.n_observed_nodes*Ij + Ii, I] = -1

        for i in range(self.sp_graph.n_observed_nodes):
            for j in np.where(self.sp_graph.Delta_q.toarray()[:,i])[0]:
                dLdw[self.sp_graph.n_observed_nodes*i + i, j] = 1

        self.grad_obj = np.ravel(delldsig @ dsigdL @ dLdw)

        # grads for d diag(Jq^-1) / dq
        if self.optimize_q == 'n-dim':
            numd = len(self.sp_graph)
            s = self.sp_graph.lre[0][0]; d = self.sp_graph.lre[0][1]
            dsigdq = np.zeros((self.sp_graph.n_observed_nodes**2,self.sp_graph.n_observed_nodes))
            idx = np.arange(self.sp_graph.n_observed_nodes**2).reshape(-1,self.sp_graph.n_observed_nodes).T[np.tril_indices(self.sp_graph.n_observed_nodes,k=-1)]
            for I in idx:
                i = I//self.sp_graph.n_observed_nodes; j = I%self.sp_graph.n_observed_nodes
                dsigdq[I,i] = 1
                dsigdq[I,j] = 1

                It = self.sp_graph.n_observed_nodes*j + i
                dsigdq[It,i] = 1
                dsigdq[It,j] = 1

            I = self.sp_graph.n_observed_nodes*d + s 
            dsigdq[I,s] = (1+self.sp_graph.c); dsigdq[I,d] = (1-self.sp_graph.c)
            It = self.sp_graph.n_observed_nodes*s + d
            dsigdq[It,s] = (1+self.sp_graph.c); dsigdq[It,d] = (1-self.sp_graph.c)

            for i in set(range(self.sp_graph.n_observed_nodes))-{s,d}:
                I = self.sp_graph.n_observed_nodes*d + i
                dsigdq[I,d] = (1-self.sp_graph.c)
                dsigdq[I,s] = self.sp_graph.c

                It = self.sp_graph.n_observed_nodes*i + d
                dsigdq[I,d] = (1-self.sp_graph.c)
                dsigdq[I,s] = self.sp_graph.c

            self.grad_obj_q = np.zeros(self.sp_graph.n_observed_nodes)
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = np.multiply(delldsig @ dsigdq, 1./self.sp_graph.n_samples_per_obs_node_permuted)
            # self.grad_obj_q = np.zeros(len(self.sp_graph))
            # self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)   
            
        if self.optimize_q == '1-dim':
            self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

        dsigdc = np.zeros((self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes))
        dsigdc[lrn[0][0],lrn[0][1]] = (self.sp_graph.c-1.5)*Rmat[lrn[0][0],lrn[0][1]] + 1/self.sp_graph.q[lrn[0][0]] - 1/self.sp_graph.q[lrn[0][1]]
        dsigdc[lrn[0][1],lrn[0][0]] = dsigdc[lrn[0][0],lrn[0][1]]

        for i in set(range(Rmat.shape[0]))-set([lrn[0][0],lrn[0][1]]):
            dsigdc[i,lrn[0][1]] = -Rmat[i,lrn[0][1]] + Rmat[i,lrn[0][0]] + (self.sp_graph.c-0.5)*Rmat[lrn[0][0],lrn[0][1]] - 1/self.sp_graph.q[lrn[0][1]] + 1/self.sp_graph.q[lrn[0][0]]
            dsigdc[lrn[0][1],i] = dsigdc[i,lrn[0][1]]

        self.grad_obj_c = np.ravel(delldsig @ dsigdc.reshape(-1,1))[0]

    def _comp_grad_obj_noc(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c) with respect to the latent variables dLoss / dL
        """

        # compute inverses
        self._comp_inv_lap()
        lrn = self.sp_graph.lre

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = (np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(1/self.sp_graph.q,(1,-1))).T
        resmat = -0.5*(Rmat + (Q1mat + Q1mat.T) - 2*np.diag(1/self.sp_graph.q))
        
        resmat[lrn[0][0],lrn[0][1]] = -0.5*((0.5*self.sp_graph.c**2-1.5*self.sp_graph.c+1)*Rmat[lrn[0][0],lrn[0][1]] + (1+self.sp_graph.c)/self.sp_graph.q[lrn[0][0]] + (1-self.sp_graph.c)/self.sp_graph.q[lrn[0][1]])
        resmat[lrn[0][1],lrn[0][0]] = resmat[lrn[0][0],lrn[0][1]]

        ## id
        for i in set(range(self.sp_graph.n_observed_nodes))-set([lrn[0][0],lrn[0][1]]):
            resmat[i,lrn[0][1]] = -0.5*((1-self.sp_graph.c)*(Rmat[i,lrn[0][1]]) + self.sp_graph.c*Rmat[i,lrn[0][0]] + 0.5*(self.sp_graph.c**2-self.sp_graph.c)*Rmat[lrn[0][0],lrn[0][1]] + 1/self.sp_graph.q[i] + (1-self.sp_graph.c)/self.sp_graph.q[lrn[0][1]] + self.sp_graph.c/self.sp_graph.q[lrn[0][0]])
            resmat[lrn[0][1],i] = resmat[i,lrn[0][1]]

        M = self.C.T @ (np.linalg.inv(self.C@resmat@self.C.T) @ (self.C@self.sp_graph.S@self.C.T) @ np.linalg.inv(self.C@resmat@self.C.T) - np.linalg.inv(self.C@resmat@self.C.T)) @self.C

        self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        self.grad_obj = gradD - gradW
        
        # grads for d diag(Jq^-1) / dq
        if self.optimize_q == 'n-dim':
            self.grad_obj_q = np.zeros(len(self.sp_graph))
            self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)   
            
        if self.optimize_q == '1-dim':
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

        ## Feb 27, 2023 (set gradient of penalty to negative of gradient if LRE was actually short range edge)
        self.grad_pen[self.sp_graph.lre_idx] = -self.grad_pen[self.sp_graph.lre_idx]
        
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

        lik = self.neg_log_lik()
        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w[~self.sp_graph.lre_idx])
        term_1 = alpha * self.sp_graph.w[~self.sp_graph.lre_idx] + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta[:,~self.sp_graph.lre_idx] @ term_1) ** 2
                
        if self.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen = 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2   
        else: 
            term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w[~self.sp_graph.lre_idx])
            term_1 = alpha * self.sp_graph.w[~self.sp_graph.lre_idx] + np.log(term_0)
            pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta[:,~self.sp_graph.lre_idx] @ term_1) ** 2   

            ## calculating penalty associated with LRE term 
            ## (just calculating penalty associated with short range edges instead of subtracting off term from penalty)
            # if len(self.sp_graph.lre)>0:
            #     term_lre = alpha * self.sp_graph.w[self.sp_graph.lre_idx] + np.log(1.0 - np.exp(-alpha * self.sp_graph.w[self.sp_graph.lre_idx]))
            #     pen_lre = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta[:,self.sp_graph.lre_idx] @ term_lre) ** 2

            #     # removing the penalty associated with LRE: no constraint on LRE weight
            #     pen = pen - pen_lre

        # loss
        loss = lik + pen
        return loss 

def full_nll_w_c(c, obj):
    """Evaluate the full negative log-likelihood for the given weights & admix. prop. c
    """

    lrn = obj.sp_graph.lre

    Rmat = -2*obj.Linv[:obj.sp_graph.n_observed_nodes,:obj.sp_graph.n_observed_nodes] + np.reshape(np.diag(obj.Linv),(1,-1)).T @ np.ones((obj.sp_graph.n_observed_nodes,1)).T + np.ones((obj.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(obj.Linv),(1,-1))
    Q1mat = (np.ones((obj.sp_graph.n_observed_nodes,1)) @ np.reshape(1/obj.sp_graph.q,(1,-1))).T
    resmat = Rmat + (Q1mat + Q1mat.T) - 2*np.diag(1/obj.sp_graph.q)

    resmat[lrn[0][0],lrn[0][1]] = (0.5*c**2-1.5*c+1)*Rmat[lrn[0][0],lrn[0][1]] + (1+c)/obj.sp_graph.q[lrn[0][0]] + (1-c)/obj.sp_graph.q[lrn[0][1]]
    resmat[lrn[0][1],lrn[0][0]] = resmat[lrn[0][0],lrn[0][1]]

    for i in set(range(obj.sp_graph.n_observed_nodes))-set([lrn[0][0],lrn[0][1]]):
        resmat[i,lrn[0][1]] = (1-c)*(Rmat[i,lrn[0][1]]) + c*Rmat[i,lrn[0][0]] + 0.5*(c**2-c)*Rmat[lrn[0][0],lrn[0][1]] + 1/obj.sp_graph.q[i] + (1-c)/obj.sp_graph.q[lrn[0][1]] + c/obj.sp_graph.q[lrn[0][0]]
        resmat[lrn[0][1],i] = resmat[i,lrn[0][1]]

    nll = -wishart.logpdf(2*obj.C @ obj.sp_graph.S @ obj.C.T, obj.sp_graph.n_snps, -0.5/obj.sp_graph.n_snps*obj.C @ resmat @ obj.C.T)

    return nll

def loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w) which returns the loss and gradient"""                
    n_edges = obj.sp_graph.size()
    theta = np.exp(z)
    theta0 = theta[:n_edges]
    obj.sp_graph.comp_graph_laplacian(theta0)
    if obj.optimize_q is not None:
        theta1 = theta[n_edges:]
        obj.sp_graph.comp_precision(s2=theta1)
    obj.inv()
    obj.grad()     

    # loss / grad
    loss = obj.loss()
    if obj.optimize_q is None:
        grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    elif obj.optimize_q == 'n-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2 + obj.grad_pen_q * obj.sp_graph.s2
    elif obj.optimize_q == '1-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2          
    return (loss, grad)

def joint_loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w,q,c) which returns the loss and gradient for ALL parameters"""                
    n_edges = obj.sp_graph.size()
    theta = np.r_[np.exp(z[:-1]), z[-1]]
    theta0 = theta[:n_edges]
    obj.sp_graph.comp_graph_laplacian(theta0)
    if obj.optimize_q is not None:
        theta1 = theta[n_edges:-1]
        obj.sp_graph.comp_precision(s2=theta1)
    obj.sp_graph.c = theta[-1]
    obj.inv()
    obj.grad()     

    # loss / grad
    loss = obj.loss()
    if obj.optimize_q is None:
        grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    elif obj.optimize_q == 'n-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:-1] = obj.grad_obj_q * obj.sp_graph.s2 + obj.grad_pen_q * obj.sp_graph.s2
        grad[-1] = obj.grad_obj_c # no penalty on admix. prop.
    elif obj.optimize_q == '1-dim':
        grad = np.zeros_like(theta)
        grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
        grad[n_edges:-1] = obj.grad_obj_q * obj.sp_graph.s2    
        grad[-1] = obj.grad_obj_c      
    return (loss, grad)

def stepwise_minimize(obj, factr, m, maxls, maxiter, bounds, verbose=False):
    """
    Minimize the negative log-likelihood iteratively with an admix. prop. c value & refit the new weights based on that until tolerance is reached. We assume the best-fit weights are passed in to this function as x0. 
    """
    optimc = True
    for bigiter in range(10):
        # first fit admix. prop. c
        if optimc: # stop it from overly optimizing over c
            resc = minimize_scalar(full_nll_w_c, args=obj, method='bounded', bounds=(0,1))
            if resc.status != 0:
                print('Warning: admix. prop. optimization failed')
            if np.abs(resc.x - obj.sp_graph.c) < 1e-3:
                optimc = False
            obj.sp_graph.c = resc.x

        x0 = np.r_[np.log(obj.sp_graph.w), np.log(obj.sp_graph.s2)]
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
            # bounds=bounds
        )
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge"
        neww = np.exp(res[0][:obj.sp_graph.size()])
        news2 = np.exp(res[0][obj.sp_graph.size():])

        if np.allclose(obj.sp_graph.w, neww, atol=1e-3) and np.allclose(obj.sp_graph.s2, news2, atol=1e-3):
            if verbose:
                print('admix. prop. converged in {} iterations!'.format(bigiter+1))
            break
        else: # update weights and s2 & continue
            obj.sp_graph.w = neww
            obj.sp_graph.s2 = news2

    return res

class Joint_SpatialGraph(SpatialGraph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True, long_range_edges=[(0,0)], c=0.2):
        """Inherit from the feems object SpatialGraph and overwrite some methods for 
        estimation of edge weights and residual variance jointly
        """               
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps,
                         long_range_edges=long_range_edges,
                         c = c)
        
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
        obj = Joint_Objective(self)
        res = minimize(neg_log_lik_w0_s2, [0.0, 0.0], method="Nelder-Mead", args=(obj))
        assert res.success is True, "did not converge"
        w0_hat = np.exp(res.x[0])
        s2_hat = np.exp(res.x[1])
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
        optimize_q=None,
        lamb_q=None,
        alpha_q=None,
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
        option='default'
    ):
        """Estimates the edge weights of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.
        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            optimize_q (:obj:'str'): indicator whether optimizing residual variances (None, 1-dim, n-dim)
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

        if option!='onlyc':
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
            obj = Joint_Objective(self, option=option)
            obj.optimize_q = optimize_q
            obj.lamb = lamb
            obj.alpha = alpha
            x0 = np.log(w_init)
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
                s2_init = self.s2 if obj.optimize_q=="1-dim" else self.s2*np.ones(len(self))
                x0 = np.r_[np.log(w_init), np.log(s2_init)]

            if option=='default':
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
            elif option=='joint':
                res = fmin_l_bfgs_b(
                    func=joint_loss_wrapper,
                    x0=np.append(x0, self.c),
                    args=[obj],
                    factr=factr,
                    m=m,
                    maxls=maxls,
                    maxiter=maxiter,
                    approx_grad=False,
                    bounds=[(lb, ub)] * len(x0) + [(0,1)]
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

            obj = Joint_Objective(self, option=option)
            obj.optimize_q = optimize_q
            obj.lamb = lamb
            obj.alpha = alpha
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            obj.inv(); obj.grad(reg=False)
            res = stepwise_minimize(
                obj=obj,
                factr=factr,
                m=m,
                maxls=maxls,
                maxiter=maxiter,
                bounds=[(lb, ub)] * int(self.n_observed_nodes+len(self.w))
            )

        # if maxiter >= 100:
        #     assert res[2]["warnflag"] == 0, "did not converge"
        if obj.optimize_q is not None:
            self.w = np.exp(res[0][:self.size()])
            self.s2 = np.exp(res[0][self.size():-1])
            if option=='joint':
                self.c = res[0][-1]
        else:    
            self.w = np.exp(res[0])
            if option=='default':
                self.c = res[0][-1]

        # print update
        if option=='default':
            self.train_loss, _ = loss_wrapper(res[0], obj)
        elif option=='joint':
            self.train_loss, _ = joint_loss_wrapper(res[0], obj)
        if verbose:
            sys.stdout.write(
                (
                    "lambda={:.3f}, "
                    "alpha={:.4f}, "
                    "converged in {} iterations, "
                    "train_loss={:.3f}\n"
                ).format(lamb, alpha, res[2]["nit"], self.train_loss)
            ) 