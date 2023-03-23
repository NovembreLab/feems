from __future__ import absolute_import, division, print_function

import numpy as np


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
        self.beta = None

        self.pen1 = 0.0
        self.pen2 = 0.0

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

    def _comp_grad_reg(self):
        """Computes gradient"""
        lamb = self.lamb
        alpha = self.alpha
        beta = self.beta

        # avoid overflow in exp
        # term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
        # term_1 = alpha * self.sp_graph.w + np.log(term_0)
        # term_2 = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term_1)
        # self.grad_pen = term_2 * (alpha / term_0)
        term = self.alpha * self.sp_graph.w + np.log(
            1 - np.exp(-self.alpha * self.sp_graph.w)
        )  # avoid overflow in exp
        self.grad_pen = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term)
        self.grad_pen = self.grad_pen * (self.alpha / (1 - np.exp(-self.alpha * self.sp_graph.w)))  
        # only fill the long range edge indices with this derivative
        ## Feb 26, 2023 - set the derivative to 0? 
        # self.grad_pen[self.sp_graph.lre_idx] = 0
        # beta * np.ones(np.sum(self.sp_graph.lre_idx))  
        # 2.0 * self.graph.w[lre_idx] if Frobenius/L-2 norm
        # np.ones(np.sum(lre_idx)) if L-1 norm

    def inv(self):
        """Computes relevant inverses for gradient computations"""
        # compute inverses
        self._solve_lap_sys()
        self._comp_mat_block_inv()
        self._comp_inv_cov()

    def grad(self, reg=True):
        """Computes relevent gradients the objective"""
        # compute derivatives
        self._comp_grad_obj()
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
        self.det = np.linalg.det(self.inv_cov) * o / self.denom

        # negative log-likelihood
        nll = self.sp_graph.n_snps * (self.tr - np.log(self.det))

        return nll

    def loss(self):
        """Evaluate the loss function given the current params"""
        lamb = self.lamb
        beta = self.beta

        lik = self.neg_log_lik()

        # index edges that are NOT in lre
        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = self.alpha * self.sp_graph.w + np.log(term_0)
        pen1 = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2

        self.pen1 = pen1 

        # lasso penalty for lre, index edges in lre
        ## Feb 26, 2023 - no penalty for long range edges
        # pen2 = beta * np.sum(self.sp_graph.w[self.sp_graph.lre_idx])
        ## relative scaling: by the inverse of the graph Laplacian?
        # larger the absolute value (lower the inverse), the lower the scaling coefficient - prompts a lower beta?
        #pen2 = beta * np.sum(self.sp_graph.w[self.sp_graph.lre_idx]/np.abs([self.sp_graph.L[i] for i in self.sp_graph.lre]))

        # self.pen2 = pen2

        # loss
        loss = lik + pen1 
        return loss


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
    """Wrapper function to optimize z=log(w) which returns the loss and gradient"""
    theta = np.exp(z)
    obj.sp_graph.comp_graph_laplacian(theta)
    obj.inv()
    obj.grad()

    #  s / grad
    loss = obj.loss()
    grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
    return (loss, grad)


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
    frequencies_ns = sp_graph.frequencies * np.sqrt(sp_graph.mu*(1-sp_graph.mu))
    mu0 = frequencies_ns.mean(axis=0) / 2 # compute mean of allele frequencies in the original scale
    mu = 2*mu0 / np.sqrt(sp_graph.mu*(1-sp_graph.mu))
    frequencies_centered = sp_graph.frequencies - mu
    emp_cov = frequencies_centered @ frequencies_centered.T / n_snps
    
    return fit_cov, inv_cov, emp_cov
    