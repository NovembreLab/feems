import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b, minimize

from .spatial_graph import SpatialGraph
from .objective import Objective, neg_log_lik_w0_s2

class Joint_Objective(Objective): 
    def __init__(self, sp_graph):
        """Inherit from the feems object Objective and overwrite some methods for evaluations 
        and gradient of feems objective when residual variance is estimated jointly with edge weights
        Args:
            sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
        """
        super().__init__(sp_graph=sp_graph)   
        
        # indicator whether optimizing residual variance jointly with edge weights
        # None  : residual variance is holding fixed
        # 1-dim : single residual variance is estimated across all nodes
        # n-dim : node-specific residual variances are estimated 
        self.optimize_q = None 
        
        # reg params for residual variance
        self.lamb_q= None
        self.alpha_q = None

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

        lik = self.neg_log_lik()
        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
                
        if self.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-self.alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen = 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2                

        # loss
        loss = lik + pen
        return loss        


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


class Joint_SpatialGraph(SpatialGraph):
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True, long_range_edges=[(0,0)]):
        """Inherit from the feems object SpatialGraph and overwrite some methods for 
        estimation of edge weights and residual variance jointly
        """               
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps,
                         long_range_edges=long_range_edges) 
        
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
                    "train_loss={:.7f}\n"
                ).format(res.nfev, self.train_loss)
            )    
    
    def fit(
        self,
        lamb,
        w_init=None,
        s2_init=None,
        alpha=None,
        optimize_q=None,
        lamb_q=1.,
        alpha_q=1.,
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
        constrained=False,
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
        obj = Joint_Objective(self)
        obj.optimize_q = optimize_q
        obj.lamb = lamb
        obj.alpha = alpha
        x0 = np.log(w_init)
        if obj.optimize_q is not None:
            obj.lamb_q = lamb_q
            obj.alpha_q = alpha_q
            s2_init = self.s2 if obj.optimize_q=="1-dim" else self.s2*np.ones(len(self))
            x0 = np.r_[np.log(w_init), np.log(s2_init)]
            if constrained:
                # print("performing constrained optimization...\n")
                bounds = np.r_[[(x-0.01, x+0.01) for x in x0[:len(w_init)]],[(lb, ub) for _ in range(len(s2_init))]]
            else: 
                bounds = [(lb, ub) for _ in range(x0.shape[0])]
                # print("performing unconstrained optimization...\n")
        res = fmin_l_bfgs_b(
            func=loss_wrapper,
            x0=x0,
            args=[obj],
            factr=factr,
            m=m,
            maxls=maxls,
            maxiter=maxiter,
            approx_grad=False,
            bounds=bounds,
        )
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge"
        if obj.optimize_q is not None:
            self.w = np.exp(res[0][:self.size()])
            self.s2 = np.exp(res[0][self.size():])
        else:    
            self.w = np.exp(res[0])

        # print update
        self.train_loss, _ = loss_wrapper(res[0], obj)
        if verbose:
            sys.stdout.write(
                (
                    "lambda={:.7f}, "
                    "alpha={:.7f}, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(lamb, alpha, res[2]["nit"], self.train_loss)
            ) 