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
        # None  : residual variance is held fixed
        # 1-dim : single residual variance is estimated across all nodes
        # n-dim : node-specific residual variances are estimated 
        self.optimize_q = 'n-dim'
        
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

    def _comp_grad_obj_c(self):
        """Computes the gradient of the objective function (now defined with admix. prop. c) with respect to the latent variables dLoss / dL
        """

        # compute inverses
        self._comp_inv_lap()

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        resmat = -0.5*(Rmat + (Q1mat + Q1mat.T) - 2*self.sp_graph.q_inv_diag)

        for ie, _ in enumerate(self.sp_graph.lre):
            if(self.sp_graph.lre[ie][0]<self.sp_graph.n_observed_nodes and self.sp_graph.lre[ie][1]<self.sp_graph.n_observed_nodes):
                resmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] = -0.5*((0.5*self.sp_graph.c[ie]**2-1.5*self.sp_graph.c[ie]+1)*Rmat[self.sp_graph.lre[0][0],self.sp_graph.lre[ie][1]] + (1+self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][0]] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][1]])
                resmat[self.sp_graph.lre[ie][1],self.sp_graph.lre[ie][0]] = resmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]]

                for i in set(range(self.sp_graph.n_observed_nodes))-set([self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]]):
                    resmat[i,self.sp_graph.lre[ie][1]] = -0.5*((1-self.sp_graph.c[ie])*Rmat[i,self.sp_graph.lre[ie][1]] + self.sp_graph.c[ie]*Rmat[i,self.sp_graph.lre[ie][0]] + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*Rmat[self.sp_graph.lre[ie][0],self.sp_graph.lre[ie][1]] + 1/self.sp_graph.q[i] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[ie][1]] + self.sp_graph.c[ie]/self.sp_graph.q[self.sp_graph.lre[ie][0]])
                    resmat[self.sp_graph.lre[ie][1],i] = resmat[i,self.sp_graph.lre[ie][1]]
            else:
                neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[self.sp_graph.lre[0][0]]))
                neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

                R1d = -2*self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][1]] + self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][0]] + self.Lpinv[self.sp_graph.lre[0][1],self.sp_graph.lre[0][1]]

                for s in neighs:
                    # convert back to appropriate indexing excluding the unsampled demes
                    s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==s][0]
                    resmat[s,self.sp_graph.lre[0][1]] = Rmat[s,self.sp_graph.lre[0][1]] + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*R1d + (1-self.sp_graph.c[ie])/self.sp_graph.q[s] + (1+self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[0][1]]
                    resmat[self.sp_graph.lre[0][1],s] = resmat[s,self.sp_graph.lre[0][1]]

                proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=self.sp_graph.lre[0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([self.sp_graph.lre[0][0]])])
                ## id
                for i in set(range(self.sp_graph.n_observed_nodes))-set([self.sp_graph.lre[0][0],self.sp_graph.lre[0][1]]+neighs):
                    Ri1 = -2*self.Lpinv[i,self.sp_graph.lre[0][0]] + self.Lpinv[i,i] + self.Lpinv[self.sp_graph.lre[0][0],self.sp_graph.lre[0][0]]
                    resmat[i,self.sp_graph.lre[0][1]] = (1-self.sp_graph.c[ie])*(Rmat[i,self.sp_graph.lre[0][1]]) + self.sp_graph.c[ie]*Ri1 + 0.5*(self.sp_graph.c[ie]**2-self.sp_graph.c[ie])*R1d + 1/self.sp_graph.q[i] + (1-self.sp_graph.c[ie])/self.sp_graph.q[self.sp_graph.lre[0][1]] + self.sp_graph.c[ie]/self.sp_graph.q[proxs]
                    resmat[self.sp_graph.lre[0][1],i] = resmat[i,self.sp_graph.lre[0][1]]

        CRCt = np.linalg.inv(self.C @ resmat @ self.C.T)
        M = self.C.T @ (CRCt @ (self.C @ self.sp_graph.S @ self.C.T) @ CRCt - CRCt) @ self.C

        self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        self.grad_obj = np.ravel(gradD - gradW)
        
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

        if self.option == 'default':
            lik = self.neg_log_lik()
        else:
            lik = self.neg_log_lik_c(self.sp_graph.c)
        term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
        term_1 = alpha * self.sp_graph.w + np.log(term_0)
        pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
                
        if self.optimize_q == 'n-dim':
            lamb_q = self.lamb_q
            alpha_q = self.alpha_q
                
            term_0 = 1.0 - np.exp(-alpha_q * self.sp_graph.s2)
            term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
            pen = 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2   

        # loss
        loss = lik + pen
        return loss 

    def neg_log_lik_c(self, c, opts):
        """Evaluate the full negative log-likelihood for the given weights & admix. prop. c
        (changing function to only take in single long-range edge with a flag for 'sampled' vs 'unsampled' source)
        """

        Rmat = -2*self.Linv[:self.sp_graph.n_observed_nodes,:self.sp_graph.n_observed_nodes] + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)).T + np.broadcast_to(np.diag(self.Linv),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.reshape(np.diag(self.Linv),(1,-1)).T @ np.ones((self.sp_graph.n_observed_nodes,1)).T + np.ones((self.sp_graph.n_observed_nodes,1)) @ np.reshape(np.diag(self.Linv),(1,-1))
        Q1mat = np.broadcast_to(self.sp_graph.q_inv_diag.diagonal(),(self.sp_graph.n_observed_nodes,self.sp_graph.n_observed_nodes)) #np.ones((self.sp_graph.n_observed_nodes,1)) @ self.sp_graph.q_inv_diag.diagonal().reshape(1,-1)
        resmat = Rmat + (Q1mat + Q1mat.T) - 2*self.sp_graph.q_inv_diag

        if opts['mode']=='sampled':
            resmat[opts['lre'][0][0],opts['lre'][0][1]] = (0.5*c**2-1.5*c+1)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + (1+c)/self.sp_graph.q[opts['lre'][0][0]] + (1-c)/self.sp_graph.q[opts['lre'][0][1]]
            resmat[opts['lre'][0][1],opts['lre'][0][0]] = resmat[opts['lre'][0][0],opts['lre'][0][1]]

            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]):
                resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Rmat[i,opts['lre'][0][0]] + 0.5*(c**2-c)*Rmat[opts['lre'][0][0],opts['lre'][0][1]] + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[opts['lre'][0][0]]
                resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]
        else:
            neighs = list(self.sp_graph.neighbors(nx.get_node_attributes(self.sp_graph,'permuted_idx')[opts['lre'][0][0]]))
            neighs = [s for s in neighs if nx.get_node_attributes(self.sp_graph,'n_samples')[s]>0]

            R1d = -2*self.Lpinv[opts['lre'][0][0],opts['lre'][0][1]] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]] + self.Lpinv[opts['lre'][0][1],opts['lre'][0][1]]

            for s in neighs:
                # convert back to appropriate indexing excluding the unsampled demes
                s = [k for k, v in nx.get_node_attributes(self.sp_graph,'permuted_idx').items() if v==s][0]
                resmat[s,opts['lre'][0][1]] = Rmat[s,opts['lre'][0][1]] + 0.5*(c**2-c)*R1d + (1-c)/self.sp_graph.q[s] + (1+c)/self.sp_graph.q[opts['lre'][0][1]]
                resmat[opts['lre'][0][1],s] = resmat[s,opts['lre'][0][1]]

            proxs = np.argmin([nx.shortest_path_length(self.sp_graph,source=opts['lre'][0][0],target=d) for d in set([k for k, v in nx.get_node_attributes(self.sp_graph,'n_samples').items() if v>0])-set([opts['lre'][0][0]])])
            ## id
            for i in set(range(self.sp_graph.n_observed_nodes))-set([opts['lre'][0][0],opts['lre'][0][1]]+neighs):
                Ri1 = -2*self.Lpinv[i,opts['lre'][0][0]] + self.Lpinv[i,i] + self.Lpinv[opts['lre'][0][0],opts['lre'][0][0]]
                resmat[i,opts['lre'][0][1]] = (1-c)*(Rmat[i,opts['lre'][0][1]]) + c*Ri1 + 0.5*(c**2-c)*R1d + 1/self.sp_graph.q[i] + (1-c)/self.sp_graph.q[opts['lre'][0][1]] + c/self.sp_graph.q[proxs]
                resmat[opts['lre'][0][1],i] = resmat[i,opts['lre'][0][1]]

        nll = -wishart.logpdf(2*self.C @ self.sp_graph.S @ self.C.T, self.sp_graph.n_snps, -0.5/self.sp_graph.n_snps*self.C @ resmat @ self.C.T)

        return nll

def cov_to_dist(S):
    """Convert a covariance matrix to a distance matrix
    """
    s2 = np.diag(S).reshape(-1, 1)
    ones = np.ones((s2.shape[0], 1))
    D = s2 @ ones.T + ones @ s2.T - 2 * S
    return(D)

def extract_outliers(obj, pthresh=0.01, top=3):
    """Function to extract outlier deme pairs based on a p-value threshold specified by the user (default: 0.01)"""
    assert pthresh>0, "pthresh should be a positive number"

    # computing pairwise covariance & distances between demes
    fit_cov, _, emp_cov = comp_mats(obj)
    fit_dist = cov_to_dist(fit_cov)[np.tril_indices(obj.sp_graph.n_observed_nodes, k=-1)]
    emp_dist = cov_to_dist(emp_cov)[np.tril_indices(obj.sp_graph.n_observed_nodes, k=-1)]

    perm_idx = query_node_attributes(obj.sp_graph, "permuted_idx")

    print('Using a significance threhsold of {:g}:\n'.format(pthresh))
    ls = []; x, y = [], []
    # computing p-values (or z-values) for each pairwise comparison after mean centering
    pvals = norm.cdf(np.log(emp_dist)-np.log(fit_dist)-np.mean(np.log(emp_dist)-np.log(fit_dist)), 0, np.std(np.log(emp_dist)-np.log(fit_dist)))
    for k in np.where(pvals < pthresh)[0]:
        # code to convert single index to matrix indices
        x.append(np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1); y.append(int(k - 0.5*x[-1]*(x[-1]-1)))
        ls.append([perm_idx[x[-1]], perm_idx[y[-1]], tuple(obj.sp_graph.nodes[x[-1]]['pos'][::-1]), tuple(obj.sp_graph.nodes[y[-1]]['pos'][::-1]), pvals[k]])

    rm = []
    for k in range(len(ls)):
        # checking the log-lik of fits with deme1 - deme2 to find the source & dest. deme
        resc = minimize(obj.neg_log_lik_c, x0=np.random.random(), args={'lre':[(x[k],y[k])],'mode':'sampled'}, method='Powell', bounds=[(0,1)])
        rescopp = minimize(obj.neg_log_lik_c, x0=np.random.random(), args={'lre':[(y[k],x[k])],'mode':'sampled'}, method='Powell', bounds=[(0,1)])
        if resc.x<1e-6 and rescopp.x<1e-6:
            rm.append(k)
        else:
            if rescopp.fun < resc.fun:
                ls[k][0] = perm_idx[y[k]]
                ls[k][1] = perm_idx[x[k]]

    # removing demes that have estimated admix. prop. ≈ 0 
    for i in sorted(rm, reverse=True):
        del ls[i]
    df = pd.DataFrame(ls, columns = ['source', 'dest.', 'source (lat., long.)', 'dest. (lat., long.)', 'pval'])

    if len(df)==0:
        print('No outliers found.')
        print('Consider raising the significance threshold slightly.')
    else:
        print('{:d} outlier deme pairs found'.format(len(df)))
        print(df.sort_values(by='pval').to_string(index=False))

        print('\nPutative destination demes (and # of times the deme appears as an outlier) experiencing admixture:')
        print(df['dest.'].value_counts().head(top))

    return df
              
def loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w,q) which returns the loss and gradient"""                
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

def coordinate_descent(obj, factr, m, maxls, maxiter, verbose):
    """
    Minimize the negative log-likelihood iteratively with an admix. prop. c value & refit the new weights based on that until tolerance is reached. We assume the best-fit weights are passed in to this function as x0. 
    """
    optimc = True
    for bigiter in range(10):
        # first fit admix. prop. c
        if optimc: # stop it from overly optimizing over c
            resc = minimize(obj.neg_log_lik_c, x0=obj.sp_graph.c, args={'lre':obj.sp_graph.lre,'mode':'sampled'}, method='Powell', bounds=[(0,1)])
            if resc.status != 0:
                print('Warning: admix. prop. optimization failed')
            if np.min(np.abs(resc.x - obj.sp_graph.c)) > 1e-4:
                optimc = False
                if bigiter > 0:
                    break
            obj.sp_graph.c = deepcopy(resc.x)

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
        )
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge (increase maxiter)"
        neww = np.exp(res[0][:obj.sp_graph.size()])
        news2 = np.exp(res[0][obj.sp_graph.size():])
        # print('length of s2:',len(news2))

        if np.allclose(obj.sp_graph.w, neww, atol=1e-3) and np.allclose(obj.sp_graph.s2, news2, atol=1e-3):
            if verbose:
                print('admix. prop. estimation converged in {} iterations!'.format(bigiter+1))
            break
        else: # update weights and s2 & continue
            obj.sp_graph.w = deepcopy(neww)
            obj.sp_graph.s2 = deepcopy(news2)
            obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.todense()); obj.grad(reg=False)

    return res

def calc_contour(obj, destpid, sourcepid, coverage='all', radius=3, sprange=[[-180,180],[-90,-90]], customid=range(100)):
    """
    Function to calculate admix. prop. values along with log-lik. values in a contour around the sampled source deme to capture uncertainty in the location of the source. 
    The flag coverage is used to signifiy how large the contour should be:
        'all'    - include all demes (sampled & unsampled) from the entire graph
        'radius' - include all demes within a certain radius of a sampled source deme (sourceid output from extract_outliers)
        'range'  - include all demes within a certain long. & lat. rectangle (e.g., [[-120,-70],[25,50]] for contiguous USA)
        'custom' - specific array of deme ids
    """

    # convert destid to obs_perm_ids scheme since the Linv matrix is based on this indexing
    perm_id = query_node_attributes(obj.sp_graph, 'permuted_idx')
    try:
        destid = np.where(perm_id[:obj.sp_graph.n_observed_nodes]==destpid)[0][0]
    except:
        print('invalid ID for destination deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')

    # creating a list of (source, dest.) pairings based on user-picked criteria
    if coverage == 'all':
        # including every possible node in graph as a putative source
        randedge = [(x,destid) for x in list(set(range(obj.sp_graph.number_of_nodes()))-set([destid]))]
    elif coverage == 'radius':
        assert type(radius) == int and radius > 0, "radius must be an integer >=1"
        try: 
            sourceid = np.where(perm_id[:obj.sp_graph.n_observed_nodes]==sourcepid)[0][0]
        except: 
            print('invalid ID for source deme, please specify valid sampled ID from graph or from output of extract_outliers function\n')
        neighs = [] 
        neighs = list(obj.sp_graph.neighbors(sourceid))
        # including all nodes within a certain radius
        for _ in range(radius):
            tempn = [list(obj.sp_graph.neighbors(n1)) for n1 in neighs]
            # dropping repeated nodes 
            neighs = np.unique(list(it.chain(*tempn)))

        randedge = [(x,destid) for x in neighs]
    elif coverage == 'range':
        randedge = []
        for n in range(obj.sp_graph.number_of_nodes()):
            # checking for lat. & long. of all possible nodes in graph
            if obj.sp_graph.nodes[n]['pos'][0] > sprange[0][0] and obj.sp_graph.nodes[n]['pos'][0] < sprange[0][1]:
                if obj.sp_graph.nodes[n]['pos'][1] > sprange[1][0] and obj.sp_graph.nodes[n]['pos'][1] < sprange[1][1]:
                    randedge.append((n,destid))
    elif coverage == 'custom':
        randedge = [(x,destid) for x in list(set(customid)-set([destid]))]

    # only include central demes (==6 neighbors), since demes on edge of range exhibit some boundary effects during estimation
    randedge = list(it.compress(randedge,np.array([sum(1 for _ in obj.sp_graph.neighbors(nx.get_node_attributes(obj.sp_graph,'permuted_idx')[i])) for i in list(set(range(obj.sp_graph.number_of_nodes()))-set([destid]))])==6))

    cest2 = np.zeros(len(randedge)); llc2 = np.zeros(len(randedge))
    for ie, e in enumerate(randedge):
        # convert all sources to valid permuted ids (so observed demes should be b/w index 0 & o-1)
        e2 = (np.where(perm_id==e[0])[0][0], destid)
        if e2[0]<obj.sp_graph.n_observed_nodes:
            res = minimize(obj.neg_log_lik_c, x0=obj.sp_graph.c, bounds=[(0,1)], method='Powell', args={'lre':[e2],'mode':'sampled'})
            cest2[ie] = res.x; llc2[ie] = res.fun
        else:
            res = minimize(obj.neg_log_lik_c, x0=obj.sp_graph.c, bounds=[(0,1)], method='Powell', args={'lre':[e2],'mode':'unsampled'})
            cest2[ie] = res.x; llc2[ie] = res.fun

    return randedge, cest2, llc2

class Joint_SpatialGraph(SpatialGraph):
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
        lamb_q=None,
        alpha_q=None,
        optimize_q='n-dim',
        factr=1e7,
        maxls=50,
        m=10,
        lb=-1e-10,
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
            optimize_q (:obj:'str'): indicator whether optimizing residual variances (None, '1-dim', 'n-dim')
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

        # creating a container to store these edges 
        self.lre = long_range_edges
        # mask for indices of edges in lre
        # self.lre_idx = np.array([val in self.lre for val in list(self.edges)])

        self.c = np.random.random(len(self.lre))

        self.option = option

        if self.option!='onlyc':
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
            obj.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            x0 = np.log(w_init)
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
                s2_init = self.s2 if obj.optimize_q=="1-dim" else self.s2*np.ones(len(self))
                x0 = np.r_[np.log(w_init), np.log(s2_init)]

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
            assert optimize_q =='n-dim', "can only do admixture estimation with option 'n-dim' for optimize_q"

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
            obj.optimize_q = optimize_q; obj.lamb = lamb; obj.alpha = alpha
            if obj.optimize_q is not None:
                obj.lamb_q = lamb_q
                obj.alpha_q = alpha_q
            obj.inv(); obj.Lpinv = np.linalg.pinv(obj.sp_graph.L.todense()); obj.grad(reg=False)
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
            if self.option != 'default':
                self.c = res[0][-len(self.lre):]

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


## ----- ORIGINAL CODE -----

# import sys
# import numpy as np
# import scipy.sparse as sp
# from scipy.optimize import fmin_l_bfgs_b, minimize

# from feems.objective import neg_log_lik_w0_s2
# from feems import Objective
# from feems import SpatialGraph


# class Joint_Objective(Objective): 
#     def __init__(self, sp_graph):
#         """Inherit from the feems object Objective and overwrite some methods for evaluations 
#         and gradient of feems objective when residual variance is estimated jointly with edge weights

#         Args:
#             sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
#         """
#         super().__init__(sp_graph=sp_graph)   
        
#         # indicator whether optimizing residual variance jointly with edge weights
#         # None  : residual variance is holding fixed
#         # 1-dim : sinlge residual variance is estimated across all nodes
#         # n-dim : node-specific residual variances are estimated 
#         self.optimize_q = None 
        
#         # reg params for residual variance
#         self.lamb_q= None
#         self.alpha_q = None

#     def _comp_grad_obj(self):
#         """Computes the gradient of the objective function with respect to the
#         latent variables dLoss / dL
#         """
#         # compute inverses
#         self._comp_inv_lap()

#         self.comp_B = self.inv_cov - (1.0 / self.denom) * np.outer(
#             self.inv_cov_sum, self.inv_cov_sum
#         )
#         self.comp_A = self.comp_B @ self.sp_graph.S @ self.comp_B
#         M = self.comp_A - self.comp_B
#         self.grad_obj_L = self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

#         # grads
#         gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
#         gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
#         self.grad_obj = gradD - gradW
        
#         # grads for d diag(Jq^-1) / dq
#         if self.optimize_q == 'n-dim':
#             self.grad_obj_q = np.zeros(len(self.sp_graph))
#             self.grad_obj_q[:self.sp_graph.n_observed_nodes] = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad)        
            
#         if self.optimize_q == '1-dim':
#             self.grad_obj_q = self.sp_graph.n_snps * (np.diag(M) @ self.sp_graph.q_inv_grad) 

#     def _comp_grad_reg(self):
#         """Computes gradient"""
#         lamb = self.lamb
#         alpha = self.alpha

#         # avoid overflow in exp
#         # term_0 = 1.0 - np.exp(-alpha * self.sp_graph.w)
#         # term_1 = alpha * self.sp_graph.w + np.log(term_0)
#         # term_2 = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term_1)
#         # self.grad_pen = term_2 * (alpha / term_0)
#         term = alpha * self.sp_graph.w + np.log(
#             1 - np.exp(-alpha * self.sp_graph.w)
#         )  # avoid overflow in exp
#         self.grad_pen = self.sp_graph.Delta.T @ self.sp_graph.Delta @ (lamb * term)
#         self.grad_pen = self.grad_pen * (alpha / (1 - np.exp(-alpha * self.sp_graph.w)))
        
#         if self.optimize_q == 'n-dim':
#             lamb_q = self.lamb_q
#             alpha_q = self.alpha_q
            
#             term = alpha_q * self.sp_graph.s2 + np.log(
#                 1 - np.exp(-alpha_q * self.sp_graph.s2)
#             )
#             self.grad_pen_q = self.sp_graph.Delta_q.T @ self.sp_graph.Delta_q @ (lamb_q * term)
#             self.grad_pen_q = self.grad_pen_q * (alpha_q / (1 - np.exp(-alpha_q * self.sp_graph.s2)))

#     def loss(self):
#         """Evaluate the loss function given the current params"""
#         lamb = self.lamb
#         alpha = self.alpha

#         lik = self.neg_log_lik()
#         term_0 = 1.0 - np.exp(-self.alpha * self.sp_graph.w)
#         term_1 = alpha * self.sp_graph.w + np.log(term_0)
#         pen = 0.5 * lamb * np.linalg.norm(self.sp_graph.Delta @ term_1) ** 2
                
#         if self.optimize_q == 'n-dim':
#             lamb_q = self.lamb_q
#             alpha_q = self.alpha_q
                
#             term_0 = 1.0 - np.exp(-self.alpha_q * self.sp_graph.s2)
#             term_1 = alpha_q * self.sp_graph.s2 + np.log(term_0)
#             pen = 0.5 * lamb_q * np.linalg.norm(self.sp_graph.Delta_q @ term_1) ** 2                

#         # loss
#         loss = lik + pen
#         return loss        


# def loss_wrapper(z, obj):
#     """Wrapper function to optimize z=log(w) which returns the loss and gradient"""                
#     n_edges = obj.sp_graph.size()
#     theta = np.exp(z)
#     theta0 = theta[:n_edges]
#     obj.sp_graph.comp_graph_laplacian(theta0)
#     if obj.optimize_q is not None:
#         theta1 = theta[n_edges:]
#         obj.sp_graph.comp_precision(s2=theta1)
#     obj.inv()
#     obj.grad()                

#     # loss / grad
#     loss = obj.loss()
#     if obj.optimize_q is None:
#         grad = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
#     elif obj.optimize_q == 'n-dim':
#         grad = np.zeros_like(theta)
#         grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
#         grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2 + obj.grad_pen_q * obj.sp_graph.s2
#     elif obj.optimize_q == '1-dim':
#         grad = np.zeros_like(theta)
#         grad[:n_edges] = obj.grad_obj * obj.sp_graph.w + obj.grad_pen * obj.sp_graph.w
#         grad[n_edges:] = obj.grad_obj_q * obj.sp_graph.s2          
#     return (loss, grad)
# 
# class Joint_SpatialGraph(SpatialGraph):
#     def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True):
#         """Inherit from the feems object SpatialGraph and overwrite some methods for 
#         estimation of edge weights and residual variance jointly
#         """               
#         super().__init__(genotypes=genotypes,
#                          sample_pos=sample_pos,
#                          node_pos=node_pos,
#                          edges=edges,
#                          scale_snps=scale_snps) 
        
#     # ------------------------- Data -------------------------        
        
#     def comp_precision(self, s2):
#         """Computes the residual precision matrix"""
#         o = self.n_observed_nodes
#         self.s2 = s2
#         if 'array' in str(type(s2)) and len(s2) > 1:
#             self.q = self.n_samples_per_obs_node_permuted/self.s2[:o]
#         elif 'array' in str(type(s2)) and len(s2) == 1:
#             self.s2 = s2[0]
#             self.q = self.n_samples_per_obs_node_permuted / self.s2
#         else:
#             self.q = self.n_samples_per_obs_node_permuted / self.s2
#         self.q_diag = sp.diags(self.q).tocsc()
#         self.q_inv_diag = sp.diags(1.0 / self.q).tocsc()
#         self.q_inv_grad = -1.0 / self.n_samples_per_obs_node_permuted
#         if 'array' in str(type(s2)) and len(s2) > 1:
#             self.q_inv_grad = -sp.diags(1./self.n_samples_per_obs_node_permuted).tocsc()    
#         else:
#             self.q_inv_grad = -1./self.n_samples_per_obs_node_permuted   
            
#     # ------------------------- Optimizers -------------------------
    
#     def fit_null_model(self, verbose=True):
#         """Estimates of the edge weights and residual variance
#         under the model that all the edge weights have the same value
#         """
#         obj = Joint_Objective(self)
#         res = minimize(neg_log_lik_w0_s2, [0.0, 0.0], method="Nelder-Mead", args=(obj))
#         assert res.success is True, "did not converge"
#         w0_hat = np.exp(res.x[0])
#         s2_hat = np.exp(res.x[1])
#         self.w0 = w0_hat * np.ones(self.w.shape[0])
#         self.s2 = s2_hat
#         self.comp_precision(s2=s2_hat)

#         # print update
#         self.train_loss = neg_log_lik_w0_s2(np.r_[np.log(w0_hat), np.log(s2_hat)], obj)
#         if verbose:
#             sys.stdout.write(
#                 (
#                     "constant-w/variance fit, "
#                     "converged in {} iterations, "
#                     "train_loss={:.7f}\n"
#                 ).format(res.nfev, self.train_loss)
#             )    
    
#     def fit(
#         self,
#         lamb,
#         w_init=None,
#         s2_init=None,
#         alpha=None,
#         optimize_q=None,
#         lamb_q=None,
#         alpha_q=None,
#         factr=1e7,
#         maxls=50,
#         m=10,
#         lb=-np.Inf,
#         ub=np.Inf,
#         maxiter=15000,
#         verbose=True,
#     ):
#         """Estimates the edge weights of the full model holding the residual
#         variance fixed using a quasi-newton algorithm, specifically L-BFGS.

#         Args:
#             lamb (:obj:`float`): penalty strength on weights
#             w_init (:obj:`numpy.ndarray`): initial value for the edge weights
#             s2_init (:obj:`int`): initial value for s2
#             alpha (:obj:`float`): penalty strength on log weights
#             optimize_q (:obj:'str'): indicator whether optimizing residual variances (None, 1-dim, n-dim)
#             lamb_q (:obj:`float`): penalty strength on the residual variances
#             alpha_q (:obj:`float`): penalty strength on log residual variances
#             factr (:obj:`float`): tolerance for convergence
#             maxls (:obj:`int`): maximum number of line search steps
#             m (:obj:`int`): the maximum number of variable metric corrections
#             lb (:obj:`int`): lower bound of log weights
#             ub (:obj:`int`): upper bound of log weights
#             maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
#             verbose (:obj:`Bool`): boolean to print summary of results
#         """
#         # check inputs
#         assert lamb >= 0.0, "lambda must be non-negative"
#         assert type(lamb) == float, "lambda must be float"
#         assert type(factr) == float, "factr must be float"
#         assert maxls > 0, "maxls must be at least 1"
#         assert type(maxls) == int, "maxls must be int"
#         assert type(m) == int, "m must be int"
#         assert type(lb) == float, "lb must be float"
#         assert type(ub) == float, "ub must be float"
#         assert lb < ub, "lb must be less than ub"
#         assert type(maxiter) == int, "maxiter must be int"
#         assert maxiter > 0, "maxiter be at least 1"

#         # init from null model if no init weights are provided
#         if w_init is None and s2_init is None:
#             # fit null model to estimate the residual variance and init weights
#             self.fit_null_model(verbose=verbose)              
#             w_init = self.w0
#         else:
#             # check initial edge weights
#             assert w_init.shape == self.w.shape, (
#                 "weights must have shape of edges"
#             )
#             assert np.all(w_init > 0.0), "weights must be non-negative"
#             self.w0 = w_init
#             self.comp_precision(s2=s2_init)

#         # prefix alpha if not provided
#         if alpha is None:
#             alpha = 1.0 / self.w0.mean()
#         else:
#             assert type(alpha) == float, "alpha must be float"
#             assert alpha >= 0.0, "alpha must be non-negative"

#         # run l-bfgs
#         obj = Joint_Objective(self)
#         obj.optimize_q = optimize_q
#         obj.lamb = lamb
#         obj.alpha = alpha
#         x0 = np.log(w_init)
#         if obj.optimize_q is not None:
#             obj.lamb_q = lamb_q
#             obj.alpha_q = alpha_q
#             s2_init = self.s2 if obj.optimize_q=="1-dim" else self.s2*np.ones(len(self))
#             x0 = np.r_[np.log(w_init), np.log(s2_init)]
#         res = fmin_l_bfgs_b(
#             func=loss_wrapper,
#             x0=x0,
#             args=[obj],
#             factr=factr,
#             m=m,
#             maxls=maxls,
#             maxiter=maxiter,
#             approx_grad=False,
#             bounds=[(lb, ub) for _ in range(x0.shape[0])],
#         )
#         if maxiter >= 100:
#             assert res[2]["warnflag"] == 0, "did not converge"
#         if obj.optimize_q is not None:
#             self.w = np.exp(res[0][:self.size()])
#             self.s2 = np.exp(res[0][self.size():])
#         else:    
#             self.w = np.exp(res[0])

#         # print update
#         self.train_loss, _ = loss_wrapper(res[0], obj)
#         if verbose:
#             sys.stdout.write(
#                 (
#                     "lambda={:.7f}, "
#                     "alpha={:.7f}, "
#                     "converged in {} iterations, "
#                     "train_loss={:.7f}\n"
#                 ).format(lamb, alpha, res[2]["nit"], self.train_loss)
#             )    