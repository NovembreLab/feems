"""The following helper functions are adapted from Ben Peters' code:
https://github.com/NovembreLab/eems-around-the-world/blob/master/subsetter/
"""

from __future__ import absolute_import, division, print_function

import fiona
import numpy as np
import scipy as sp
from scipy.stats import norm, cauchy
from scipy.special import digamma, polygamma
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, Polygon, shape
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def load_tiles(s):
    tiles = fiona.collection(s)
    return [shape(t["geometry"]) for t in tiles]


def wrap_longitude_tiles(tile, long):
    tile = Point(tile)
    if np.max(tile.xy[0]) < long or np.min(tile.xy[0]) < long:
        tile = translate(tile, xoff=360.0)
    return tile.xy[0][0], tile.xy[1][0]


def create_tile_dict(tiles, bpoly, translated, long): 
    pts = dict()  # dict saving ids
    rev_pts = dict()
    edges = set()
    pts_in = dict()  # dict saving which points are in region

    for c, poly in enumerate(tiles):
        x, y = poly.exterior.xy
        points = zip(np.round(x, 3), np.round(y, 3))
        if translated:
            points = [wrap_longitude_tiles(p, long) for p in points] 
        else:
            points = [p for p in points]
        for p in points:
            if p not in pts_in:
                # check if point is in region
                with np.errstate(invalid="ignore"):
                    pts_in[p] = bpoly.intersects(Point(p))
                if pts_in[p]:
                    pts[p] = len(pts)  # if so, give id
                    rev_pts[len(rev_pts)] = p

        for i in range(3):
            pi, pj = points[i], points[i + 1]
            if pts_in[pi] and pts_in[pj]:
                if pts[pi] < pts[pj]:
                    edges.add((pts[pi] + 1, pts[pj] + 1))
                else:
                    edges.add((pts[pj] + 1, pts[pi] + 1))

    pts = [Point(rev_pts[p]) for p in range(len(rev_pts))]
    return pts, rev_pts, edges


def unique2d(a):
    x, y = a.T
    b = x + y * 1.0j
    idx = np.unique(b, return_index=True)[1]
    return a[idx]


def get_closest_point_to_sample(points, samples):
    usamples = unique2d(samples)
    dists = dict(
        (tuple(s), np.argmin([Point(s).distance(Point(p)) for p in points]))
        for s in usamples
    )

    res = [dists[tuple(s)] for s in samples]
    return np.array(res)


def prepare_graph_inputs(coord, ggrid, translated=False, buffer=0, outer=None, wrap_longitude=-40):
    """Prepares the graph input files for feems adapted from Ben Peters
    eems-around-the-world repo

    Args:
        sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
        ggrid (:obj:`str`): path to global grid shape file
        translated (:obj:`bool`): to handle the 'date line problem'
        transform (:obj:`bool`): to translate x coordinates
        buffer (:obj:`float`) buffer on the convex hull of sample pts
        outer (:obj:`numpy.ndarray`): q x 2 matrix of coordinates of outer
            polygon
        wrap_longitude (:obj:`int`): flag to pass in a specific value of the longitude from which to start wrapping (i.e., more fine-tuned control of the translated flag). For example, with North America (and mainly Alaska), we recommend a longitude of -40.
    """
    # no outer so construct with buffer
    if outer is None:
        points = MultiPoint([(x, y) for x, y in coord])
        xy = points.convex_hull.buffer(buffer).exterior.xy
        outer = np.array([xy[0].tolist(), xy[1].tolist()]).T

    if translated:
        outer[:, 0] = outer[:, 0] + 360.0

    # intersect outer with discrete global grid
    bpoly = Polygon(outer)
    bpoly2 = translate(bpoly, xoff=-360.0)
    tiles2 = load_tiles(ggrid)

    np.seterr(invalid='ignore')
    tiles3 = [t for t in tiles2 if bpoly.intersects(t) or bpoly2.intersects(t)]
    pts, rev_pts, e = create_tile_dict(tiles3, bpoly, translated, wrap_longitude)

    # construct grid array
    grid = []
    for i, v in rev_pts.items():
        grid.append((v[0], v[1]))
    grid = np.array(grid)

    # TODO add a more informative message on how users can get out of this pickle
    assert grid.shape[0] != 0, "grid is empty changing translation"

    # un-translate
    if translated:
        pts = []
        for p in range(len(rev_pts)):
            pts.append(Point(rev_pts[p][0] - 360.0, rev_pts[p][1]))
        grid[:, 0] = grid[:, 0] - 360.0
        outer[:, 0] = outer[:, 0] - 360.0

    # construct edge array
    edges = np.array(list(e))
    ipmap = get_closest_point_to_sample(pts, coord)
    res = (outer, edges, grid, ipmap)
    return res

def get_outlier_idx(emp_dist, fit_dist, fdr=0.1):
    bh = benjamini_hochberg(emp_dist, fit_dist, fdr=fdr)

    max_res_node = []
    for k in np.where(bh)[0]:
        # code to convert single index to matrix indices
        x = np.floor(np.sqrt(2*k+0.25)-0.5).astype('int')+1; y = int(k - 0.5*x*(x-1))

        max_res_node.append([x, y])

    return max_res_node

def get_robust_normal_pvals_lower(data, q=25):
    """
    Compute lower-tail p-values using robust normal parameters
    """

    # Get the percentiles
    lower_q, upper_q = np.percentile(data, [q, 100-q])
    
    z_lower = sp.stats.norm.ppf(q/100)
    z_upper = sp.stats.norm.ppf((100-q)/100)

    # The distance between percentiles in z-scores
    z_diff = z_upper - z_lower

    # First find sigma: (upper_q - lower_q) = sigma * (z_upper - z_lower)
    sigma = (upper_q - lower_q) / z_diff
    
    # Then find mu: mu = lower_q - sigma * z_lower
    mu = lower_q - sigma * z_lower
    
    z_scores = (data - mu)/sigma
    p_values = sp.stats.norm.cdf(z_scores)
    
    return p_values, mu, sigma

def parametric_bootstrap(sp_graph, emp_dist, fit_dist, lamb, lamb_q, optimize_q='n-dim', numdraws=100, fdr=0.1, dfscaler=5):
    """
    Apply the parametric bootstrap procedure to a obtain a list of p-values for points.
    Required:
        emp_cov, fit_cov (numpy.array)
    Optional:    
        numdraws (int), fdr (float), dfscaler (float): False discovery rate threshold.
    """
    from .objective import Objective, comp_mats
    
    n = sp_graph.n_observed_nodes

    tril_idx = np.tril_indices(n, k=-1)
    
    emp_distmat = np.zeros((n, n))
    emp_distmat[tril_idx] = emp_dist; emp_distmat += emp_distmat.T
    fit_distmat = np.zeros((n, n))
    fit_distmat[tril_idx] = fit_dist; fit_distmat += fit_distmat.T
    D_sample = np.zeros_like(fit_distmat)

    bootstrapped_distances = np.zeros((numdraws, n, n)); bootstrapped_fits = np.zeros_like(bootstrapped_distances)

    oldw = sp_graph.w; olds2 = sp_graph.s2

    C = np.vstack((-np.ones(n-1), np.eye(n-1))).T
    
    print('\n\tNumber of random draws in bootstrap:', end=' ')
    for d in range(numdraws):
        if d%20 == 0:
            print(d, end='...')
            
        # random draw given the EEMS scale matrix
        W = -sp.stats.wishart.rvs(df=sp_graph.n_snps/dfscaler, scale=-dfscaler*(C@fit_distmat@C.T)/sp_graph.n_snps)
        
        # holder for random distance matrix
        D_sample[1:,0] = -np.diagonal(W)/2
        D_sample[0,1:] = D_sample[1:,0]
        for i in range(1,n):
            for j in range(i+1,n):
                D_sample[i,j] = W[i-1,j-1] + D_sample[i,0] + D_sample[j,0]
                D_sample[j,i] = D_sample[i,j]
    
        bootstrapped_distances[d, :, :] = D_sample
    
        # constructing the covariance matrix for fitting
        Sigma = dist_to_cov(D_sample)

        # refitting the weights on the newly drawn samples
        sp_graph.S = Sigma
        
        sp_graph.fit(lamb=lamb, lamb_q=lamb_q, optimize_q=optimize_q)
        # sp_graph.fit(lamb=lamb, lamb_q=lamb_q, optimize_q=optimize_q, option='onlyc', long_range_edges=edges)
        
        objn = Objective(sp_graph); #objn.inv(); objn.grad(reg=False); objn.Linv_diag = objn._comp_diag_pinv()
        fit_cov2, _, _ = comp_mats(objn)
        bootstrapped_fits[d, :, :] = cov_to_dist(fit_cov2)

    print('done!')

    # updating the graph to the baseline state
    sp_graph.S = sp_graph.frequencies @ sp_graph.frequencies.T / sp_graph.n_snps
    sp_graph._update_graph(oldw, olds2)
    
    # p_values = np.zeros_like(sp_graph.Dhat)
    # for i in range(0,n):
    #     for j in range(i+1, n):  # Iterate over upper triangular 
    #         p_values[i, j] = np.mean(np.log(bootstrapped_distances[:, i, j]/bootstrapped_fits[:, i, j]) <= 
    #                              np.log(emp_distmat[i, j]/fit_distmat[i, j]))
    #         p_values[j, i] = p_values[i, j] 

    # p_values = p_values[np.tril_indices(n, k=-1)]

    log_ratios_emp = np.log(emp_distmat[tril_idx] / fit_distmat[tril_idx])
    log_ratios_boot = np.log(bootstrapped_distances[:, tril_idx[0], tril_idx[1]] /
                         bootstrapped_fits[:, tril_idx[0], tril_idx[1]])

    # Compute p-values for upper triangular elements
    p_values = np.mean(log_ratios_boot <= log_ratios_emp, axis=0)
    
    m = len(p_values)  # total number of hypotheses
    sorted_p_values = np.sort(p_values)

    sorted_indices = np.argsort(p_values)
    critical_values = np.array([fdr * (i + 1) / m for i in range(m)])

    # Find the largest p-value that meets the Benjamini-Hochberg criterion
    is_significant = sorted_p_values <= critical_values
    if np.any(is_significant):
        max_significant = np.max(np.where(is_significant)[0])  # max index where condition is true
    else:
        max_significant = -1  # no significant results
    # All p-values with rank <= max_significant are significant
    significant_indices = sorted_indices[:max_significant + 1]
    results = np.zeros(m, dtype=bool)
    results[significant_indices] = True
    # max_significant + 1 because indices are 0-based, but k should be 1-based
    return results

def benjamini_hochberg(emp_dist, fit_dist, fdr=0.1):
    """
    Apply the Benjamini-Hochberg procedure to a list of p-values to determine significance
    and the largest k such that p_(k) <= k/m * FDR.
    Required:
        emp_dist, fit_dist (numpy.array)
    Optional:    
        fdr (float): False discovery rate threshold.
    """

    logratio = np.log(emp_dist/fit_dist)
    # logratio = emp_dist-fit_dist

    mean_logratio = np.mean(logratio)
    var_logratio = np.var(logratio,ddof=1)
    logratio_norm = (logratio-mean_logratio)/np.sqrt(var_logratio)
    
    # p_value_neg = sp.stats.norm.cdf(logratio_norm)
    # p_values = p_value_neg
    ## if you want to look for outliers in the other directions
    # p_values=1-p_value_neg

    p_values, _, _ = get_robust_normal_pvals_lower(logratio_norm, 25)

    # X = sm.add_constant(fit_dist)
    # mod = sm.OLS(emp_dist, X)
    # res = mod.fit()
    # muhat, betahat = res.params

    # p_values, _, _ = get_robust_normal_pvals_lower(res.resid, 25)

    m = len(p_values)  # total number of hypotheses
    sorted_p_values = np.sort(p_values)

    sorted_indices = np.argsort(p_values)
    critical_values = np.array([fdr * (i + 1) / m for i in range(m)])

    # Find the largest p-value that meets the Benjamini-Hochberg criterion
    is_significant = sorted_p_values <= critical_values
    if np.any(is_significant):
        max_significant = np.max(np.where(is_significant)[0])  # max index where condition is true
    else:
        max_significant = -1  # no significant results
    # All p-values with rank <= max_significant are significant
    significant_indices = sorted_indices[:max_significant + 1]
    results = np.zeros(m, dtype=bool)
    results[significant_indices] = True
    # max_significant + 1 because indices are 0-based, but k should be 1-based
    return results

def left_tail_gauss_outliers(emp_dist, fit_dist, alpha=0.05, fdr_threshold=0.1, return_z=False, plot_results=False):
    """
    • Computes stat = log(emp/fit)
    • Fits σ̂ = sqrt(mean(stat²))  (MLE with μ fixed = 0)
    • Flags points in left tail with P(X < stat_i) < alpha.

    Returns
    -------
    outliers : boolean array
    z        : optional z-scores
    """
    stat = np.log(emp_dist / fit_dist) - np.mean(np.log(emp_dist / fit_dist))

    # MLE of σ with μ=0
    sigma_hat = np.sqrt(np.mean(stat**2))

    # z-scores relative to N(0,σ̂²)
    z = stat / sigma_hat

    # one-sided p‑values for left tail
    p_left = norm.cdf(z)

    outliers = p_left < alpha   # extreme negative side only

    outliers, p_adj, _, _ = multipletests(p_left, alpha=fdr_threshold, method='fdr_bh')

    if plot_results:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))

        xx = np.linspace(stat.min(), stat.max(), 600)
        pdf0 = norm.pdf(xx, 0, sigma_hat)
        plt.hist(stat, bins=50, density=True, alpha=0.4, color='grey')
        plt.plot(xx, pdf0, label='Null  N(0,σ²)')
        plt.legend(); plt.show()
    
    return (outliers, z) if return_z else outliers

def fixed_null_gauss_cauchy(emp_dist, fit_dist, post_thresh=0.9, n_iter=200, tol=1e-6, plot_results=False):
    """
    Null:  N(0, σ0²)
    Alt :  Cauchy(0, s1)
    Returns boolean array of outliers (left & right tails if extreme).
    """
    x = np.log(emp_dist / fit_dist)
    x = x - np.mean(x)
    
    # ── init ────────────────────────────────
    pi0    = 0.8
    sigma0 = np.std(x)
    s1     = 0.5 * sigma0           # initial half‑Cauchy scale

    ll_old = -np.inf
    two_over_pi = 2/np.pi

    for _ in range(n_iter):
        # PDFs
        p0 = pi0 * norm.pdf(x, 0, sigma0)
        p1 = (1-pi0) * two_over_pi * s1 / (x**2 + s1**2) * (x <= 0)

        denom = p0 + p1 + 1e-300
        r0 = p0 / denom
        r1 = p1 / denom

        # --- M‑step ---
        pi0    = r0.mean()
        sigma0 = np.sqrt(np.sum(r0 * x**2) / r0.sum())

        # crude but stable scale update for half‑Cauchy
        s1 = np.sqrt(np.sum(r1 * x**2) / r1.sum())
        s1 = max(s1, 1e-4)

        # convergence check
        ll = np.sum(np.log(denom))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    post_alt = r1
    outliers = post_alt > post_thresh

    # ---------- optional plot ----------
    if plot_results:
        xx = np.linspace(x.min(), x.max(), 600)
        pdf0 = pi0 * norm.pdf(xx, 0, sigma0)
        pdf1 = (1-pi0) * two_over_pi * s1 / (xx**2 + s1**2) * (xx <= 0)

        plt.hist(x, bins=50, density=True, alpha=0.4, color='grey')
        plt.semilogy(xx, pdf0, label=f'Null N(0, σ₀={sigma0:.3g})')
        plt.semilogy(xx, pdf1, label=f'Half‑Cauchy s₁={s1:.3g}',
                 color='orange')
        plt.title('Gaussian + half‑Cauchy mixture')
        plt.legend(); plt.show()

    return outliers

def gauss_t_mixture(emp_dist, fit_dist,
                    df=3,                 # degrees of freedom of Student‑t (heavy tail)
                    post_thresh=0.9,      # posterior prob cutoff
                    n_iter=200, tol=1e-6,
                    plot_results=True):
    """
    Null  : N(0, σ0²)
    Alt   : t_df(0, s1)
    Returns boolean array flagging outliers.
    """
    x = np.log(emp_dist / fit_dist) 
    x = x - np.mean(x)
    # ---------- initial params ----------
    pi0    = 0.8
    sigma0 = np.std(x)
    s1     = 2 * sigma0          # initial scale for t

    ll_old = -np.inf
    for _ in range(n_iter):
        p0 = pi0 * norm.pdf(x, 0, sigma0)
        p1 = (1-pi0) * t.pdf(x, df, loc=0, scale=s1)
        denom = p0 + p1 + 1e-300
        r0 = p0/denom
        r1 = p1/denom

        # ----- M‑step -----
        pi0    = r0.mean()
        sigma0 = np.sqrt(np.sum(r0 * x**2) / r0.sum())

        # scale update for Student‑t (approximate M‑step)
        s1 = np.sqrt(np.sum(r1 * x**2) / r1.sum())
        # small floor to avoid collapse
        s1 = max(s1, 1e-3)

        ll = np.sum(np.log(denom))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    post_alt = r1
    outliers = post_alt > post_thresh

    # ---------- optional plot ----------
    if plot_results:
        xx = np.linspace(x.min(), x.max(), 600)
        pdf0 = pi0 * norm.pdf(xx, 0, sigma0)
        pdf1 = (1-pi0) * t.pdf(xx, df, loc=0, scale=s1)
        plt.hist(x, bins=50, density=True, alpha=0.4, color='grey')
        plt.plot(xx, pdf0, label=f'Null N(0, σ₀²={sigma0:.3f})')
        plt.plot(xx, pdf1, label=f't(df={df}, s={s1:.3f}) alt', c='orange')
        plt.title('Gaussian + Student‑t mixture')
        plt.legend(); plt.show()

    return outliers

def fixed_null_gauss_laplace(emp_dist, fit_dist,
                             post_thresh=0.9,
                             n_iter=200,
                             tol=1e-6,
                             mu0=0.0,
                             plot_results=False):
    """
    Two‑component mixture:
        • Null  : N(mu0, σ0²)  (mu0 fixed, here 0)
        • Alt   : one‑sided Laplace on x ≤ mu0
                  f1(x) = (1/b) * exp((x-muL)/b)  for x ≤ muL
    Returns boolean array of outlier flags.
    """
    x = np.log(emp_dist / fit_dist)
    x = x - np.mean(x)
    
    N = x.size

    # ---- initial parameters ----
    pi0   = 0.8
    sigma0 = np.std(x)
    muL   = 0 #np.percentile(x, 25)    # initial Laplace location
    b     = np.std(x)        # initial scale

    ll_old = -np.inf
    for _ in range(n_iter):
        # densities for each component
        p0 = pi0 * norm.pdf(x, mu0, sigma0)
        p1 = (1 - pi0) * (1./b) * np.exp((x - muL)/b) * (x <= muL)

        denom = p0 + p1 + 1e-300
        r0 = p0 / denom
        r1 = p1 / denom

        # ---- M‑step ----
        pi0 = r0.mean()
        sigma0 = np.sqrt(np.sum(r0*(x - mu0)**2) / r0.sum())

        # weighted median for Laplace location
        # muL_old = muL
        # cdf = np.cumsum(r1[np.argsort(x)])
        # muL = np.sort(x)[np.searchsorted(cdf, r1.sum()/2.)]

        # weighted mean absolute deviation for scale
        b = np.sum(r1 * (muL - x) * (x <= muL)) / r1.sum()
        b = max(b, 1e-6)

        # convergence check
        ll = np.sum(np.log(denom))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    post_alt = r1
    outliers = post_alt > post_thresh

    if plot_results:
        xx = np.linspace(x.min(), x.max(), 600)
        pdf0 = pi0 * norm.pdf(xx, mu0, sigma0)
        pdf1 = (1 - pi0) * (1./b) * np.exp((xx - muL)/b) * (xx <= muL)
        plt.hist(x, bins=50, density=True, alpha=0.4, color='grey')
        plt.plot(xx, pdf0, label='Null  N(0,σ₀²)')
        plt.plot(xx, pdf1, label=f'Laplace tail (μ={muL:.3f}, b={b:.3f})',
                 color='orange')
        plt.title('Gaussian + Laplace mixture')
        plt.legend(); plt.show()

    return outliers

def fixed_null_gauss_exp(emp_dist, fit_dist, threshold=0.9, n_iter=200, tol=1e-6,
                         mu0=0.0, plot_results=False):
    """
    Two‑component mixture:  Null = N(mu0, sigma0²)   Alt = shift‑Exponential.
    Alt pdf:  f1(x) = λ exp[ λ (x - x0) ]   for x <= x0,   0 otherwise.
    (Here x0 = mu0 so distribution is only on the left side.)
    """
    x = np.log(emp_dist / fit_dist)
    x = x - np.mean(x)
    
    N = x.size

    # initialise parameters
    pi0 = 0.9
    sigma0 = np.std(x)
    lam   = 0.1 / sigma0           # controls how sharp the left tail is
    x0    = mu0                     # cut‑off

    ll_old = -np.inf
    for _ in range(n_iter):
        # ---------- E‑step ----------
        p0 = pi0 * norm.pdf(x, mu0, sigma0)
        p1 = (1-pi0) * lam*np.exp(lam*(x-mu0)) * (x <= x0)
        denom = p0 + p1 + 1e-300
        r0 = p0/denom
        r1 = p1/denom

        # ---------- M‑step ----------
        pi0 = r0.mean()
        # null variance update
        sigma0 = np.sqrt(np.sum(r0*(x-mu0)**2) / r0.sum())
        # alt λ update   (only uses left‑tail points)
        lam = r1.sum() / np.sum(r1*(mu0 - x))

        # ---------- check convergence ----------
        ll = np.sum(np.log(denom))
        if np.abs(ll-ll_old) < tol:
            break
        ll_old = ll

    post_alt = r1
    outliers = post_alt > threshold

    if plot_results:
        xx = np.linspace(x.min(), x.max(), 500)
        pdf_null = pi0*norm.pdf(xx, mu0, sigma0)
        pdf_alt  = (1-pi0)*lam*np.exp(lam*(xx-mu0))*(xx<=mu0)
        plt.hist(x, bins=50, density=True, color='grey', alpha=0.4)
        plt.plot(xx, pdf_null, label='Null N(0,σ²)')
        plt.plot(xx, pdf_alt,  label='Alt  Exp(λ)', color='orange')
        plt.legend(); plt.show()

    return outliers

def mixture_model_fixed_null(emp_dist, fit_dist, threshold=0.9, n_iter=100, tol=1e-6, plot_results=False):
    """
    Outlier detection by 2-component Gaussian mixture,
    with component 0 forced to have mean=0 (the null).
    
    Required:
        emp_dist, fit_dist: arrays of same length

    Optional:
        threshold: posterior probability cutoff for calling outlier
        n_iter: maximum EM iterations
        tol: EM convergence tolerance on log-likelihood
        plot_results: if True, shows histogram + fitted Gaussians
    
    Returns:
        outliers: boolean array (True = flagged as outlier)
    """
    # 1) statistic 
    stat = np.log(emp_dist / fit_dist)

    # centering
    stat = stat - np.mean(stat)

    N = len(stat)
    # 2) initialize parameters
    #    π0, π1 = 0.8 / 0.2
    pi0, pi1 = 0.8, 0.2
    mu0 = 0.0               # fixed
    mu1 = np.percentile(stat, 5)  # outlier mean init
    sigma0 = np.std(stat)       # null σ
    sigma1 = np.std(stat)       # outlier σ

    ll_old = -np.inf

    # EM loop
    for _ in range(n_iter):
        # E-step: compute responsibilities
        p0 = pi0 * norm.pdf(stat, mu0, sigma0)
        p1 = pi1 * norm.pdf(stat, mu1, sigma1)
        denom = p0 + p1 + 1e-300
        r0 = p0 / denom
        r1 = p1 / denom

        # M-step: update π and μ1, σ0, σ1 (but keep μ0=0)
        pi0 = np.mean(r0)
        pi1 = 1 - pi0

        # null variance:
        sigma0 = np.sqrt(np.sum(r0 * (stat - mu0)**2) / np.sum(r0))
        # outlier mean & variance:
        mu1 = np.sum(r1 * stat) / np.sum(r1)
        sigma1 = np.sqrt(np.sum(r1 * (stat - mu1)**2) / np.sum(r1))

        # check convergence (log-likelihood)
        ll = np.sum(np.log(p0 + p1 + 1e-300))
        if np.abs(ll - ll_old) < tol:
            break
        ll_old = ll

    # posterior outlier probability = r1
    outliers = (r1 > threshold)

    # optional plotting
    if plot_results:
        import matplotlib.pyplot as plt
        xgrid = np.linspace(stat.min(), stat.max(), 500)
        h = plt.hist(stat, bins=50, density=True, alpha=0.4, color='gray')
        plt.plot(xgrid, pi0*norm.pdf(xgrid, mu0, sigma0),  label='null (μ=0)',   c='C0')
        plt.plot(xgrid, pi1*norm.pdf(xgrid, mu1, sigma1),  label='outlier',      c='C1')
        plt.legend()
        plt.title("Fixed-null GMM fit on statistic")
        plt.xlabel("log(emp/fit)")
        plt.show()

    return outliers


def mixture_model_outlier_detection(emp_dist, fit_dist, threshold=0.9, plot_results=False):
    """
    Mixture model outlier detection with optional constraint on component separation.

    Required:
        emp_dist: array-like, empirical observed distances
        fit_dist: array-like, fitted distances

    Optional:
        threshold: float, posterior probability threshold for outlier detection (default 0.9)

    Returns:
        outlier_flags: boolean array, True if detected as an outlier
    """

    # Step 1: Compute standardized log ratio statistic
    stat = np.log(emp_dist / fit_dist)
    stat = (stat - np.mean(stat)) / np.std(stat)  # optional: center and scale

    stat = stat.reshape(-1, 1)  # sklearn expects 2D input
    gmm = GaussianMixture(n_components=2, random_state=0).fit(stat)

    # # Predict posterior probabilities
    # probs = gmm.predict_proba(stat)
    
    # # Pick the component with the lower mean (outliers = more negative)
    # outlier_component = np.argmin(gmm.means_.flatten())
    # outlier_probs = probs[:, outlier_component]
    
    # return outlier_probs > threshold
    means = gmm.means_.flatten()
    covs = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    responsibilities = gmm.predict_proba(stat.reshape(-1, 1))
    minor_component = np.argmin(means)  # Target the lower-mean component (lower stat → outlier)
    prob_minor = responsibilities[:, minor_component]

    # Step 3: Declare outliers
    outlier_flags = prob_minor > threshold

    # Step 4: Optional plotting
    if plot_results:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left panel: histogram and GMM fits ---
        x_grid = np.linspace(stat.min()-1, stat.max()+1, 1000)
        y0 = weights[0] * norm.pdf(x_grid, means[0], covs[0])
        y1 = weights[1] * norm.pdf(x_grid, means[1], covs[1])

        axs[0].hist(stat, bins=50, density=True, alpha=0.5, color='gray', label='Data Histogram')
        axs[0].plot(x_grid, y0, label='Component 0')
        axs[0].plot(x_grid, y1, label='Component 1')
        axs[0].legend()
        axs[0].set_title("Fitted Gaussian Mixture on Statistic")
        axs[0].set_xlabel("Standardized log(emp/fit)")
        axs[0].set_ylabel("Density")

        # --- Right panel: scatterplot of distances ---
        sc = axs[1].scatter(fit_dist, emp_dist, c=prob_minor, cmap='Reds', edgecolor='k', s=20)
        plt.colorbar(sc, ax=axs[1], label='Posterior Outlier Probability')
        plt.plot(fit_dist[outlier_flags],emp_dist[outlier_flags],'r*',alpha=0.6,markersize=15); 
        axs[1].set_xlabel("Fitted Distance")
        axs[1].set_ylabel("Empirical Distance")
        axs[1].set_title("Distances colored by Outlier Posterior")
        plt.tight_layout()
        plt.show()

    return outlier_flags
    
def pairwise_PCA_distances(genotypes, numPC = None):
    """Function to compute pairwise distance between individuals on a PCA plot
    genotypes (matrix) : input used for FEEMSmix
    numPC (int) : number of PCs to use when computing the distances
    """
    n, p = genotypes.shape
    
    if numPC is None:
        numPC = n-1
        
    pca = PCA(n_components=numPC)
    pcacoord = pca.fit_transform((genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0))

    pcdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(pcacoord, metric='euclidean'))
    D_geno = sp.spatial.distance.squareform(sp.spatial.distance.pdist(genotypes, metric="sqeuclidean")) / p
    tril_idx = np.tril_indices(n, k=-1)
    y = D_geno[tril_idx]
    x = pcdist[tril_idx]

    return x, y

def pairwise_admixture_distances(pfile, qfile, genotypes):
    """Function to compute pairwise distance between individuals based on the admixture model G = 2QP^\top
    K (number of ancestral populations) will be inferred from the shape of the .P and .Q file
    Required:
        pfile (path) : path to .P file
        qfile (path) : path to .Q file
        genotypes (matrix) : input used for FEEMSmix
    """

    print("Reading in .P file...")
    P = np.loadtxt(pfile)
    K = P.shape[1]
    print("Number of loci: {:d}, K: {:d}".format(P.shape[0], K))

    print("Reading in .Q file...")
    Q = np.loadtxt(qfile)
    if K != Q.shape[1]:
        print("The number of source populations (K) do not match between the .P and .Q files")
        return 
    print("Number of individuals: {:d}".format(Q.shape[0]))

    G = 2 * Q @ P.T

    admixdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(G, metric='euclidean'))

    D_geno = sp.spatial.distance.squareform(sp.spatial.distance.pdist(genotypes, metric="sqeuclidean")) / P.shape[0]
    tril_idx = np.tril_indices(Q.shape[0], k=-1)
    y = D_geno[tril_idx]
    x = admixdist[tril_idx]

    return x, y
    
def cov_to_dist(S):
    """Convert a covariance matrix to a distance matrix
    """
    s2 = np.diag(S).reshape(-1, 1)
    ones = np.ones((s2.shape[0], 1))
    D = s2 @ ones.T + ones @ s2.T - 2 * S
    return D 

def dist_to_cov(D):
    """Convert a distance matrix to a covariance matrix."""
    n = D.shape[0]
    row_mean = np.mean(D, axis=1)
    col_mean = np.mean(D, axis=0)
    total_mean = np.mean(D)
    
    # Apply the transformation
    S = -0.5 * (D - row_mean - col_mean + total_mean)
    return S