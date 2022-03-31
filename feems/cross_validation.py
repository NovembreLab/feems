import math
from copy import deepcopy

import numpy as np
from sklearn.model_selection import KFold

from .objective import Objective, comp_mats
from .spatial_graph import query_node_attributes


def run_cv(
    sp_graph,
    lamb_grid,
    alpha_grid=None,
    n_folds=5,
    lb=1e-6,
    ub=1e6,
    factr=1e10,
    random_state=500,
    outer_verbose=True,
    inner_verbose=False,
    alpha_fact=1.0
):
    """Run cross-validation."""
    # s2 initialization
    sp_graph.fit_null_model(verbose=inner_verbose)
    w0 = sp_graph.w0
    s2 = sp_graph.s2
    if alpha_grid is None:
        alpha_grid = np.array([alpha_fact / w0[0]])

    # default is None i.e., leave-one-out CV
    if n_folds is None:
        n_folds = sp_graph.n_observed_nodes

    # setup cv indicies
    is_train = setup_k_fold_cv(sp_graph, n_folds, random_state=random_state)

    # CV error
    n_lamb = lamb_grid.shape[0]
    n_alpha = alpha_grid.shape[0]
    cv_err = np.empty((n_folds, n_lamb, n_alpha))

    # loop
    for fold in range(n_folds):
        if outer_verbose:
            print("\n fold=", fold)

        # partition into train and test sets
        if sp_graph.factor is not None:
            sp_graph.factor = None
        sp_graph_train, sp_graph_test = train_test_split(
            sp_graph, 
            is_train[:, fold]
        )

        # set of initialization for warmstart
        init_list = [w0]
        for a, alpha in enumerate(alpha_grid):
            w_init = init_list[-1]
            s2_init = s2
            for i, lamb in enumerate(lamb_grid):
                if outer_verbose:
                    print(
                        "\riteration lambda={}/{} alpha={}/{}".format(
                            i + 1, n_lamb, a + 1, n_alpha
                        ),
                        end="",
                    )
                # fit on train set
                lamb = float(lamb)
                alpha = float(alpha)
                sp_graph_train.fit(
                    lamb=lamb,
                    w_init=w_init,
                    s2_init=s2_init, optimize_q='n-dim', lamb_q=1., alpha_q=1., 
                    alpha=alpha,
                    factr=factr,
                    lb=math.log(lb),
                    ub=math.log(ub),
                    verbose=inner_verbose,
                )

                # evaluate on the validation set
                _, err = predict_snps(sp_graph, sp_graph_train, sp_graph_test)
                cv_err[fold, i, a] = err

                w_init = deepcopy(sp_graph_train.w)
                if i == 0:
                    init_list.append(w_init)

    return cv_err


def setup_k_fold_cv(sp_graph, n_splits=5, random_state=12):
    """Setup cross-validation indicies.

    Args:
        sp_graph (:obj:`SpatialGraph`): SpatialGraph class
        n_splits (:obj:`int`): number of CV folds
        random_state (:obj:`int`): random seed

    Returns:
        is_train (:obj:`numpy.ndarray`): n x k matrix storing boolean values
            determining
        if the sample is in training set for the kth fold
    """
    # number of individuals
    n_samples = sp_graph.sample_pos.shape[0]

    # get indicies
    permuted_idx = query_node_attributes(sp_graph, "permuted_idx")
    observed_permuted_idx = permuted_idx[: sp_graph.n_observed_nodes]
    assned_node_idx = sp_graph.assned_node_idx

    # splits data for cross-validation (holding out nodes)
    is_train = np.zeros((n_samples, n_splits), dtype=bool)
    kf = KFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )  # k-fold cv object

    for k, (train_node_idx, _) in enumerate(kf.split(observed_permuted_idx)):
        for i in range(n_samples):
            if assned_node_idx[i] in observed_permuted_idx[train_node_idx]:
                is_train[i, k] = True

    return is_train


def copy_spatial_graph(sp_graph, subsample_idx):
    """Copy SpatialGraph object and reassign spatial positions for subsamples

    Args:
        sp_graph (:obj:`SpatialGraph`): SpatialGraph class
        subsample_idx (:obj:`numpy.ndarray`): indexes indicating if the sample should be included

    Returns:
        sp_graph_copy (:obj:`SpatialGraph`): SpatialGraph class
    """
    sp_graph_copy = deepcopy(sp_graph)
    # add spatial coordinates to node attributes
    for i in range(len(sp_graph_copy)):
        sp_graph_copy.nodes[i]["n_samples"] = 0
        sp_graph_copy.nodes[i]["sample_idx"] = []

    sp_graph_copy.sample_pos = sp_graph.sample_pos[subsample_idx]
    sp_graph_copy._assign_samples_to_nodes(
        sp_graph_copy.sample_pos, sp_graph_copy.node_pos
    )
    sp_graph_copy._permute_nodes()

    n_samples_per_node = query_node_attributes(sp_graph_copy, "n_samples")
    permuted_idx = query_node_attributes(sp_graph_copy, "permuted_idx")
    n_samps = n_samples_per_node[permuted_idx]
    sp_graph_copy.n_samples_per_obs_node_permuted = n_samps[
        : sp_graph_copy.n_observed_nodes
    ]
    sp_graph_copy._create_perm_diag_op()  # create perm operator
    sp_graph_copy.factor = None

    # estimate allele frequencies at observed locations (in permuted order)
    sp_graph_copy.genotypes = sp_graph.genotypes[subsample_idx, :]
    sp_graph_copy._estimate_allele_frequencies()

    if sp_graph.scale_snps:
        sp_graph_copy.frequencies = sp_graph_copy.frequencies / np.sqrt(
            sp_graph.mu * (1 - sp_graph.mu)
        )

    # compute precision
    sp_graph_copy.comp_precision(s2=1)

    # estimate sample covariance matrix
    sp_graph_copy.S = (
        sp_graph_copy.frequencies @ sp_graph_copy.frequencies.T / sp_graph_copy.n_snps
    )

    return sp_graph_copy


def train_test_split(sp_graph, is_train):
    """Create SpatialGraph classes for train and test sets.

    Args:
        sp_graph (:obj:`SpatialGraph`): SpatialGraph class
        is_train (:obj:`numpy.ndarray`): indicies for train and validation sets

    Returns:
        train_sp_graph (:obj:`SpatialGraph`):
        test_sp_graph (:obj:`SpatialGraph`):
    """
    # copy SpatialGraph objects on train and test sets
    train_sp_graph = copy_spatial_graph(sp_graph, is_train)
    test_sp_graph = copy_spatial_graph(sp_graph, ~is_train)

    return train_sp_graph, test_sp_graph


def predict_snps(sp_graph, sp_graph_train, sp_graph_test):
    # create obj
    obj = Objective(sp_graph)

    # update graph laplacian
    obj.sp_graph.comp_graph_laplacian(sp_graph_train.w)

    # fitted covariance
    fit_cov, _, _ = comp_mats(obj)

    # predict SNPs
    n_snps = sp_graph.n_snps
    frequencies_ns = sp_graph.frequencies * np.sqrt(sp_graph.mu * (1 - sp_graph.mu))
    mu0 = frequencies_ns.mean(axis=0) / 2
    mu_f = np.sqrt(sp_graph.mu * (1 - sp_graph.mu))
    mu_frequencies = 2 * mu0 / mu_f
    frequencies = deepcopy(sp_graph.frequencies)

    ids = np.empty(sp_graph.n_observed_nodes, dtype=bool)
    permuted_idx = query_node_attributes(sp_graph, "permuted_idx")
    permuted_idx_train = query_node_attributes(sp_graph_train, "permuted_idx")
    for i, idx in enumerate(permuted_idx[: sp_graph.n_observed_nodes]):
        if idx in permuted_idx_train[: sp_graph_train.n_observed_nodes]:
            ids[i] = True
        else:
            ids[i] = False
    train_ids = np.argwhere(ids == True).reshape(-1)
    test_ids = np.argwhere(ids == False).reshape(-1)

    cov_te_tr = fit_cov[np.ix_(test_ids, train_ids)]
    cov_tr_tr = fit_cov[np.ix_(train_ids, train_ids)]
    pred_frequencies = mu_frequencies + cov_te_tr @ np.linalg.solve(
        cov_tr_tr, frequencies[train_ids, :] - mu_frequencies
    )
    l2_err = np.linalg.norm(
        pred_frequencies * mu_f - frequencies[test_ids] * mu_f
    ) ** 2 / (len(test_ids) * n_snps)

    return pred_frequencies, l2_err
