#!/usr/bin/env python

# Python script to run FEEMS/FEEMSmix from a terminal versus interactively
# Please be mindful of default options when fitting and plotting 
# If a full fit has already been run, then you can comment out specific 
# chunks and feed in appropriate values where needed
# April 2025

# importing libraries
# base
import numpy as np
from importlib import resources
from sklearn.impute import SimpleImputer
from pandas_plink import read_plink
import statsmodels.api as sm
import pickle

# viz
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# feems
from feems.utils import prepare_graph_inputs
from feems.cross_validation import run_cv_joint
from feems.viz import plot_FEEMSmix_summary
from feems import SpatialGraph, Viz

# change matplotlib fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = "Arial"

#---------- CHANGE VARIABLES BELOW ----------
path_to_plink = "/path/to/plink/files"
path_to_sample_coords = "/path/to/sample/coordinates" 
K = 3 # number of long-range edges to fit
path_to_output_dir = "/path/to/output/dir"

# optional, if chosen change outer=outer in prepare_graph_output
# outer = np.loadtxt("/path/to/outer/boundary/file")

#---------- INPUT FILES ----------
print('Reading in input data...')
(bim, fam, G) = read_plink(path_to_plink)
coord = np.loadtxt(path_to_sample_coords)

# imputing any missing genotypes
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
genotypes = imp.fit_transform((np.array(G)).T)
print("n_samples={}, n_snps={}\n".format(genotypes.shape[0], genotypes.shape[1]))

# discrete global grid (DGG), could supply custom triangular grid
data_path = str(resources.files('feems') / 'data')
grid_path = "{}/grid_250.shp".format(data_path)

outer, edges, grid, _ = prepare_graph_inputs(coord=coord, 
                                             ggrid=grid_path,
                                             translated=False, 
                                             buffer=1,
                                             outer=None)

#---------- GRAPH SETUP ----------
print('\nSetting up graph...')
sp_graph = SpatialGraph(genotypes, coord, grid, edges, scale_snps=True)

# change projection here
projection = ccrs.PlateCarree()

#---------- CROSS VALIDATION ----------
print('\nRunning cross-validation scheme...')
## this chunk only needs to be run ONCE per data set
# choosing a discrete log-space grid between 0.01 and 100 
lamb_grid = np.geomspace(0.01, 100., 10, endpoint=True)[::-1]
lamb_q_grid = np.geomspace(0.01, 100., 5, endpoint=True)[::-1]

# using only 5-fold here for faster runtime 
# but recommended is leave-one-out (default: n_folds = None)
cv_err = run_cv_joint(sp_graph, lamb_grid, lamb_q_grid, n_folds=5, factr=1e10)
mean_cv_err = np.mean(cv_err, axis=0)

lamb_q_cv = lamb_q_grid[np.where(mean_cv_err == np.min(mean_cv_err))[0][0]]
lamb_cv = lamb_grid[np.where(mean_cv_err == np.min(mean_cv_err))[1][0]]
print(r"\nlambda_CV values: ({}, {})".format(lamb_cv, lamb_q_cv))

#---------- BASELINE FEEMS FIT ----------
print('\nFitting baseline FEEMS...')
sp_graph.fit(lamb = lamb_cv, lamb_q = lamb_q_cv, optimize_q='n-dim')

# visualizing the baseline migration surface
fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=projection)  
v = Viz(ax, sp_graph, projection=projection, edge_width=.5, 
        edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
        obs_node_size=7.5, sample_pt_color="black", 
        cbar_font_size=10)
v.draw_map(); v.draw_edges(use_weights=True)
v.draw_obs_nodes(use_ids=False); v.draw_edge_colorbar()
fig.savefig(path_to_output_dir+"/baseline.pdf")

#---------- FINDING OUTLIERS ---------- 
outliers_df = sp_graph.extract_outliers(fraction_of_pairs=0.01)

#---------- FEEMSMIX FIT ----------
print('\nFitting FEEMSmix with K = {} LREs...'.format(K))
# see docs/notebooks/further-exploration.ipynb for other modes in which to run this fit
seq_results = sp_graph.sequential_fit(
    outliers_df=outliers_df, 
    lamb=lamb_cv, lamb_q=lamb_q_cv, optimize_q='n-dim', 
    nedges=K, nedges_to_same_deme=2, top=10,
    search_area='all',
    fraction_of_pairs=0.01
)

# storing the FEEMSmix output in a pickle for easy access
filehandler = open(path_to_output_dir+"/seq_results.pkl", 'wb')
pickle.dump(seq_results, filehandler)

# visualizing the LREs over the baseline fit
fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=projection)  
v = Viz(ax, sp_graph, projection=projection, edge_width=.5, 
        edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
        obs_node_size=7.5, sample_pt_color="black", 
        cbar_font_size=10)
v.draw_map(); v.draw_edges(use_weights=True); v.draw_edge_colorbar()
v.draw_LREs(seq_results)
v.draw_c_colorbar()
fig.savefig(path_to_output_dir+"/LREs.pdf")

# final summary of FEEMSmix fit
plot_FEEMSmix_summary(seq_results, sequential=True)
fig = plt.gcf()
fig.savefig(path_to_output_dir+"/FEEMSmix_summary.pdf")
