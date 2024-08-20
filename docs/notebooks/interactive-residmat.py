#!/usr/bin/env python
# coding: utf-8

# In[22]:

from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go


# In[9]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# base
import numpy as np
import networkx as nx
from sklearn.impute import SimpleImputer
import pkg_resources
import itertools as it
import math
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
from copy import deepcopy
import scipy.sparse as sp
import pandas as pd
from pandas_plink import read_plink
from scipy.stats.distributions import chi2

# viz
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# feems
from feems.utils import prepare_graph_inputs
from feems import SpatialGraph, Viz, Objective, FeemsMix
from feems.sim import setup_graph, setup_graph_long_range, simulate_genotypes
from feems.spatial_graph import query_node_attributes
from feems.objective import comp_mats
from feems.cross_validation import run_cv
from feems.helper_funcs import plot_default_vs_long_range, comp_genetic_vs_fitted_distance, plot_estimated_vs_simulated_edges, cov_to_dist

# change matplotlib fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = "Arial"


# In[3]:


app = dash.Dash(__name__)


# In[4]:


df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])


# In[7]:


app.run_server()
# %tb


# In[11]:


n_rows, n_columns = 6, 10

lrn = [(0,59)]

## using 1.0 to ensure all nodes are sampled equally well (default params otherwise: 4x8 grid)
graph, coord, grid, edge = setup_graph_long_range(n_rows=n_rows, n_columns=n_columns, corridor_w=1.0, barrier_w=1.0, barrier_prob=1.0, long_range_nodes=lrn, long_range_edges=[5.])
w_init = np.array(list(nx.get_edge_attributes(graph, 'w').values()))

gen_test = simulate_genotypes(graph, target_n_snps=1000)


# In[17]:


# sp_Graph = SpatialGraph(gen_test, coord, grid, edge)
fig = plt.figure()
# projection=ccrs.EquidistantConic(central_longitude=np.median(coord[:,0]), central_latitude=np.median(coord[:,1]))
ax = fig.add_subplot(1, 1, 1, projection=projection)
v = Viz(ax, sp_Graph, projection=projection, edge_width=1.5, 
        edge_alpha=1, edge_zorder=100, sample_pt_size=20, 
        obs_node_size=7.5, sample_pt_color="black", 
        cbar_font_size=10)
v.draw_map()
v.draw_edges(use_weights=False)
v.draw_obs_nodes(use_ids=False) 
# v.draw_edge_colorbar()


# In[51]:
# fig
# use the wolves data set here...
data_path = pkg_resources.resource_filename("feems", "data/")
(bim, fam, G) = read_plink("{}/wolvesadmix".format(data_path))
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
genotypes = imp.fit_transform((np.array(G)).T)

print("n_samples={}, n_snps={}".format(genotypes.shape[0], genotypes.shape[1]))

# setup graph
coord = np.loadtxt("{}/wolvesadmix_longrange.coord".format(data_path), skiprows=1)  # sample coordinates
outer = np.loadtxt("{}/wolvesadmix.outer".format(data_path))  # outer coordinates
grid_path = "{}/grid_100.shp".format(data_path) 

# graph input files
outer, edges, grid, _ = prepare_graph_inputs(coord=coord, 
                                             ggrid=grid_path,
                                             translated=True,
                                             buffer=0,
                                             outer=outer)

sp_graph = SpatialGraph(genotypes, coord, grid, edges)

# In[51]:
# sp_graph.fit(lamb=0.8)
# tril_idx = np.tril_indices(sp_graph.n_observed_nodes, k=-1)
# obj = Objective(sp_graph)
# fit_cov, _, emp_cov = comp_mats(obj)
# fit_dist = cov_to_dist(fit_cov)[tril_idx]
# emp_dist = cov_to_dist(emp_cov)[tril_idx]

# X = sm.add_constant(fit_dist)
# mod = sm.OLS(emp_dist, X)
# res = mod.fit()

#resnode[np.tril_indices_from(resnode, k=-1)] = 1.0#res.resid
resmat = np.zeros((94,94))#np.arange(94*94).reshape(94,94)
# resmat[np.triu_indices_from(resmat, k=0)] = 0
resmat[np.tril_indices_from(resmat, k=-1)] = res.resid
resmat[np.triu_indices_from(resmat, k=0)] = 0.


# In[49]:
# below i will write code to create a simple bubble plot in plotly with the pops scaled based on sample size...
## DON'T FORGET TO JITTER COORDS!!
# jit_coord = coord + np.random.normal(loc=0.0, scale=0.0, size=coord.shape)
permuted_idx = np.array(list(nx.get_node_attributes(sp_graph, 'permuted_idx').values()))
rescoord = sp_graph.node_pos[permuted_idx[:sp_graph.n_observed_nodes], :]

# scale size with np.sqrt(sp_graph.n_samples_per_obs_node_permuted) but need to ensure that order of pops is the same...
# fig = go.Figure(go.Scattergeo(lat=jit_coord[:,1], lon=jit_coord[:,0], opacity=0.8, projection='geojson', center={'lat':np.median(jit_coord[:,1]),'lon':np.median(jit_coord[:,0])}, fitbounds='locations', title='Sampled populations', showlegend=False, textfont=dict(family='Arial')))
figloc = px.scatter_geo(lat=rescoord[:,1], lon=rescoord[:,0], opacity=0.8, projection='robinson', center={'lat':np.median(rescoord[:,1]),'lon':np.median(rescoord[:,0])}, fitbounds='locations', title='Sampled populations')
figloc.update_geos(resolution=50,coastlinecolor='#9fa0a4',landcolor='#cccccc') 
figloc.update_traces(marker=dict(color=['olivedrab']*111, 
size=8.*np.sqrt(sp_graph.n_samples_per_obs_node_permuted), 
line=dict(color='black', width=0.5)))


# In[50]:
# plotlyfig = plotly.tools.mpl_to_plotly(fig)
# plotlyfig
# resmat = np.arange(94*94).reshape(94,94)
resmat[np.triu_indices_from(resmat, k=0)] = 0
figres = px.imshow(resmat, color_continuous_scale='balance',color_continuous_midpoint=0,aspect='equal',title='Residual matrix', labels={'color':'residual'})
figres.update_layout(font_family='Arial')
figres.update_xaxes(showspikes=True)
figres.update_yaxes(showspikes=True)


# In[1]:
bigfig = make_subplots(rows=1, cols=2, specs=[[{"type": "scattergeo"}, {"type": "xy"}]], column_widths=[0.6,0.4])
# bigfig = make_subplots(rows=1, cols=1)

bigfig.add_trace(go.Scattergeo(lat=rescoord[:,1], lon=rescoord[:,0], opacity=0.8, textfont={'family':'Arial'}), row=1, col=1)
bigfig.update_traces(marker=dict(color=['olivedrab']*111, 
size=8.*np.sqrt(sp_graph.n_samples_per_obs_node_permuted), 
line=dict(color='black', width=0.5)))
bigfig.update_layout(autosize=True, clickmode='event')
bigfig.update_geos(resolution=50,coastlinecolor='#9fa0a4',landcolor='#cccccc',fitbounds='locations', projection={'type':'robinson'}, center={'lat':np.median(rescoord[:,1]),'lon':np.median(rescoord[:,0])})

bigfig.add_trace(go.Heatmap(z=resmat, colorscale='balance', showscale=True, zmid=0), row=1, col=2)
bigfig.update_layout(font_family='Arial')
bigfig.update_xaxes(showspikes=True)
bigfig.update_yaxes(showspikes=True)


# In[ ]:
## need to a @app.callback and update_graph - first, plot the locations - have the user select a couple of pops, then plot the residual matrix but highlight based on selection

## couple of things - first, see if you can update the heatmap with selected elements - highlight/zoom whatever, second, check the dcc.Graph and @app.Callback function specs to see what needs to be passed back and forth...

# app.layout = html.Div([
#     dcc.Graph(figure=bigfig, style={'width': 1500, 'height': 1000})
# ])

app.layout = html.Div([
    dcc.Graph(id='locs',figure=bigfig, style={'width': 1600, 'height': 800})
])


@app.callback(Output('locs', 'children'), Input('trace0', 'value'), Input('trace1', 'value'))
def on_click(x):
    if trace0 is None & trace1 is None:
        raise PreventUpdate
    elif trace0 is None:
        return 'The {} gives {} value...'.format(trace0,resmat[trace0,:])
    else:
        return 'The {} gives {} value...'.format(trace0,resmat[trace1,:])

app.run_server()




# %%
