from __future__ import absolute_import, division, print_function

import cartopy.feature as cfeature
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Proj
import matplotlib.patches as patches

from .spatial_graph import query_node_attributes
from .objective import Objective

class Viz(object):
    def __init__(
        self,
        ax,
        sp_graph,
        weights=None, 
        oldweights=None, halfrange=100,
        projection=None,
        coastline_m="50m",
        coastline_linewidth=0.5,
        sample_pt_size=1,
        sample_pt_linewidth=0.5,
        sample_pt_color="#d9d9d9",
        sample_pt_jitter_std=0.0,
        sample_pt_alpha=1.0,
        sample_pt_zorder=2,
        obs_node_size=10,
        obs_node_textsize=7,
        obs_node_linewidth=0.5,
        obs_node_color="#d9d9d9",
        obs_node_alpha=1.0,
        obs_node_zorder=2,
        edge_color="#d9d9d9",
        edge_width=1,
        edge_alpha=1.0,
        edge_zorder=2,
        abs_max=2,
        cbar_font_size=12,
        cbar_nticks=3,
        cbar_orientation="horizontal",
        cbar_ticklabelsize=12,
        cbar_width="20%",
        cbar_height="5%",
        cbar_loc="lower left",
        cbar_bbox_to_anchor=(0.05, 0.2, 1, 1),
        # ell_scaler=np.sqrt(3.0) / 6.0,
        # ell_edgecolor="gray",
        # ell_lw=0.2,
        # ell_abs_max=0.5,
        target_dist_pt_size=10,
        target_dist_pt_linewidth=0.5,
        target_dist_pt_alpha=1.0,
        target_dist_pt_zorder=2,
        seed=1996,
    ):
        """A visualization module for feems

        Args:
        """
        # main attributes
        self.ax = ax
        self.ax.axis("off")
        self.sp_graph = sp_graph
        self.grid = sp_graph.node_pos
        self.coord = sp_graph.sample_pos
        self.projection = projection
        self.seed = seed
        np.random.seed = self.seed
        self.sp_graph.option = 'default'
        self.obj = Objective(sp_graph); self.obj.inv(); self.obj.grad(reg=False)

        # ------------------------- Attributes -------------------------
        self.coastline_m = coastline_m
        self.coastline_linewidth = coastline_linewidth

        # sample pt
        self.sample_pt_size = sample_pt_size
        self.sample_pt_linewidth = sample_pt_linewidth
        self.sample_pt_color = sample_pt_color
        self.sample_pt_zorder = sample_pt_zorder
        self.samplte_pt_alpha = sample_pt_alpha
        self.sample_pt_jitter_std = sample_pt_jitter_std
        self.sample_pt_alpha = sample_pt_alpha

        # obs nodes
        self.obs_node_size = obs_node_size
        self.obs_node_textsize = obs_node_textsize
        self.obs_node_alpha = obs_node_alpha
        self.obs_node_linewidth = obs_node_linewidth
        self.obs_node_color = obs_node_color
        self.obs_node_zorder = obs_node_zorder

        # edge
        self.edge_width = edge_width
        self.edge_alpha = edge_alpha
        self.edge_zorder = edge_zorder
        self.edge_color = edge_color

        # colorbar
        self.abs_max = abs_max
        self.cbar_font_size = cbar_font_size
        self.cbar_nticks = cbar_nticks
        self.cbar_orientation = cbar_orientation
        self.cbar_ticklabelsize = cbar_ticklabelsize
        self.cbar_width = cbar_width
        self.cbar_height = cbar_height
        self.cbar_loc = cbar_loc
        self.cbar_bbox_to_anchor = cbar_bbox_to_anchor

        # target correlations
        self.target_dist_pt_size = target_dist_pt_size
        self.target_dist_pt_linewidth = target_dist_pt_linewidth
        self.target_dist_pt_alpha = target_dist_pt_alpha
        self.target_dist_pt_zorder = target_dist_pt_zorder

        # colors
        self.eems_colors = [
            "#994000",
            "#CC5800",
            "#FF8F33",
            "#FFAD66",
            "#FFCA99",
            "#FFE6CC",
            "#FBFBFB",
            "#CCFDFF",
            "#99F8FF",
            "#66F0FF",
            "#33E4FF",
            "#00AACC",
            "#007A99",
        ]
        self.edge_cmap = clr.LinearSegmentedColormap.from_list(
            "eems_colors", self.eems_colors, N=256
        )
        self.edge_norm = clr.LogNorm(
            vmin=10 ** (-self.abs_max), vmax=10 ** (self.abs_max)
        )
        self.halfrange = halfrange
        self.change_norm = clr.CenteredNorm(halfrange=self.halfrange)
        self.dist_cmap = plt.get_cmap("viridis_r")
        self.c_cmap = plt.get_cmap('Greys')

        # extract node positions on the lattice
        self.idx = nx.adjacency_matrix(self.sp_graph).nonzero()

        # edge weights
        if weights is None:
            self.weights = recover_nnz_entries(sp_graph) 
        else:
            self.weights = weights
        self.norm_log_weights = np.log10(self.weights) - np.mean(np.log10(self.weights))

        if oldweights is not None:
            self.foldchange = recover_nnz_entries_foldchange(sp_graph, oldweights)
        self.n_params = int(len(self.weights) / 2)

        # plotting maps
        if self.projection is not None:
            self.proj = Proj(projection.proj4_init)
            self.coord = project_coords(self.coord, self.proj)
            self.grid = project_coords(self.grid, self.proj)

    def draw_map(self, latlong=True):
        """Draws the underlying map projection
        latlong: 
            - True to draw gridlines picked from underlying graph (default) or 
            - tuple of ([lats], [longs]) coordinates to draw custom gridlines or 
            - False for no gridlines"""
        
        with np.errstate(invalid='ignore'):
            self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        # self.ax.add_feature(cfeature.LAND, facecolor="#ffffff", zorder=0)
            self.ax.coastlines(
                self.coastline_m,
                color="#636363",
                linewidth=self.coastline_linewidth,
                zorder=0,
            )

        if latlong is not False:
            if latlong==True:
                gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, zorder=0)
                gl.top_labels=False; gl.right_labels=False
                gl.xlabel_style = {'rotation': 45}; gl.ylabel_style = {'rotation': 315}
            elif len(latlong)==2:
                gl = self.ax.gridlines(xlocs=latlong[1], ylocs=latlong[0], draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, zorder=0)
                gl.top_labels=False; gl.right_labels=False
                gl.xlabel_style = {'rotation': 45}; gl.ylabel_style = {'rotation': 315}
            else:
                print('Please specify valid option for latlong: True (default) or tuple/list of [[lats],[longs]] for custom gridlines. When specifying just a single coordinate list, leave an empty list for the other coordinate.')
            

    def draw_samples(self, labels=None):
        """Draw the individual sample coordinates"""
        jit_coord = self.coord + np.random.normal(
            loc=0.0, scale=self.sample_pt_jitter_std, size=self.coord.shape
        )
        self.ax.scatter(
            jit_coord[:, 0],
            jit_coord[:, 1],
            edgecolors="black",
            linewidth=self.sample_pt_linewidth,
            s=self.sample_pt_size,
            alpha=self.sample_pt_alpha,
            color=self.sample_pt_color,
            marker=".",
            zorder=self.sample_pt_zorder,
        )

        ## if labels are provided
        if labels is not None:
            for txt, i in labels:
                self.ax.annotate(txt, (jit_coord[i, 0], jit_coord[i, 1]), color='k', fontsize=1, ha='center', va='center') 

    def draw_obs_nodes(self, use_ids=False):
        """Draw the observed node coordinates"""
        permuted_idx = query_node_attributes(self.sp_graph, "permuted_idx")
        obs_perm_ids = permuted_idx[: self.sp_graph.n_observed_nodes]
        obs_grid = self.grid[obs_perm_ids, :]
        if use_ids:
            for i, perm_id in enumerate(obs_perm_ids):
                self.ax.text(
                    obs_grid[i, 0],
                    obs_grid[i, 1],
                    str(perm_id),
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=self.obs_node_textsize,
                    zorder=self.obs_node_zorder,
                )
        else:
            self.ax.scatter(
                obs_grid[:, 0],
                obs_grid[:, 1],
                edgecolors="black",
                linewidth=self.obs_node_linewidth,
                s=self.obs_node_size * np.sqrt(self.sp_graph.n_samples_per_obs_node_permuted),
                alpha=self.obs_node_alpha,
                color=self.obs_node_color,
                zorder=self.obs_node_zorder,
            )

    def draw_edges(self, use_weights=False, use_foldchange=False):
        """Draw the edges of the graph"""
        if not use_foldchange:
            if use_weights:
                nx.draw(
                    self.sp_graph,
                    ax=self.ax,
                    node_size=0.0,
                    edge_cmap=self.edge_cmap,
                    # edge_norm=self.edge_norm,
                    alpha=self.edge_alpha,
                    pos=self.grid,
                    width=self.edge_width,
                    edgelist=list(np.column_stack(self.idx)),
                    edge_color=self.norm_log_weights,
                    edge_vmin=-self.abs_max,
                    edge_vmax=self.abs_max,
                )
            else:
                nx.draw(
                    self.sp_graph,
                    ax=self.ax,
                    node_size=0.0,
                    alpha=self.edge_alpha,
                    pos=self.grid,
                    width=self.edge_width,
                    edgelist=list(np.column_stack(self.idx)),
                    edge_color=self.edge_color,
                )
        else:
            nx.draw(
                    self.sp_graph,
                    ax=self.ax,
                    node_size=0.0,
                    edge_cmap=self.edge_cmap,
                    # edge_norm=self.change_norm,
                    alpha=self.edge_alpha,
                    pos=self.grid,
                    width=self.edge_width,
                    edgelist=list(np.column_stack(self.idx)),
                    edge_color=self.foldchange,
                    edge_vmin=-self.halfrange,
                    edge_vmax=self.halfrange,
            )

    def draw_edge_colorbar(self):
        """Draws colorbar"""
        self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap, norm=self.edge_norm)
        # self.edge_sm._A = []
        self.edge_tick_locator = ticker.LogLocator(base=10, numticks=self.cbar_nticks)
        self.edge_axins = inset_axes(
            self.ax,
            width=self.cbar_width,
            height=self.cbar_height,
            loc=self.cbar_loc,
            bbox_to_anchor=self.cbar_bbox_to_anchor,
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )
        self.edge_cbar = plt.colorbar(
            self.edge_sm, cax=self.edge_axins, orientation=self.cbar_orientation
        )
        self.edge_cbar.locator = self.edge_tick_locator
        self.edge_cbar.update_ticks()
        self.edge_cbar.ax.tick_params(which="minor", length=0)
        # TODO make this nicefrac instead
        self.edge_cbar.ax.set_title(r"$w/\widebar{w}$", loc="center")
        self.edge_cbar.ax.set_title(
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size
        )
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)

        # add min and max ticks to colorbar based on data 
        if np.min(self.norm_log_weights)<-1:
            self.edge_cbar.ax.axvline(10**np.min(self.norm_log_weights), color='w')
        else:
            self.edge_cbar.ax.axvline(10**np.min(self.norm_log_weights), color='dimgrey')
        if np.max(self.norm_log_weights)>1:
            self.edge_cbar.ax.axvline(10**np.max(self.norm_log_weights), color='w')
        else:
            self.edge_cbar.ax.axvline(10**np.max(self.norm_log_weights), color='dimgrey')

    def draw_edge_change_colorbar(self):
        """Draws colorbar of relative change (but in both directions)"""
        self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap, norm=self.change_norm)
        # self.edge_sm._A = []
        self.edge_tick_locator = ticker.LinearLocator(numticks=3)
        self.edge_axins = inset_axes(
            self.ax,
            width=self.cbar_width,
            height=self.cbar_height,
            loc=self.cbar_loc,
            bbox_to_anchor=self.cbar_bbox_to_anchor,
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )
        self.edge_cbar = plt.colorbar(
            self.edge_sm, cax=self.edge_axins, orientation=self.cbar_orientation
        )
        self.edge_cbar.locator = self.edge_tick_locator
        self.edge_cbar.update_ticks()
        self.edge_cbar.ax.tick_params(which="minor", length=0)
        self.edge_cbar.ax.set_title(r"relative change (%)", loc="center")
        self.edge_cbar.ax.set_title(
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size
        )
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)

    def draw_arrow(self, lre, c, lw=1, hw=4, hl=6, tw=1.25, fs=10, alpha=0.8):
        # self.ax.arrow(self.grid[lre[0][0],0],self.grid[lre[0][0],1],dx=self.grid[lre[0][1],0]-self.grid[lre[0][0],0],dy=self.grid[lre[0][1],1]-self.grid[lre[0][0],1],ec=self.c_cmap(c), fc='k', length_includes_head=True,linewidth=lw,head_width=hw,head_length=hl)
        ## the code below is for cases when we have unsampled demes so the node IDs are permuted
        # if mode=='sampled':
        #     permuted_idx = query_node_attributes(self.sp_graph, "permuted_idx")
        #     obs_perm_ids = permuted_idx[: self.sp_graph.n_observed_nodes]
        #     obs_grid = self.grid[obs_perm_ids, :]
        #     self.ax.arrow(obs_grid[lre[0][0],0],obs_grid[lre[0][0],1],dx=obs_grid[lre[0][1],0]-obs_grid[lre[0][0],0],dy=obs_grid[lre[0][1],1]-obs_grid[lre[0][0],1],ec=self.c_cmap(c), fc=self.c_cmap(c), length_includes_head=True,linewidth=lw,head_width=hw,head_length=hl,alpha=alpha)
        #     self.ax.annotate(np.round(c,2),(obs_grid[lre[0][1],0],obs_grid[lre[0][1],1]),fontsize=fs)
        # else:
        #     self.ax.arrow(self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][0]],0],self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][0]],1],dx=self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][1]],0]-self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][0]],0],dy=self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][1]],1]-self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][0]],1],ec=self.c_cmap(c), fc=self.c_cmap(c), length_includes_head=True,linewidth=lw,head_width=hw,head_length=hl)
        #     self.ax.annotate(np.round(c,2),(self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][1]],0],self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[lre[0][1]],1]),fontsize=fs)

        ## drawing curved arrows
        # style = "Simple, tail_width=3, head_width=8, head_length=15"
        # kw = dict(arrowstyle=style, edgecolor='k', facecolor='k', zorder=5)
        # self.ax.add_patch(patches.FancyArrowPatch((self.grid[lre[0][0],0],self.grid[lre[0][0],1]), (self.grid[lre[0][1],0],self.grid[lre[0][1],1]), connectionstyle="arc3,rad=-.3", **kw))
        style = "Simple, tail_width={}, head_width={}, head_length={}".format(tw,hw,hl)
        kw = dict(arrowstyle=style, edgecolor='k', facecolor=self.c_cmap(c), zorder=5, linewidth=lw)
        self.ax.add_patch(patches.FancyArrowPatch((self.grid[lre[0][0],0],self.grid[lre[0][0],1]), (self.grid[lre[0][1],0],self.grid[lre[0][1],1]), connectionstyle="arc3,rad=-.3", **kw))

    def draw_c_colorbar(self, df=None):
        "Draws simple colorbar from 0 to 1 scale for admixture proportion"
        self.c_axins = inset_axes(self.ax, loc='upper right', width = "10%", height = "2%", borderpad=2)
        self.c_axins.set_title(r"$\hat{c}$", fontsize = self.cbar_font_size, loc='center')
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.c_cmap), cax=self.c_axins, shrink=0.1, orientation='horizontal', ticks=np.linspace(0,1,3))
        self.c_cbar.ax.tick_params(labelsize = self.cbar_ticklabelsize)
        if df is not None:
            self.c_cbar.ax.axvline(df['admix. prop.'].loc[df['scaled log-lik']==0].values[0], color='r', linewidth=1)

    def draw_c_contour(self, df, ll_thresh=-10, levels=3, fs=8):
        """Draws two tricontours of admix. prop. estimates & log-lik
        cest_levels: int or list"""
        ## VS: draw stars & crosses for MLE and destination
        # self.ax.tricontourf([self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[x[0]],0] for x in nodes],[self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[x[0]],1] for x in nodes],df['admix. prop.'],cmap='Greys',vmin=0,vmax=1,alpha=0.7,levels=cest_levels); 
        # CS = self.ax.tricontour([self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[x[0]],0] for x in nodes],[self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[x[0]],1] for x in nodes],df['admix. prop.'],cmap='Greys',vmin=0,vmax=1,alpha=0.7,levels=cest_levels); self.ax.clabel(CS, inline=1, fontsize=cest_fs, colors='k')
        # # drawing a star at the location of the MLE
        # self.ax.scatter(self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[nodes[np.argmin(loglik)][0]],0],self.grid[nx.get_node_attributes(self.sp_graph,'permuted_idx')[nodes[np.argmin(loglik)][0]],1],marker='*',zorder=2,facecolors='k',edgecolors='k')

        idx = np.where(df['scaled log-lik'] > ll_thresh)
        self.ax.tricontourf([self.grid[x[0],0] for x in df['(source, dest.)'].iloc[idx]],[self.grid[x[0],1] for x in df['(source, dest.)'].iloc[idx]], df['admix. prop.'].iloc[idx], cmap='Greys', vmin=0, vmax=1, alpha=0.7, extend=None, levels=levels); 
        CS = self.ax.tricontourf([self.grid[x[0],0] for x in df['(source, dest.)'].iloc[idx]],[self.grid[x[0],1] for x in df['(source, dest.)'].iloc[idx]], df['admix. prop.'].iloc[idx], cmap='Greys', vmin=0, vmax=1, alpha=0.7, extend=None, levels=levels)
        self.ax.clabel(CS, inline=1, fontsize=fs, colors='k')
        ## drawing a star at the location of the MLE
        # self.ax.scatter(self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],0],self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],1], marker='*', zorder=2, facecolors='k', edgecolors='k')
        self.draw_c_colorbar(df)

    def draw_ll_contour(self, df, levels=-10, fs=8, draw_c_contour=False): 
        
        ## drawing contours
        # self.ax.tricontourf([self.grid[x[0],0] for x in df['(source, dest.)']],[self.grid[x[0],1] for x in df['(source, dest.)']], df['scaled log-lik'], cmap='Greens', extend='max', alpha=0.7, levels=levels, zorder=2) 
        # CS = self.ax.tricontour([self.grid[x[0],0] for x in df['(source, dest.)']],[self.grid[x[0],1] for x in df['(source, dest.)']], df['scaled log-lik'], cmap='Greens', extend='max', alpha=0.7, levels=levels, zorder=2)
        # self.ax.clabel(CS, inline=1, fontsize=fs, colors='k')
        # # drawing a star at the location of the MLE
        # self.ax.scatter(self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],0],self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],1], marker='*', zorder=2, facecolors='k', edgecolors='k')
        # self.draw_c_colorbar(df)

        ## drawing points
        permuted_idx = query_node_attributes(self.sp_graph, "permuted_idx")
        ll = plt.get_cmap('Greens_r').resampled(np.abs(levels)+2)
        
        # only display points that have scaled log-lik > -20
        for idx, row in df.loc[df['scaled log-lik']>=levels].iterrows():
            self.ax.scatter(self.grid[row['(source, dest.)'][0],0], self.grid[row['(source, dest.)'][0],1], marker='o', zorder=2, edgecolors='white', facecolors=ll(int(-row['scaled log-lik'])), linewidth=0.5, s=50)
        # for ix, x in enumerate(df['(source, dest.)']):
        #     self.ax.scatter(self.grid[x[0],0], self.grid[x[0],1], marker='o', zorder=2, edgecolors='white', facecolors=ll(int(-df['scaled log-lik'].iloc[ix])), linewidth=0.1, s=20)
            # self.ax.scatter(self.grid[permuted_idx[x[0]],0], self.grid[permuted_idx[x[0]],1], marker='o', zorder=2, edgecolors='white', facecolors=ll(int(-df['scaled log-lik'].iloc[ix])), linewidth=0.1, s=6)
        # self.ax.scatter(self.grid[df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()][0],0],self.grid[df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()][0],1], marker='*', s=100, zorder=2, alpha=0.8, facecolors='grey', edgecolors='k')

        ## histogram of c values across the landscape
        # inset_axes(self.ax, loc = 'upper center', width = '15%', height = '10%')
        # plt.hist(df['admix. prop.'], weights=1/np.sqrt(-2*df['scaled log-lik']+1), color='grey', bins=np.linspace(0,1,20), alpha=0.6); plt.xticks(np.around(np.linspace(0,1,4),1)) 
        # plt.axvline(df['admix. prop.'].iloc[df['scaled log-lik'].argmax()],color='k',linewidth=0.7)
        # plt.axvline(np.min(df['admix. prop.'].iloc[np.where(df['scaled log-lik']>-20)]),color='red',ls='--',linewidth=0.7)
        # plt.axvline(np.max(df['admix. prop.'].iloc[np.where(df['scaled log-lik']>-20)]),color='red',ls='--',linewidth=0.7)

        ## drawing an arrow from MLE source to destination
        self.draw_arrow([df['(source, dest.)'].iloc[df['log-lik'].argmax()]], df['admix. prop.'].iloc[df['log-lik'].argmax()])
        
        ## profile likelihood for c at MLE
        # lre = [(np.where(permuted_idx==df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()][0])[0][0],np.where(permuted_idx==df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()][1])[0][0])] #if np.where(permuted_idx==df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()][0])[0]<self.sp_graph.n_observed_nodes else [df['(source, dest.)'].iloc[df['scaled log-lik'].argmax()]]
        cgrid = np.linspace(0,1,30)
        # source = 'sampled' if lre[0][0]<self.sp_graph.n_observed_nodes else 'unsampled'
        cprofll = np.zeros(len(cgrid))
        for ic, c in enumerate(cgrid):
            try:
                cprofll[ic] = -self.obj.eems_neg_log_lik(c, {'edge':[df['(source, dest.)'].iloc[np.argmax(df['log-lik'])]], 'mode':'compute'})
            except:
                cprofll[ic] = np.nan

        cprofll2 = np.zeros((np.sum(df['scaled log-lik']>-2),len(cgrid)))
        for idx, ed in enumerate(df['(source, dest.)'].loc[df['scaled log-lik']>-2]):
            for ic, c in enumerate(cgrid):
                try:
                    cprofll2[idx,ic] = -self.obj.eems_neg_log_lik(c, {'edge':[ed], 'mode':'compute'})
                except:
                    cprofll2[idx,ic] = np.nan

        # inset_axes(self.ax, loc = "lower left", bbox_to_anchor=(0.15, 0.1, 1, 1), bbox_transform=self.ax.transAxes, width = '15%', height = '10%')
        inset_axes(self.ax, loc = "lower left", width = '15%', height = '10%')            
        # #TODO: take care of discretization here
        # #TODO: change font size here (maybe it's ok but labels seem a bit big)
        plt.plot(cgrid, cprofll, color='grey'); 
        plt.ylim((np.nanmax(cprofll)+levels, np.nanmax(cprofll)-levels/20)); #plt.axvline(df['admix. prop.'].iloc[df['scaled log-lik'].argmax()],color='k',linewidth=0.4); 
        plt.plot(cgrid, cprofll2.T, color='grey', alpha=0.5, linewidth=0.3)
        #TODO: change font size of MLE value to be 70% of 0 & 1 (may need two commands for this)
        plt.xticks(ticks=[0, cgrid[np.nanargmax(cprofll)], 1],labels=[0, round(cgrid[np.nanargmax(cprofll)], 2), 1]); 
        lb = np.where(cprofll >= np.nanmax(cprofll) - 5)[0][0]; ub = np.where(cprofll >= np.nanmax(cprofll) - 5)[0][-1]
        plt.axvline(cgrid[lb], color='red', ls='--', linewidth=0.5) 
        plt.axvline(cgrid[ub], color='red', ls='--', linewidth=0.5)
        
        ## drawing the colorbar for the log-lik surface
        self.c_axins = inset_axes(self.ax, loc = 'center right', width = "10%", height = "2%",)
        self.c_axins.set_title(r"scaled $\ell$", fontsize = self.cbar_font_size)
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=clr.Normalize(levels-1,0), cmap=ll.reversed()), boundaries=np.arange(levels-1,1), cax=self.c_axins, shrink=0.1, orientation='horizontal')
        self.c_cbar.set_ticks([levels,0])

def recover_nnz_entries(sp_graph):
    """Permute W matrix and vectorize according to the CSC index format"""
    W = sp_graph.inv_triu(sp_graph.w, perm=False)
    w = np.array([])
    idx = nx.adjacency_matrix(sp_graph).nonzero()
    idx = list(np.column_stack(idx))
    for i in range(len(idx)):
        w = np.append(w, W[idx[i][0], idx[i][1]])
    return w

def recover_nnz_entries_foldchange(sp_graph, oldweights):
    """Permuting the edge change matrix instead"""
    # norm_newweights = (newweights - np.mean(newweights))/np.std(newweights)
    # norm_weights = (sp_graph.w - np.mean(sp_graph.w))/np.std(sp_graph.w)
    W = sp_graph.inv_triu((sp_graph.w-oldweights)*100/oldweights, perm=False)
    w = np.array([])
    idx = nx.adjacency_matrix(sp_graph).nonzero()
    idx = list(np.column_stack(idx))
    for i in range(len(idx)):
        w = np.append(w, W[idx[i][0], idx[i][1]])
    return w

def project_coords(X, proj):
    """Project coordinates"""
    P = np.empty(X.shape)
    for i in range(X.shape[0]):
        x, y = proj(X[i, 0], X[i, 1])
        P[i, 0] = x
        P[i, 1] = y
    return P
