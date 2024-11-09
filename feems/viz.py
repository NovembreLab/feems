from __future__ import absolute_import, division, print_function

import cartopy.feature as cfeature
from copy import copy, deepcopy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as clr
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import ticker
import networkx as nx
import numpy as np
from scipy.linalg import pinvh
from pyproj import Proj
from statsmodels.api import add_constant, OLS

from .spatial_graph import query_node_attributes
from .objective import Objective
from .utils import benjamini_hochberg

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
        ell_scaler=np.sqrt(3.0) / 6.0,
        ell_edgecolor="gray",
        ell_lw=0.2,
        ell_abs_max=0.5,
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

    def draw_map(
        self, 
        latlong=True
    ):
        """Viz function to draw the underlying map projection.

        Optional:
            latlong (:obj:): 
                - True to draw gridlines picked from underlying graph (default) OR
                - False for no gridlines OR
                - tuple of ([lats], [longs]) coordinates to draw custom gridlines

        Returns: 
            None
        """
        
        np.seterr(invalid='ignore')
        self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        # self.ax.add_feature(cfeature.LAND, facecolor="#ffffff", zorder=0)
        self.ax.coastlines(
            self.coastline_m,
            color="#636363",
            linewidth=self.coastline_linewidth,
            zorder=0,
        )

        if latlong is not False:
            if latlong:
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
                # using an alternating grey and black color to prevent overplotting
                if perm_id%2:
                    self.ax.text(
                        obs_grid[i, 0],
                        obs_grid[i, 1],
                        str(perm_id),
                        horizontalalignment="center",
                        verticalalignment="center",
                        size=self.obs_node_textsize*1.05,
                        zorder=self.obs_node_zorder,
                        color='grey'
                    )
                else:    
                    self.ax.text(
                        obs_grid[i, 0],
                        obs_grid[i, 1],
                        str(perm_id),
                        horizontalalignment="center",
                        verticalalignment="center",
                        size=self.obs_node_textsize,
                        zorder=self.obs_node_zorder,
                        color='k'
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

    def draw_het(self):
        """Draws heterozygosity values across the map"""
        norm = plt.Normalize(np.min(1/self.sp_graph.q),np.max(1/self.sp_graph.q))
        for i in range(len(self.sp_graph)):
            if i < self.sp_graph.n_observed_nodes:
                self.ax.scatter(self.grid[self.sp_graph.perm_idx[i],0], self.grid[self.sp_graph.perm_idx[i],1], marker='o', zorder=2, edgecolors='#444444', facecolors=plt.get_cmap('Purples')(norm(1/self.sp_graph.q[i])), linewidth=2*self.obs_node_linewidth, s=2*self.obs_node_size)
            else:
                self.ax.scatter(self.grid[self.sp_graph.perm_idx[i],0], self.grid[self.sp_graph.perm_idx[i],1], marker='h', zorder=2, edgecolors='white', facecolors=plt.get_cmap('Purples')(norm(self.sp_graph.q_prox[i-self.sp_graph.n_observed_nodes])), linewidth=0.5*self.obs_node_linewidth, s=3*self.obs_node_size)

        self.c_axins = inset_axes(self.ax, 
                                  loc = 'upper right',
                                  width='15%',
                                  height='3%')
        self.c_axins.set_title(r"het. ($\propto N$)")
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Purples')), cax=self.c_axins, orientation='horizontal')
        
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
        self.edge_cbar.ax.set_title(r"${}^w/{}_\widebar{w}$", loc="center")
        self.edge_cbar.ax.set_title(
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size*1.5
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

    def draw_arrow(self, lre, c, hw=5, hl=8, tw=2):
        """Viz function to draw an arrow between two nodes on the graph & colored by an admixture proportion c between 0 and 1 (grey-scale, with 0 being white). Typically, an internal function, but can also be called externally.  
        Required:
            lre (:obj:`list of tuple`): [(source, destination)] ID as displayed by baseline FEEMS viz
            c (:obj:`float`): admixture proportion of long-range edge

        Optional:
            hw (:obj:`float`): head width
            hl (:obj:`float`): head length
            tw (:obj:`float`): tail width

        Returns:
            None
        """
        
        style = "Simple, tail_width={}, head_width={}, head_length={}".format(tw, hw, hl)
        kw = dict(arrowstyle=style, 
                  edgecolor='k', 
                  facecolor=self.c_cmap(c), 
                  zorder=5, 
                  linewidth=0.2*tw)

        arrow = patches.FancyArrowPatch((self.grid[lre[0][0],0],self.grid[lre[0][0],1]), (self.grid[lre[0][1],0],self.grid[lre[0][1],1]), connectionstyle="arc3,rad=-.3", **kw, mutation_scale=1)
        self.ax.add_patch(arrow)

    def draw_admixture_pies(
        self,
        qfile, 
        mode='demes',
        colors=["#e6ab02", "#a6761d", "#66a61e", "#7570b3", "#e7298a", "#1b9e77", "#d95f02", "#666666"],
        radius=0.2,
        edgecolor='black'
    ):
        """Reads in .Q matrix and plots the admixture pies across all sampled demes.
        """
        
        print("Reading in .Q file...")
        Q = np.loadtxt(qfile)
        if np.sum(self.sp_graph.n_samples_per_obs_node_permuted) != Q.shape[0]:
            print("The number of samples in FEEMS genotype matrix do not match the number of samples in .Q file")
            return 
        print("Number of individuals: {:d}, K = {:d}".format(Q.shape[0], Q.shape[1]))

        if Q.shape[1] > 8:
            print("Please provide a custom list of colors for K = {:d} (default color scheme only goes up to K = 8)".format(Q.shape[1]))
            return

        if mode == 'samples':
            for i in range(Q.shape[0]):
                self._draw_admix_pie(Q[i, :], 
                                     self.coord[i, 0], 
                                     self.coord[i, 1], 
                                     colors[:Q.shape[1]], 
                                     radius=radius,
                                     linewidth=radius*1.1,
                                     edgecolor=edgecolor,
                                     ax=self.ax)
        else:
            for i in self.sp_graph.perm_idx[:self.sp_graph.n_observed_nodes]:
                self._draw_admix_pie(np.mean(Q[self.sp_graph.nodes[i]['sample_idx'], :],axis=0).tolist(), 
                                     self.grid[i, 0], 
                                     self.grid[i, 1], 
                                     colors[:Q.shape[1]], 
                                     radius=radius, 
                                     linewidth=radius*1.1,
                                     edgecolor=edgecolor,
                                     ax=self.ax)

    def _draw_admix_pie(
        self,
        admix_fracs, 
        x, y, 
        colors,
        radius=.18, 
        inset_width=.5,
        inset_height=.5,
        loc=10,
        linewidth=.2,
        edgecolor="black",
        ax=None
    ):
        """Draws a single admixture pie on a axis
        """
        xy = (x, y)
        ax_i = inset_axes(ax, 
                          width=inset_width, 
                          height=inset_height, 
                          loc=loc, 
                          bbox_to_anchor=(x, y),
                          bbox_transform=ax.transData, 
                          borderpad=0)
        wedges, t = ax_i.pie(admix_fracs, 
                             colors=colors, 
                             center=xy, 
                             radius=radius, 
                             wedgeprops={"linewidth": linewidth, 
                                         "edgecolor": edgecolor})

    def draw_c_colorbar(
        self,
        c_cbar_loc = 'upper right',
        c_cbar_width = 10, 
        c_cbar_height = 3
    ):
        "Viz function to draw a simple colorbar from 0 to 1 scale for admixture proportion"
        self.c_axins = inset_axes(self.ax, loc=c_cbar_loc, width = str(c_cbar_width)+'%', height = str(c_cbar_height)+'%', borderpad=2)
        self.c_axins.set_title(r"$\hat{c}$", fontsize = self.cbar_font_size*1.2, loc='center')
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.c_cmap), cax=self.c_axins, shrink=0.1, orientation='horizontal', ticks=[0,0.5,1])
        self.c_cbar.ax.tick_params(labelsize = self.cbar_ticklabelsize*0.8)

    def draw_c_contour(
        self, 
        df, 
        levels=-5, 
        c_cbar_loc=None,
        c_cbar_width=None,
        c_cbar_height=None
    ):
        """Viz function to draw the log-likelihood contour for the source of a particular destination deme. 
        Required:
            df (:obj:`pandas.DataFrame`): DataFrame containing the output of sp_graph.calc_contour or sp_graph.calc_joint_contour
            
        Optional: 
            levels (:obj:`int`): value specifying the lower bound on the log-likelihood to include in the contour (with the maximum scaled to be 0)
            c_cbar_loc (:obj:`str`): location of the colorbar for the admixture proportion (e.g., 'upper right', 'upper center', etc.)
            c_cbar_height (:obj:`int`): height of the colorbar
            c_cbar_width (:obj:`int`): width of the colorbar

        Returns:
            None
        """

        if c_cbar_loc is None:
            c_cbar_loc = 'upper right'
        if c_cbar_width is None:
            c_cbar_width = 10
        if c_cbar_height is None:
            c_cbar_height = 2
        
        idx = np.where(df['scaled log-lik'] >= levels)
        
        if len(idx[0]) < 3:
            print("Not enough demes fall within the specified levels threshold to draw a contour, consider decreasing levels.")
            return 
            
        self.ax.tricontourf([self.grid[x[0],0] for x in df['(source, dest.)'].iloc[idx]],[self.grid[x[0],1] for x in df['(source, dest.)'].iloc[idx]], df['admix. prop.'].iloc[idx], cmap='Greys', vmin=0, vmax=1, alpha=0.7, extend='both', levels=np.linspace(0,1,21)); 
        CS = self.ax.tricontourf([self.grid[x[0],0] for x in df['(source, dest.)'].iloc[idx]], [self.grid[x[0],1] for x in df['(source, dest.)'].iloc[idx]], df['admix. prop.'].iloc[idx], cmap='Greys', vmin=0, vmax=1, alpha=0.7, extend='both', levels=np.linspace(0,1,21))
        self.ax.clabel(CS, inline=False, levels=np.linspace(0,1,21)[((np.linspace(0,1,21) >= np.min(df['admix. prop.'].iloc[idx])) & (np.linspace(0,1,21) <= np.max(df['admix. prop.'].iloc[idx])))], fontsize=self.obs_node_textsize, colors='k')
        
        # drawing a X at the location of the MLE
        self.ax.scatter(self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],0], self.grid[df.iloc[df['scaled log-lik'].argmax(),0][0],1], marker='x', zorder=3, facecolors='firebrick', s=2*self.obs_node_size)
        self.draw_c_colorbar(c_cbar_loc, c_cbar_width, c_cbar_height)

        # drawing a circle around the destination
        self.ax.scatter(self.grid[df['(source, dest.)'].iloc[0][1],0],self.grid[df['(source, dest.)'].iloc[0][1],1], marker='o', zorder=3, facecolors='dodgerblue', alpha=0.5, s=3*self.obs_node_size)

    def draw_outliers(
        self, 
        outliers_df,
        linewidth=None
    ):
        """Viz function to draw lines between outlier pairs of demes and highlight putative destination demes. 
        Required:
            outliers_df (:obj:`pandas.DataFrame`): DataFrame containing the output of sp_graph.extract_outliers

        Optional
            linewidth (:obj:`float`): thickness of line connecting the demes

        Returns:
            None
        """

        if linewidth is None:
            linewidth = 2*self.obs_node_linewidth
            
        for i in range(outliers_df.shape[0]): 
            self.ax.plot([self.grid[outliers_df['source'].iloc[i],0], self.grid[outliers_df['dest.'].iloc[i],0]],
                         [self.grid[outliers_df['source'].iloc[i],1], self.grid[outliers_df['dest.'].iloc[i],1]], 
                         linewidth=linewidth, color='grey')
        for dest in np.unique(outliers_df['dest.']):
            self.ax.plot(self.grid[dest, 0], self.grid[dest, 1], 'o', 
                         color='dodgerblue', markersize=10*np.log10(np.sum(outliers_df['dest.']==dest)+1), alpha=0.5)      
    
    def draw_loglik_contour(
        self, 
        df, 
        levels=-10, 
        magnifier=200, 
        draw_arrow=True, 
        loglik_node_size=None,
        cbar_font_size=None, 
        cbar_ticklabelsize=None, 
        profile_bbox_to_anchor=None,
        profile_c_loc=None, profile_c_height=None, profile_c_width=None,
        lbar_loc=None, lbar_height=None, lbar_width=None
    ): 
        """Viz function to draw the log-likelihood contour for the source of a particular destination deme. 
        Required:
            df (:obj:`pandas.DataFrame`): DataFrame containing the output of sp_graph.calc_contour or sp_graph.calc_joint_contour
            
        Optional: 
            levels (:obj:`int`): value specifying the lower bound on the log-likelihood to include in the contour (with the maximum scaled to be 0)
            magnifier (:obj:`int`): percentage scaler on the size of the arrow with 100 being a magnification of 1 (default: 200% or 2x)
            draw_arrow (:obj:`Bool`): flag on whether to draw an arrow from the MLE source or not 
            loglik_node_size (:obj:`float`): (=2.5*obs_node_size, inherits from baseline FEEMS viz)
            cbar_font_size (:obj:`float`): (inherits from baseline FEEMS viz)
            cbar_ticklabelsize (:obj:float): (inherits from baseline FEEMS viz)
            profile_c_loc (:obj:`str`): location of the plot for the profile likelihood at MLE (e.g., 'lower left', 'center left', etc.)
            lbar_loc (:obj:`str`): location of the colorbar for log-likelihood contour (e.g., 'center right', 'lower right', etc.)
            profile_c_width (:obj:`int`): width of plot of profile likelihood
            profile_c_height (:obj:`int`): height of plot of profile likelihood
            lbar_c_width (:obj:`int`): width of colorbar for log-likelihood
            lbar_c_height (:obj:`int`): height of colorbar for log-likelihood

        Returns:
            None
        """

        self.obj = Objective(self.sp_graph); self.obj.inv(); self.obj.grad(reg=False)
        self.obj.Linv_diag = self.obj._comp_diag_pinv()
        
        # code to set default values 
        if loglik_node_size is None:
            loglik_node_size = self.obs_node_size*2.5
        if cbar_font_size is None:
            cbar_font_size = self.cbar_font_size*0.8
        if cbar_ticklabelsize is None:
            cbar_ticklabelsize = self.cbar_ticklabelsize
        if profile_c_loc is None:
            profile_c_loc = 'lower left'
        if profile_c_width is None:
            profile_c_width = 15
        if profile_c_height is None:
            profile_c_height = 10
        if lbar_loc is None:
            lbar_loc = 'lower right'
        if lbar_width is None:
            lbar_width = 10
        if lbar_height is None:
            lbar_height = 2

        # convert to fraction for easy scaling
        mag = magnifier/100

        # creating colormap
        ll = plt.get_cmap('Greens_r', np.abs(levels)+2)
        
        # only display points that have scaled log-lik > levels
        for idx, row in df.loc[df['scaled log-lik']>=levels].iterrows():
            self.ax.scatter(self.grid[row['(source, dest.)'][0],0], self.grid[row['(source, dest.)'][0],1], 
                            marker='h', zorder=2, edgecolors='white', 
                            facecolors=ll(int(-row['scaled log-lik'])), 
                            linewidth=0.5*self.obs_node_linewidth, s=2*loglik_node_size)
            
        # drawing an arrow from MLE source to destination
        if draw_arrow:
            self.draw_arrow([df['(source, dest.)'].iloc[df['log-lik'].argmax()]], df['admix. prop.'].iloc[df['log-lik'].argmax()],
                            tw=mag*0.2*self.obs_node_size, 
                            hw=mag*0.4*self.obs_node_size, 
                            hl=mag*0.5*self.obs_node_size)
            
        # cgrid = np.linspace(0,1,30)
        if df['admix. prop.'].iloc[df['log-lik'].argmax()]>=0.25:
            if df['admix. prop.'].iloc[df['log-lik'].argmax()]<=0.75:
                cgrid = np.linspace(df['admix. prop.'].iloc[df['log-lik'].argmax()]-0.25, df['admix. prop.'].iloc[df['log-lik'].argmax()]+0.25, 20)
            else:
                cgrid = np.linspace(0.6, 1, 20)
        else:
            cgrid = np.linspace(0, 0.4, 20)

        cprofll = np.zeros(len(cgrid))
        for ic, c in enumerate(cgrid):
            try:
                cprofll[ic] = -self.obj.eems_neg_log_lik(c, {'edge':[df['(source, dest.)'].iloc[np.argmax(df['log-lik'])]], 'mode':'compute'})
            except:
                cprofll[ic] = np.nan
        
        cprofll2 = np.zeros((np.sum(df['scaled log-lik']>-3),len(cgrid)))
        for idx, ed in enumerate(df['(source, dest.)'].loc[df['scaled log-lik']>-3]):
            for ic, c in enumerate(cgrid):
                try:
                    cprofll2[idx,ic] = -self.obj.eems_neg_log_lik(c, {'edge':[ed], 'mode':'compute'})
                except:
                    cprofll2[idx,ic] = np.nan

        inset_axes(self.ax, 
                   loc = profile_c_loc, 
                   width = str(profile_c_width)+'%', 
                   height = str(profile_c_height)+'%')            

        plt.plot(cgrid, cprofll, color='grey')
        plt.ylim((np.nanmax(cprofll)+levels, np.nanmax(cprofll)-levels/20))
        plt.plot(cgrid, cprofll2.T, color='grey', alpha=0.5, linewidth=0.3)
        plt.xticks(ticks=[0, 1], labels=[0, 1], fontsize=cbar_ticklabelsize); plt.xlabel(r'$c$', labelpad=-6, fontsize=cbar_ticklabelsize)
        plt.yticks(fontsize=cbar_ticklabelsize)
        plt.title(r'profile $\ell$ at MLE', fontsize=1.2*cbar_font_size)
        plt.text(cgrid[np.nanargmax(cprofll)], -0.2, round(cgrid[np.nanargmax(cprofll)], 2), fontsize=0.8*cbar_ticklabelsize, ha='center', va='top', transform=plt.gca().transAxes)

        lb = np.where(cprofll >= np.nanmax(cprofll) - 2)[0][0]; ub = np.where(cprofll >= np.nanmax(cprofll) - 2)[0][-1]
        plt.axvline(cgrid[lb], color='red', ls='--', linewidth=self.obs_node_linewidth) 
        plt.axvline(cgrid[ub], color='red', ls='--', linewidth=self.obs_node_linewidth)
        # lb = np.where(cprofll >= np.nanmax(cprofll) - 5)[0][0]; ub = np.where(cprofll >= np.nanmax(cprofll) - 5)[0][-1]
        # plt.axvline(cgrid[lb], color='red', ls='dotted', linewidth=self.obs_node_linewidth, alpha=0.6) 
        # plt.axvline(cgrid[ub], color='red', ls='dotted', linewidth=self.obs_node_linewidth, alpha=0.6)
        
        # drawing the colorbar for the log-lik surface
        self.c_axins = inset_axes(self.ax, 
                                  loc = lbar_loc, 
                                  width = str(lbar_width)+'%', 
                                  height = str(lbar_height)+'%')
        self.c_axins.set_title(r"scaled $\ell$", fontsize = 1.2*cbar_font_size)
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=clr.Normalize(levels-1,0), cmap=ll.reversed()), boundaries=np.arange(levels-1,1), cax=self.c_axins, shrink=0.1, orientation='horizontal')
        self.c_cbar.set_ticks([levels,0]); self.c_cbar.ax.tick_params(labelsize=cbar_ticklabelsize)

def draw_FEEMSmix_fit(
    v,
    ind_results,
    levels=-10,
    demes=None,
    draw_c_contour=False,
    draw_edges_mle=False,
    magnifier=100,
    dpi=200,
    figsize=(4,10)
):
    """Wrapper function to plot the entire suite of fits from separately fitting each edge
    Required:
        v (:obj:`feems.Viz`): Viz object created previously 
        ind_results (:obj:`dict`): output from sp_graph.independent_fit(...)
        
    Optional:
        levels (:obj:`int`): value specifying the lower bound on the log-likelihood to include in the contour (with the maximum scaled to be 0)
        demes (:obj:`int` or `list`): number or list of edge indices to plot (for any single index, use e.g., [980])
        draw_c_contour (:obj:`Bool`): whether to include a contour of the admixture proportions
        draw_edges_mle (:obj:`Bool`): whether to draw edge weights at MLE
        magnifier (:obj:`int`): percentage scaler on the size of the arrow with 100 being a magnification of 1
        dpi (:obj:`int`): resolution of figure
        figsize (:obj:`tuple`): (width, height) of matplotlib plot 

    Returns: 
        None        
    """
    
    v.obj = Objective(v.sp_graph)

    mag = magnifier / 100
    
    fig = plt.figure(dpi=dpi, figsize=figsize)

    alldemes = [ind_results[i]['deme'] for i in range(1,len(ind_results))]

    gs = GridSpec(2, 1)
    # plot all the edges in a single figure 
    vall = copy(v)
    axs = fig.add_subplot(gs[0, 0], projection=vall.projection)
    vall.ax = axs; vall.draw_map()
    vall.draw_edges(use_weights=True)
    vall.draw_obs_nodes()
    for i in range(1, len(ind_results)):
        vall.draw_arrow([ind_results[i]['joint_contour_df']['(source, dest.)'].iloc[ind_results[i]['joint_contour_df']['log-lik'].argmax()]], ind_results[i]['joint_contour_df']['admix. prop.'].iloc[ind_results[i]['joint_contour_df']['log-lik'].argmax()],
                            tw=mag*0.2*v.obs_node_size, 
                            hw=mag*0.4*v.obs_node_size, 
                            hl=mag*0.5*v.obs_node_size)            
    
    if hasattr(demes, '__len__'):
        matches = [alldemes.index(i)+1 for i in demes]
    else:
        if demes is None:
            demes = len(ind_results)
            matches = np.arange(1, demes)
        else: 
            matches = np.arange(1, demes+1)

    ax_list = [axs]; cnt = 1
    for i, idx in enumerate(matches):
        df = ind_results[idx]['joint_contour_df'].combine_first(ind_results[idx]['contour_df'])
        df['scaled log-lik'] = df['log-lik']-np.nanmax(df['log-lik'])
        vnew = copy(v)
        if draw_c_contour:
            if draw_edges_mle:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*3, 1, i*3+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_contour(df, levels=levels)
                vnew.draw_obs_nodes(use_ids=False)
                
                vnewnew = copy(v)
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*3, 1, i*3+2, projection=vnew.projection)
                ax_list[-1].set_title('admix. prop. contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnewnew.ax = ax_list[-1]
                vnewnew.draw_map()
                vnewnew.draw_c_contour(df, levels=levels)
                vnewnew.edge_alpha=0.; vnewnew.draw_edges(use_weights=False)
                vnewnew.draw_obs_nodes(use_ids=False)

                vnewnewnew = copy(v)
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*3, 1, i*3+3, projection=vnew.projection)
                ax_list[-1].set_title('jointly estimated weights for deme {:d}'.format(ind_results[idx]['deme']))
                vnewnewnew.ax = ax_list[-1]
                vnewnewnew.draw_map()
                vnewnewnew.sp_graph._update_graph(ind_results[idx]['mle_w'], ind_results[idx]['mle_s2'])
                vnewnewnew.draw_edges(use_weights=True)
                vnewnewnew.draw_obs_nodes(use_ids=False)
                vnewnewnew.draw_arrow([ind_results[idx]['joint_contour_df']['(source, dest.)'].iloc[ind_results[idx]['joint_contour_df']['log-lik'].argmax()]], ind_results[idx]['joint_contour_df']['admix. prop.'].iloc[ind_results[idx]['joint_contour_df']['log-lik'].argmax()],
                            tw=mag*0.2*v.obs_node_size, 
                            hw=mag*0.4*v.obs_node_size, 
                            hl=mag*0.5*v.obs_node_size)
            else:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*2, 1, i*2+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_contour(df, levels=levels)
                vnew.draw_obs_nodes(use_ids=False)
                
                vnewnew = copy(v)
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*2, 1, i*2+2, projection=vnew.projection)
                ax_list[-1].set_title('admix. prop. contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnewnew.ax = ax_list[-1]
                vnewnew.draw_map()
                vnewnew.edge_alpha=0.; vnewnew.draw_edges(use_weights=False)
                vnewnew.draw_c_contour(df, levels=levels)
                vnewnew.draw_obs_nodes(use_ids=False)
        else:
            if draw_edges_mle:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches), 1, i+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_contour(df, levels=levels)
                vnew.draw_obs_nodes(use_ids=False)
                
                vnewnewnew = copy(v)
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*3, 1, i*3+3, projection=vnew.projection)
                ax_list[-1].set_title('jointly estimated weights for deme {:d}'.format(ind_results[idx]['deme']))
                vnewnewnew.ax = ax_list[-1]
                vnewnewnew.draw_map()
                vnewnewnew.sp_graph._update_graph(ind_results[idx]['mle_w'], ind_results[idx]['mle_s2'])
                vnewnewnew.draw_edges(use_weights=True)
                vnewnewnew.draw_obs_nodes(use_ids=False)
                vnewnewnew.draw_arrow([ind_results[idx]['joint_contour_df']['(source, dest.)'].iloc[ind_results[idx]['joint_contour_df']['log-lik'].argmax()]], ind_results[idx]['joint_contour_df']['admix. prop.'].iloc[ind_results[idx]['joint_contour_df']['log-lik'].argmax()],
                            tw=mag*0.2*v.obs_node_size, 
                            hw=mag*0.4*v.obs_node_size, 
                            hl=mag*0.5*v.obs_node_size)
            else:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches), 1, i+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood contour for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                # TODO is there a way to get limits without adding edges?
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_contour(df, levels=levels)
                vnew.draw_obs_nodes(use_ids=False)

def plot_FEEMSmix_result(
    diag_results, 
    dpi=200, 
    figsize=(6,3)
):
    """Wrapper function to plot the diagnostic fits from the results of the independent or sequential fits.
    Required:
        diag_results (dict): output from sp_graph.independent_fit or sp_graph.sequential_fit

    Optional: 
        dpi (int): resolution of figure
        figsize (tuple): (width, height) of matplotlib plot   
    """
    
    fig, axs = plt.subplots(1, len(diag_results), dpi=dpi, figsize=figsize, sharey=True, constrained_layout=True)
    
    X = add_constant(diag_results[0]['fit_dist'])
    mod = OLS(diag_results[0]['emp_dist'], X)
    res = mod.fit()
    muhat, betahat = res.params
    axs[0].scatter(diag_results[0]['fit_dist'], diag_results[0]['emp_dist'], marker=".", alpha=0.8, zorder=0, color="k", s=20)
    bh = benjamini_hochberg(diag_results[0]['emp_dist'], diag_results[0]['fit_dist'], fdr=diag_results[0]['fdr'])
    axs[0].scatter(diag_results[0]['fit_dist'][bh], diag_results[0]['emp_dist'][bh], marker='x', color='r', s=20, alpha=0.8)
    x_ = np.linspace(np.min(diag_results[0]['fit_dist']), np.max(diag_results[0]['fit_dist']), 12);
    axs[0].plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=2);
    axs[0].text(0.65, 0.2, "R²={:.3f}".format(res.rsquared), transform=axs[0].transAxes, size='large')
    axs[0].set_title('Baseline')
    fig.supxlabel('fitted distance'); fig.supylabel('genetic distance');
    
    for i in range(1, len(diag_results)):
        X = add_constant(diag_results[i]['fit_dist'])
        mod = OLS(diag_results[0]['emp_dist'], X)
        res = mod.fit()
        muhat, betahat = res.params
        axs[i].scatter(diag_results[i]['fit_dist'], diag_results[0]['emp_dist'], marker=".", alpha=0.8, zorder=0, color="k", s=20)
        axs[i].scatter(diag_results[i]['fit_dist'][bh], diag_results[0]['emp_dist'][bh], marker='.', color='r', s=20, alpha=0.8)
        axs[i].scatter(diag_results[i-1]['fit_dist'][bh], diag_results[0]['emp_dist'][bh], marker='x', color='r', s=20, alpha=0.8)
        for il in range(np.sum(bh)):
            axs[i].annotate(
                '',  # No text
                xy=(diag_results[i-1]['fit_dist'][bh][il], diag_results[0]['emp_dist'][bh][il]),  # Arrow end (head)
                xytext=(diag_results[i]['fit_dist'][bh][il], diag_results[0]['emp_dist'][bh][il]),  # Arrow start (tail)
                arrowprops=dict(
                    arrowstyle="<-",  # Arrow style
                    color='r',  # Color of the arrow
                    linewidth=1  # Width of the arrow line
                )
            )
        x_ = np.linspace(np.min(diag_results[i]['fit_dist']), np.max(diag_results[i]['fit_dist']), 12); 
        axs[i].plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=2); 
        axs[i].text(0.65, 0.2, "R²={:.3f}".format(res.rsquared), transform=axs[i].transAxes, size='large')
        axs[i].set_title('On fitting deme {:d}'.format(diag_results[i]['deme']))

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

def add_ax_subplot(n, ax, gs, fig, projection):
    # Expand the GridSpec layout
    gs = GridSpec(n + 1, 1)
    # Adjust the positions of existing subplots
    for i, axis in enumerate(ax):
        axis.set_position(gs[i].get_position(fig))
        axis.set_subplotspec(gs[i])
    # Add a new subplot
    new_ax = fig.add_subplot(gs[n, 0], projection=projection)
    ax.append(new_ax)
    return ax, gs