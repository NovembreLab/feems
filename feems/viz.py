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
from scipy.stats import linregress
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
        longlat=None
    ):
        """Viz function to draw the underlying map projection.

        Optional:
            longlat (:obj:): 
                - True to draw gridlines picked from underlying graph (default) OR
                - False for no gridlines OR
                - tuple of ([longs], [lats]) coordinates to draw custom gridlines

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

        if isinstance(longlat, tuple):
            if len(longlat) != 2:
                raise ValueError('Please specify valid option for longlat: True (default) or tuple/list of [[longs],[lats]] for custom gridlines. When specifying just a single coordinate list, leave an empty list for the other coordinate.')
            gl = self.ax.gridlines(xlocs=longlat[1], ylocs=longlat[0], draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, zorder=0)
            gl.top_labels=False; gl.right_labels=False
            gl.xlabel_style = {'rotation': 45}; gl.ylabel_style = {'rotation': 315}
        elif longlat:
            gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, zorder=0)
            gl.top_labels=False; gl.right_labels=False
            gl.xlabel_style = {'rotation': 45}; gl.ylabel_style = {'rotation': 315}

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
                        f'{perm_id}\u0332',
                        horizontalalignment="center",
                        verticalalignment="center",
                        size=self.obs_node_textsize,
                        zorder=self.obs_node_zorder,
                        color='#8D8F8F'
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
                        color='#4A6FA5'
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
                                  height='3%', borderpad=0)
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
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size*1.2
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

    def draw_arrow(self, lre, c, hw=5, hl=8, tw=2, mutation_scale=1, chiSq=20, ax=None):
        """Viz function to draw an arrow between two nodes on the graph & colored by an admixture proportion c between 0 and 1 (grey-scale, with 0 being white). Typically, an internal function, but can also be called externally.  
        
        Required:
            lre (:obj:`list of tuple`): [(source, destination)] ID as displayed by baseline FEEMS viz
            c (:obj:`float`): admixture proportion of LRE

        Optional:
            hw (:obj:`float`): head width
            hl (:obj:`float`): head length
            tw (:obj:`float`): tail width

        Returns:
            None
        """
        
        style = "Simple, tail_width={}, head_width={}, head_length={}".format(tw, hw, hl)
        if chiSq >= 11:
            kw = dict(arrowstyle=style, 
                      edgecolor='k', 
                      facecolor=self.c_cmap(c), 
                      zorder=3, 
                      linestyle='-',
                      linewidth=0.2*tw) 
        else:
            kw = dict(arrowstyle=style, 
                      edgecolor='k', 
                      facecolor=self.c_cmap(c), 
                      zorder=3, 
                      linewidth=0.2*tw,
                      linestyle=(5, (5, 5)))

        arrow = patches.FancyArrowPatch((self.grid[lre[0][0],0],self.grid[lre[0][0],1]), (self.grid[lre[0][1],0],self.grid[lre[0][1],1]), connectionstyle="arc3,rad=-.3", **kw, mutation_scale=mutation_scale, alpha=0.9)
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
        self.c_axins.set_title(r"$\hat{c}$", fontsize = self.cbar_font_size, loc='center')
        self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.c_cmap), cax=self.c_axins, shrink=0.1, orientation='horizontal', ticks=[0,0.5,1]); self.c_cbar.set_ticklabels([0,0.5,1])
        self.c_cbar.ax.tick_params(labelsize = self.cbar_ticklabelsize*0.8)

    def draw_c_surface(
        self, 
        df, 
        levels=-10, 
        c_cbar_loc=None,
        c_cbar_width=None,
        c_cbar_height=None
    ):
        """Viz function to draw the log-likelihood surface for the source fraction c of a particular destination deme. 
        Required:
            df (:obj:`pandas.DataFrame`): DataFrame containing the output of sp_graph.calc_surface or sp_graph.calc_joint_surface
            
        Optional: 
            levels (:obj:`int`): value specifying the lower bound on the log-likelihood to include in the surface (with the maximum scaled to be 0)
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
            print("Not enough demes fall within the specified levels threshold to draw a surface, consider decreasing levels.")
            return 
            
        for idx, row in df.loc[df['scaled log-lik']>=levels].iterrows():
            self.ax.scatter(self.grid[row['(source, dest.)'][0],0], self.grid[row['(source, dest.)'][0],1], 
                            c=row['scaled log-lik'],
                            marker='h', zorder=2, edgecolors='white', cmap='Greys', edgecolor='k',
                            # facecolors=ll(int(-row['scaled log-lik'])), 
                            linewidth=0.25*self.obs_node_linewidth, s=5*self.obs_node_size)
        
        # drawing a X at the location of the MLE
        self.draw_c_colorbar(c_cbar_loc, c_cbar_width, c_cbar_height)

        self.draw_arrow([df['(source, dest.)'].iloc[df['log-lik'].argmax()]], df['admix. prop.'].iloc[df['log-lik'].argmax()])

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

        softmin_stat = lambda group: np.log(-np.sum(group * np.exp(-group)))
        oser = outliers_df.groupby('dest.')['scaled diff.'].apply(softmin_stat)
        for i in range(outliers_df.shape[0]): 
            self.ax.plot([self.grid[outliers_df['source'].iloc[i],0], self.grid[outliers_df['dest.'].iloc[i],0]],
                         [self.grid[outliers_df['source'].iloc[i],1], self.grid[outliers_df['dest.'].iloc[i],1]], 
                         linewidth=linewidth, color='grey')
        for dest in np.unique(outliers_df['dest.']):
            # self.ax.plot(self.grid[dest, 0], self.grid[dest, 1], 'o', 
            #              color='dodgerblue', markersize=10*np.log10(np.sum(outliers_df['dest.']==dest)+self.obs_node_size), alpha=0.5)      
            self.ax.plot(self.grid[dest, 0], self.grid[dest, 1], 'o',
                         color='dodgerblue', markersize=3*self.obs_node_size**(oser.loc[dest]/oser.max()), alpha=0.5)
    
    def draw_loglik_surface(
        self, 
        df, 
        mutation_scale=2, 
        draw_arrow=True, 
        loglik_node_size=None,
        cbar_font_size=None, 
        cbar_ticklabelsize=None, 
        profile_bbox_to_anchor=None,
        profile_c_loc=None, profile_c_height=None, profile_c_width=None,
        lbar_loc=None, lbar_height=None, lbar_width=None
    ): 
        """Viz function to draw the log-likelihood surface for the source of a particular destination deme. 
        Required:
            df (:obj:`pandas.DataFrame`): DataFrame containing the output of sp_graph.calc_surface or sp_graph.calc_joint_surface
            
        Optional: 
            levels (:obj:`int`): value specifying the lower bound on the log-likelihood to include in the surface (with the maximum scaled to be 0)
            mutation_scale (:obj:`int`): scaler on the size of the arrow with 1 being a magnification of 1x (default: 2x)
            draw_arrow (:obj:`Bool`): flag on whether to draw an arrow from the MLE source or not 
            loglik_node_size (:obj:`float`): (=2.5*obs_node_size, inherits from baseline FEEMS viz)
            cbar_font_size (:obj:`float`): (inherits from baseline FEEMS viz)
            cbar_ticklabelsize (:obj:float): (inherits from baseline FEEMS viz)
            profile_c_loc (:obj:`str`): location of the plot for the profile likelihood at MLE (e.g., 'lower left', 'center left', etc.)
            lbar_loc (:obj:`str`): location of the colorbar for log-likelihood surface (e.g., 'center right', 'lower right', etc.)
            profile_c_width (:obj:`int`): width of plot of profile likelihood
            profile_c_height (:obj:`int`): height of plot of profile likelihood
            lbar_c_width (:obj:`int`): width of colorbar for log-likelihood
            lbar_c_height (:obj:`int`): height of colorbar for log-likelihood

        Returns:
            None
        """
        # TODO plot log-lik surface for a certain deme (passed in by user WITH seq_results) while accounting for all previous edges

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
            lbar_width = 15
        if lbar_height is None:
            lbar_height = 2

        # creating colormap
        # ll = plt.get_cmap('Greens_r', np.abs(levels)+2)

        bounds = np.array([-200, -50, -10, -2, 0])

        # colors = plt.cm.Greens(np.linspace(0, 1, len(bounds)))
        colors = ['#cccccc', 'white', '#bae4b3', '#74c476', 'darkgreen']
        custom_cmap = clr.ListedColormap(colors)
        norm = clr.BoundaryNorm(bounds, custom_cmap.N, extend='min')
        
        # for idx, row in df.loc[df['scaled log-lik']>=np.nanmin(df['scaled log-lik'])].iterrows():
        for idx, row in df.loc[df['scaled log-lik']>=-200].iterrows():
            self.ax.scatter(self.grid[row['(source, dest.)'][0],0], self.grid[row['(source, dest.)'][0],1], 
                            c=row['scaled log-lik'],
                            marker='h', zorder=2, edgecolors='white', cmap=custom_cmap, norm=norm, edgecolor='grey',
                            # facecolors=ll(int(-row['scaled log-lik'])), 
                            linewidth=0.25*self.obs_node_linewidth, s=3*loglik_node_size)
            
        # drawing an arrow from MLE source to destination
        if draw_arrow:
            self.draw_arrow([df['(source, dest.)'].iloc[df['log-lik'].argmax()]], df['admix. prop.'].iloc[df['log-lik'].argmax()],
                            mutation_scale=mutation_scale)
            
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
                cprofll[ic] = -self.obj.eems_neg_log_lik([c], {'edge':[df['(source, dest.)'].iloc[np.argmax(df['log-lik'])]], 'mode':'compute'})
            except:
                cprofll[ic] = np.nan
        
        cprofll2 = np.zeros((np.sum(df['scaled log-lik']>-2),len(cgrid)))
        for idx, ed in enumerate(df['(source, dest.)'].loc[df['scaled log-lik']>-2]):
            for ic, c in enumerate(cgrid):
                try:
                    cprofll2[idx,ic] = -self.obj.eems_neg_log_lik([c], {'edge':[ed], 'mode':'compute'})
                except:
                    cprofll2[idx,ic] = np.nan
        
        inset_axes(self.ax, 
                   loc = profile_c_loc, 
                   width = str(profile_c_width)+'%', 
                   height = str(profile_c_height)+'%', borderpad=2)            

        plt.plot(cgrid, cprofll-np.nanmax(cprofll), color='grey')
        # plt.ylim((np.nanmax(cprofll)+levels, np.nanmax(cprofll)-levels/20))
        plt.ylim((-10, 1))
        plt.plot(cgrid, cprofll2.T - np.nanmax(cprofll), color='grey', alpha=0.5, linewidth=0.3)
        plt.xticks(ticks=[0, 1], labels=[0, 1], fontsize=cbar_ticklabelsize); #plt.xlabel(r'$c$', labelpad=-6, fontsize=cbar_ticklabelsize)
        plt.yticks(fontsize=cbar_ticklabelsize)
        plt.title(r'profile $\ell$ for $c$', fontsize=1.2*cbar_font_size)
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
                                  height = str(lbar_height)+'%', borderpad=2)
        self.c_axins.set_title(r"scaled $\ell$", fontsize = 1.1*cbar_font_size)
        # self.c_cbar = plt.colorbar(plt.cm.ScalarMappable(norm=clr.Normalize(levels-1,0), cmap=ll.reversed()), boundaries=np.arange(levels-1,1), cax=self.c_axins, shrink=0.1, orientation='horizontal'); self.c_cbar.set_ticks([levels,0]);
        self.c_cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), cax=self.c_axins, shrink=0.1, orientation='horizontal', ticks=bounds, extendfrac=0.2);
        self.c_cbar.set_ticklabels(ticklabels=['≤-200','-50','-10','-2','0'])
        # self.c_axins.set_xscale('function', functions=(lambda x: -np.log1p(np.abs(x)), lambda x: np.exp(-x) - 1))
        self.c_cbar.ax.tick_params(labelsize=0.8*cbar_ticklabelsize, rotation=45, pad=3)

    def draw_LREs(
        self,
        seq_results,
        exclude=None,
        magnifier=1,
    ):
        """Viz function to draw the LREs from the fitted object.
        Required:
            seq_results (:obj:`dict`): dictionary containing the output of sp_graph.sequential_fit(...)
            
        Optional: 
            exclude (:obj:`list`): list of indices indicating which LREs to exclude from final fit (1-index based, default is None)
            magnifier (:obj:`float`): magnifier on the size of the arrows of the LREs

        Returns:
            None
        """

        idx = range(1, len(seq_results))
        if exclude is not None:
            idx = set(range(1,len(seq_results)))-set(exclude)
            alldemes = [seq_results[i]['deme'] for i in set(range(1,len(seq_results)))-set(exclude)]
        else:
            alldemes = [seq_results[i]['deme'] for i in range(1,len(seq_results))]

        mut_scale = magnifier*np.logspace(-0.5,0.5,len(seq_results))[::-1]

        for i in list(idx)[::-1]: 
            self.draw_arrow([seq_results[i]['joint_surface_df']['(source, dest.)'].iloc[seq_results[i]['joint_surface_df']['log-lik'].argmax()]], 
                            seq_results[i]['joint_surface_df']['admix. prop.'].iloc[seq_results[i]['joint_surface_df']['log-lik'].argmax()],
                            mutation_scale=mut_scale[i-1],
                            chiSq=seq_results[i-1]['chiSq'])  
    

def draw_FEEMSmix_surface(
    v,
    ind_results,
    demes=None,
    draw_arrow=True,
    draw_c_surface=False,
    magnifier=1,
    dpi=300,
    figsize=(4,10)
):
    """Wrapper function to plot the entire suite of fits from separately fitting each edge
    Required:
        v (:obj:`feems.Viz`): Viz object created previously 
        ind_results (:obj:`dict`): output from sp_graph.sequential_fit(...)
        
    Optional:
        demes (:obj:`int` or `list`): number or list of edge indices to plot (for any single index, use e.g., [980])
        draw_c_surface (:obj:`Bool`): whether to include a surface of the admixture proportions
        draw_arrow (:obj:`Bool`): whether to draw LREs as arrows
        magnifier (:obj:`float`): scaler on the size of the arrows 
        dpi (:obj:`int`): resolution of figure
        figsize (:obj:`tuple`): (width, height) of matplotlib plot 

    Returns: 
        None        
    """
    
    v.obj = Objective(v.sp_graph)
    
    fig = plt.figure(dpi=dpi, figsize=figsize)

    alldemes = [ind_results[i]['deme'] for i in range(1,len(ind_results))]

    gs = GridSpec(2, 1)
    # plot all the edges in a single figure 
    vall = copy(v)
    axs = fig.add_subplot(gs[0, 0], projection=vall.projection)
    vall.ax = axs; vall.draw_map()
    vall.draw_edges(use_weights=True)
    vall.draw_obs_nodes()

    mut_scale = magnifier*np.logspace(-0.5,0.5,len(ind_results))[::-1]
    for i in range(1, len(ind_results)):
        vall.draw_arrow([ind_results[i]['joint_surface_df']['(source, dest.)'].iloc[ind_results[i]['joint_surface_df']['log-lik'].argmax()]], ind_results[i]['joint_surface_df']['admix. prop.'].iloc[ind_results[i]['joint_surface_df']['log-lik'].argmax()],
                       mutation_scale=mut_scale[i-1])   
    
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
        df = ind_results[idx]['joint_surface_df'].combine_first(ind_results[idx]['surface_df'])
        df['scaled log-lik'] = df['log-lik']-np.nanmax(df['log-lik'])
        vnew = copy(v)
        if draw_c_surface:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*2, 1, i*2+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood surface for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_surface(df, draw_arrow=draw_arrow, loglik_node_size=v.obs_node_size)
                vnew.draw_obs_nodes(use_ids=False)
                
                vnewnew = copy(v)
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches)*2, 1, i*2+2, projection=vnew.projection)
                ax_list[-1].set_title('admix. prop. surface for deme {:d}'.format(ind_results[idx]['deme']))
                vnewnew.ax = ax_list[-1]
                vnewnew.draw_map()
                vnewnew.edge_alpha=0.; vnewnew.draw_edges(use_weights=False)
                vnewnew.draw_c_surface(df, levels=levels)
                vnewnew.draw_obs_nodes(use_ids=False)
        else:
                ax_list, gs = add_ax_subplot(cnt, ax_list, gs, fig, v.projection); cnt += 1
                # ax = fig.add_subplot(len(matches), 1, i+1, projection=vnew.projection)
                ax_list[-1].set_title('log-likelihood surface for deme {:d}'.format(ind_results[idx]['deme']))
                vnew.ax = ax_list[-1]
                vnew.draw_map() 
                vnew.edge_alpha=0.; vnew.draw_edges(use_weights=False)
                vnew.draw_loglik_surface(df, draw_arrow=draw_arrow, loglik_node_size=v.obs_node_size)
                vnew.draw_obs_nodes(use_ids=False)

def plot_FEEMSmix_summary(
    diag_results, 
    sequential,
    dpi=200, 
    figsize=(6,3),
    inset_frac=0.4,
):
    """Wrapper function to plot the diagnostic fits from the results of the independent or sequential fits.
    Required:
        diag_results (dict): output from sp_graph.sequential_fit(...) or sp_graph.independent_fit(...)
        sequential (bool): flag to indicate whether results from from a sequential_fit or not

    Optional: 
        dpi (int): resolution of figure
        figsize (tuple): (width, height) of matplotlib plot
        inset_frac (float): size of inset plot as a fraction of larger plot (default: 0.4)
    """

    if sequential: 
        fig, axs = plt.subplots(1, 2, dpi=dpi, figsize=figsize, constrained_layout=True)
        axs[0].plot(range(len(diag_results)),[linregress(diag_results[i]['fit_dist'],diag_results[0]['emp_dist'])[2]**2 for i in range(len(diag_results))], '-o', color='royalblue')
        axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs[0].set_ylabel(r'$R^2$'); axs[0].set_xlabel('# of added edges (deme ID)') 
        axs[0].set_xticks(range(len(diag_results)),['baseline']+['{:d} ('.format(i)+str(diag_results[i]['deme'])+')' for i in range(1,len(diag_results))],rotation=45); axs[0].grid()
    
        emp_dist = diag_results[0]['emp_dist']; fit_dist = diag_results[0]['fit_dist']
        logratio = (np.log(emp_dist/fit_dist) - np.mean(np.log(emp_dist/fit_dist)))/np.std(np.log(emp_dist/fit_dist))
        # bh = [np.where(np.round(logratio,5)==np.round(diag_results[0]['outliers_df']['scaled diff.'].iloc[i],5))[0][0] for i in range(len(diag_results[0]['outliers_df']))]
        bh = np.argsort(logratio)[:len(diag_results[0]['outliers_df'])//2]
        
        X = add_constant(diag_results[len(diag_results)-1]['fit_dist'])
        mod = OLS(emp_dist, X)
        res = mod.fit()
        muhat, betahat = res.params
        
        x_range = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
        y_range = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]
        
        # Inset dimensions based on data range
        inset_width = inset_frac * x_range / y_range  # Adjust size relative to aspect ratio
        inset_height = inset_frac
        inset_x = 1 - inset_width - 0.025  # Adjust margins as needed
        inset_y = 0.025
        
        # Generalized R² position: top-left of the plot
        r2_x = 0.1  
        r2_y = 0.85
        
        axs[1].plot(diag_results[len(diag_results)-1]['fit_dist'], emp_dist, '.k', alpha=0.7, markersize=6); 
        axs[1].axline((np.min(fit_dist),np.min(fit_dist)*betahat+muhat), slope=betahat, color='orange', ls='--', lw=2.5)
        axs[1].yaxis.set_major_locator(plt.MaxNLocator(3)); axs[1].xaxis.set_major_locator(plt.MaxNLocator(3))
        axs[1].plot(diag_results[len(diag_results)-1]['fit_dist'][bh], emp_dist[bh], '.', color='r', markersize=8, alpha=0.5)
        axs[1].set_ylabel('Genetic distance'); axs[1].set_xlabel(r'Fitted distance');
        # axs.text(0.2, 1., "R²={:.3f}".format(res.rsquared), fontsize=12); 
        axs[1].text(r2_x, r2_y, f"R²={res.rsquared:.3f}", fontsize=10, transform=axs[1].transAxes); 
        
        axins = axs[1].inset_axes([inset_x, inset_y, inset_width, inset_height])
        # axins = axs.inset_axes([0.5, 0.03, 0.45, 0.4])
        X = add_constant(fit_dist)
        mod = OLS(emp_dist, X)
        res = mod.fit()
        muhat, betahat = res.params
        axins.plot(fit_dist, emp_dist, '.k', alpha=0.5, markersize=6*inset_frac); 
        axins.axline((np.min(fit_dist),np.min(fit_dist)*betahat+muhat), slope=betahat, color='orange', ls='--', lw=3*inset_frac)
        axins.plot(fit_dist[bh], emp_dist[bh], '.', color='r', markersize=8*inset_frac, alpha=0.5); 
        axins.yaxis.set_major_locator(plt.MaxNLocator(2)); axins.xaxis.set_major_locator(plt.MaxNLocator(2))
        # axins.text(0.2, 1, "R²={:.3f}".format(res.rsquared), fontsize=10); 
        axins.text(r2_x-0.05, r2_y-0.1, f"R²={res.rsquared:.3f}", fontsize=7, transform=axins.transAxes)
        axins.set_xticklabels([]); axins.tick_params(axis='y',direction='in');
        axins.set_yticklabels([]); axins.tick_params(axis='x',direction='in');
    else:
        cols = min(len(diag_results), 5)  # Keep it column-heavy, max 4 columns
        rows = (len(diag_results) + cols - 1) // cols  # Compute required rows
        
        # Create figure and gridspec
        fig = plt.figure(figsize=(cols * 3.5, (rows + 1) * 3.5))  # Adjust figure size
        spec = GridSpec(rows + 1, cols, figure=fig, height_ratios=[1.5] + [1] * rows)

        # Compute start and end columns for centering the summary plot
        summary_col_start = max(0, cols // 4)
        summary_col_end = min(cols, summary_col_start + cols // 2)
        
        # Create the summary plot (spans full width)
        ax_summary = fig.add_subplot(spec[0, summary_col_start:summary_col_end])
        ax_summary.plot(range(len(diag_results)),[linregress(diag_results[i]['fit_dist'],diag_results[0]['emp_dist'])[2]**2 for i in range(len(diag_results))], '-o', color='royalblue')
        ax_summary.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_summary.set_ylabel(r'$R^2$'); ax_summary.set_xlabel('edge to deme ID') 
        ax_summary.set_xticks(range(len(diag_results)),['baseline']+[str(diag_results[i]['deme']) for i in range(1,len(diag_results))],rotation=45); ax_summary.grid()
        ax_summary.axhline(linregress(diag_results[0]['fit_dist'],diag_results[0]['emp_dist'])[2]**2, color='k', linestyle='--')

        # Create individual fit plots
        axs = []
        for i in range(len(diag_results)):
            row = (i // cols) + 1  # Start from row index 1 (row 0 is for summary)
            col = i % cols
            ax = fig.add_subplot(spec[row, col], sharex=axs[0] if axs else None, sharey=axs[0] if axs else None)
            axs.append(ax)

        # Adjust layout for better spacing
        plt.subplots_adjust(hspace=0.5)  # Increased vertical space

        emp_dist = diag_results[0]['emp_dist']; fit_dist = diag_results[0]['fit_dist']
        logratio = (np.log(emp_dist/fit_dist) - np.mean(np.log(emp_dist/fit_dist)))/np.std(np.log(emp_dist/fit_dist))
        bh = np.argsort(logratio)[:len(diag_results[0]['outliers_df'])]
        X = add_constant(fit_dist)
        mod = OLS(emp_dist, X)
        res = mod.fit()
        muhat, betahat = res.params
        
        axs[0].scatter(diag_results[0]['fit_dist'], diag_results[0]['emp_dist'], marker=".", alpha=0.8, zorder=0, color="k", s=20)
        
        x_ = np.linspace(np.min(diag_results[0]['fit_dist']), np.max(diag_results[0]['fit_dist']), 12);
        axs[0].plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=2);
        axs[0].yaxis.set_major_locator(plt.MaxNLocator(3)); axs[0].xaxis.set_major_locator(plt.MaxNLocator(3))
        axs[0].scatter(diag_results[0]['fit_dist'][bh], diag_results[0]['emp_dist'][bh], marker='o', color='r', s=20, alpha=0.5)
        axs[0].text(0.6, 0.2, "R²={:.3f}".format(res.rsquared), transform=axs[0].transAxes, size='large')
        axs[0].set_title('Baseline\nLL = {:.1f}'.format(diag_results[0]['log-lik']))
        axs[1].set_ylabel('Genetic distance'); fig.supxlabel('Fitted distance');
        
        for i in range(1, len(diag_results)):
            X = add_constant(diag_results[i]['fit_dist'])
            mod = OLS(diag_results[0]['emp_dist'], X)
            res = mod.fit()
            muhat, betahat = res.params
            # bh = np.where(np.abs(diag_results[i]['fit_dist'] - diag_results[i-1]['fit_dist']) >= 0.21 * (np.max(diag_results[0]['fit_dist'])-np.min(diag_results[0]['fit_dist'])))[0]
            axs[i].scatter(diag_results[i]['fit_dist'], diag_results[0]['emp_dist'], marker=".", alpha=0.8, zorder=0, color="k", s=20)
            axs[i].scatter(diag_results[i]['fit_dist'][bh], diag_results[0]['emp_dist'][bh], marker='o', color='r', s=20, alpha=0.5)
            x_ = np.linspace(np.min(diag_results[i]['fit_dist']), np.max(diag_results[i]['fit_dist']), 12); 
            axs[i].yaxis.set_major_locator(plt.MaxNLocator(3)); axs[i].xaxis.set_major_locator(plt.MaxNLocator(3))
            axs[i].plot(x_, muhat + betahat * x_, zorder=2, color="orange", linestyle='--', linewidth=2); 
            axs[i].text(0.6, 0.2, "R²={:.3f}".format(res.rsquared), transform=axs[i].transAxes, size='large')
            axs[i].set_title('LRE to {:d}\nLL = {:.1f}'.format(diag_results[i]['deme'],diag_results[i]['log-lik']))

        
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