from __future__ import absolute_import, division, print_function

import cartopy.feature as cfeature
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Proj

from .spatial_graph import query_node_attributes


class Viz(object):
    def __init__(
        self,
        ax,
        sp_graph,
        weights=None, 
        newweights=None, halfrange=50,
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

        # extract node positions on the lattice
        self.idx = nx.adjacency_matrix(self.sp_graph).nonzero()

        # edge weights
        if weights is None:
            self.weights = recover_nnz_entries(sp_graph)
        else:
            self.weights = weights
        self.norm_log_weights = np.log10(self.weights) - np.mean(np.log10(self.weights))

        if newweights is not None:
            self.foldchange = recover_nnz_entries_foldchange(sp_graph, newweights)
        self.n_params = int(len(self.weights) / 2)

        # plotting maps
        if self.projection is not None:
            self.proj = Proj(projection.proj4_init)
            self.coord = project_coords(self.coord, self.proj)
            self.grid = project_coords(self.grid, self.proj)

    def draw_map(self):
        """Draws the underlying map projection"""
        self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        self.ax.coastlines(
            self.coastline_m,
            color="#636363",
            linewidth=self.coastline_linewidth,
            zorder=0,
        )

    def draw_samples(self):
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
                    edge_norm=self.change_norm,
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
        self.edge_cbar.ax.set_title(r"log10(w)", loc="center")
        self.edge_cbar.ax.set_title(
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size
        )
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)

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
        self.edge_cbar.ax.set_title(r"relative-change (%)", loc="center")
        self.edge_cbar.ax.set_title(
            self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size
        )
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)


def recover_nnz_entries(sp_graph):
    """Permute W matrix and vectorize according to the CSC index format"""
    W = sp_graph.inv_triu(sp_graph.w, perm=False)
    w = np.array([])
    idx = nx.adjacency_matrix(sp_graph).nonzero()
    idx = list(np.column_stack(idx))
    for i in range(len(idx)):
        w = np.append(w, W[idx[i][0], idx[i][1]])
    return w

def recover_nnz_entries_foldchange(sp_graph, newweights):
    """Permuting the edge change matrix instead"""
    # norm_newweights = (newweights - np.mean(newweights))/np.std(newweights)
    # norm_weights = (sp_graph.w - np.mean(sp_graph.w))/np.std(sp_graph.w)
    W = sp_graph.inv_triu((newweights-sp_graph.w)*100/sp_graph.w, perm=False)
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
