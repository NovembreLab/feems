"""The following helper functions are adapted from Ben Peters' code:
https://github.com/NovembreLab/eems-around-the-world/blob/master/subsetter/
"""

from __future__ import absolute_import, division, print_function

import fiona
import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Point, Polygon, shape


def load_tiles(s):
    tiles = fiona.collection(s)
    return [shape(t["geometry"]) for t in tiles]


def wrap_america(tile):
    tile = Point(tile)
    if np.max(tile.xy[0]) < -40 or np.min(tile.xy[0]) < -40:
        tile = translate(tile, xoff=360.0)
    return tile.xy[0][0], tile.xy[1][0]


def create_tile_dict(tiles, bpoly):
    pts = dict()  # dict saving ids
    rev_pts = dict()
    edges = set()
    pts_in = dict()  # dict saving which points are in region

    for c, poly in enumerate(tiles):
        x, y = poly.exterior.xy
        points = zip(np.round(x, 3), np.round(y, 3))
        points = [wrap_america(p) for p in points]
        for p in points:
            if p not in pts_in:
                # check if point is in region
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


def prepare_graph_inputs(coord, ggrid, translated, buffer=0, outer=None):
    """Prepares the graph input files for feems adapted from Ben Peters
    eems-around-the-world repo

    Args:
        sample_pos (:obj:`numpy.ndarray`): spatial positions for samples
        ggrid (:obj:`str`): path to global grid shape file
        transform (:obj:`bool`): to translate x coordinates
        buffer (:obj:`float`) buffer on the convex hull of sample pts
        outer (:obj:`numpy.ndarray`): q x 2 matrix of coordinates of outer
            polygon
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
    tiles3 = [t for t in tiles2 if bpoly.intersects(t) or bpoly2.intersects(t)]
    pts, rev_pts, e = create_tile_dict(tiles3, bpoly)

    # construct grid array
    grid = []
    for i, v in rev_pts.items():
        grid.append((v[0], v[1]))
    grid = np.array(grid)

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
