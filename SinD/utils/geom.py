import numpy as np
import torch
from scipy.spatial import Delaunay

def alpha_shape(points: np.ndarray, alpha: float, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.

    modified from: https://stackoverflow.com/a/50159452
    """
    assert points.shape[0] > 3, "Need at least four points"

    points = np.unique(points, axis=0)
    tri = Delaunay(points)
    triangles = []

    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        if circum_r < alpha:
            triangles.append(np.stack([pa, pb, pc]))

    return triangles

def calculate_rot_bboxes_and_triangle(centroids: torch.FloatTensor, length: torch.FloatTensor, width: torch.FloatTensor, rotation: torch.FloatTensor):
    """
    Calculate bounding box vertices and triangle vertices from centroid, width and length.

    Args:
    ---
    - centroids: center point of bbox [N, 2]
    - length: length of bbox [N]
    - width: width of bbox [N]
    - rotation: rotation of main bbox axis (along length) [N]
    
    Returns:
    ---
    - vertices of bbox: [N, 4, 2]
    - vertices of triangle: [N, 3, 2]
    """
    assert centroids.shape[-1] == 2

    # Preallocate
    batch_size = centroids.shape[0]
    rotated_bbox_vertices = torch.empty((batch_size, 4, 2), dtype=torch.float32)

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2
    
    th, r = cart2pol(rotated_bbox_vertices)
    rotated_bbox_vertices = pol2cart(th + rotation[:, *([None] * (th.ndim - 1))], r)
        
    rotated_bbox_vertices += centroids[:, None, :] #(n, 4, 2)

    # Calculate triangle vertices
    triangle_factor = 0.75

    triangles = torch.zeros((batch_size, 3, 2), dtype=torch.float32)

    triangles[:, 0, :] = rotated_bbox_vertices[:, 3, :] + (
                (rotated_bbox_vertices[:, 2, :] - rotated_bbox_vertices[:, 3, :]) * triangle_factor)
    triangles[:, 1, :] = rotated_bbox_vertices[:, 0, :] + (
                (rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 0, :]) * triangle_factor)
    triangles[:, 2, :] = rotated_bbox_vertices[:, 2, :] + ((rotated_bbox_vertices[:, 1, :] - rotated_bbox_vertices[:, 2, :]) * 0.5)

    return rotated_bbox_vertices, triangles


def cart2pol(cart: torch.FloatTensor):
    """
    Transform cartesian to polar coordinates.

    Args:
    ---
    - cart: [..., 2]
    
    Returns:
    ---
    - angle in radian: [...]
    - radius: [...]
    """
    x = cart[..., 0]
    y = cart[..., 1]

    theta = torch.arctan2(y, x)
    r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    return theta, r


def pol2cart(theta: torch.FloatTensor, r: torch.FloatTensor):
    """
    Transform polar to cartesian coordinates.

    Args:
    ---
    - theta: [...]
    - r: [...]
    
    Returns:
    ---
    - cartesian coordinates: [..., 2]
    """
    cart = torch.zeros((*theta.shape, 2), dtype=torch.float32)
    cart[..., 0] = r * np.cos(theta)
    cart[..., 1] = r * np.sin(theta)

    return cart