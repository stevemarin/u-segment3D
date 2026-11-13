# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:35:23 2023

@author: fyz11

Gradient-descent ("inverse watershed") instance segmentation.

Every foreground pixel/voxel is seeded with a point. Points follow the gradient field
(e.g. combined 2D-to-3D Cellpose flows) and condense at cell centres. The cells are then
the connected components of the smoothed density of the final point positions, and each
original pixel/voxel gets the label of the cluster its point ended in.
"""

import os
import time

import cc3d
import multiprocess as mp
import numpy as np
import skimage.measure as skmeasure
import skimage.segmentation as sksegmentation
from multiprocess.pool import ThreadPool
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from tiler import Tiler
from tqdm import tqdm


def _mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return []


def _normalize99(Y, lower=0.01, upper=99.99):
    """normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile
    Upper and lower percentile ranges configurable.

    Parameters
    ----------
    Y: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN.
    upper: float
        upper percentile above which pixels are sent to 1.0

    lower: float
        lower percentile below which pixels are sent to 0.0

    Returns
    --------------
    normalized array with a minimum of 0 and maximum of 1

    """
    X = Y.copy()

    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))


def _interp2(query_pts, grid_shape, I_ref, method="linear", cast_uint8=False):
    spl = RegularGridInterpolator(
        (np.arange(grid_shape[0]), np.arange(grid_shape[1])),
        I_ref,
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    I_query = spl((query_pts[..., 0], query_pts[..., 1]))

    if cast_uint8:
        I_query = np.uint8(I_query)

    return I_query


def _interp3(query_pts, grid_shape, I_ref, method="linear", cast_uint8=False):
    spl_3 = RegularGridInterpolator(
        (np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2])),
        I_ref,
        method=method,
        bounds_error=False,
        fill_value=0,
    )

    I_query = spl_3((query_pts[..., 0], query_pts[..., 1], query_pts[..., 2]))
    if cast_uint8:
        I_query = np.uint8(I_query)

    return I_query


# =============================================================================
# Clustering the final point positions
# =============================================================================
def _point_density_foreground(pts, shape, smooth_sigma, thresh_factor, mask):
    """Smoothed density of the (integer-truncated) point positions, and its thresholded foreground.

    The threshold is mean + thresh_factor * std of the density (over ``mask`` if given),
    or just the mean if thresh_factor is None.
    """
    votes_grid_acc = np.zeros(shape)

    # one vote per occupied pixel (fancy-index += counts duplicate positions once).
    pts_index = tuple((pts[:, dim]).astype(np.int32) for dim in range(len(shape)))
    votes_grid_acc[pts_index] += 1.0

    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)

    if thresh_factor is not None:
        if mask is not None:
            thresh = np.mean(votes_grid_acc[mask]) + thresh_factor * np.std(votes_grid_acc[mask])
        else:
            thresh = np.mean(votes_grid_acc) + thresh_factor * np.std(votes_grid_acc)
    else:
        thresh = np.mean(votes_grid_acc)

    return votes_grid_acc, votes_grid_acc > thresh


def _label_original_positions(cell_seg_connected, pts, pts0, mask):
    """Give each starting position pts0 the cluster label at its final position pts."""
    ndim = cell_seg_connected.ndim
    pts_index = tuple((pts[:, dim]).astype(np.int32) for dim in range(ndim))
    pts0_index = tuple((pts0[:, dim]).astype(np.int32) for dim in range(ndim))

    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[pts0_index] = cell_seg_connected[pts_index]

    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        cell_seg_connected_original[mask == 0] = 0  # also mask the predicted.

    return cell_seg_connected_original


def connected_components_pts_2D(
    pts, pts0, shape, smooth_sigma=1, thresh_factor=None, mask=None, min_area=1
):
    """Label 2D points by the connected components of their smoothed final-position density.

    Returns
    -------
    cell_seg_connected_original : labels at the starting positions pts0
    cell_seg_connected : labels of the clusters at the final positions pts
    votes_grid_acc : the point density
    """
    votes_grid_acc, votes_grid_binary = _point_density_foreground(
        pts, shape, smooth_sigma, thresh_factor, mask
    )

    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=1)

    # remove clusters with area <= min_area
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected), 0)
    if len(cell_uniq_regions) > 0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas <= min_area]

        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected == invalid] = 0

    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]

    cell_seg_connected_original = _label_original_positions(cell_seg_connected, pts, pts0, mask)

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc


def connected_components_pts_3D(
    pts, pts0, shape, smooth_sigma=1, thresh_factor=None, mask=None, min_area=1
):
    """Label 3D points by the connected components of their smoothed final-position density.

    Note: min_area is not applied in 3D.

    Returns
    -------
    cell_seg_connected_original : labels at the starting positions pts0
    cell_seg_connected : labels of the clusters at the final positions pts
    votes_grid_acc : the point density
    """
    votes_grid_acc, votes_grid_binary = _point_density_foreground(
        pts, shape, smooth_sigma, thresh_factor, mask
    )

    cell_seg_connected = cc3d.connected_components(votes_grid_binary)

    cell_seg_connected_original = _label_original_positions(cell_seg_connected, pts, pts0, mask)

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc


# =============================================================================
# Gradient fields
# =============================================================================
def _sdf_distance_transform(binary, rev_sign=True):
    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)

    res = (
        ndimage.distance_transform_edt(neg_binary) * neg_binary
        - (ndimage.distance_transform_edt(pos_binary) - 1) * pos_binary
    )

    if rev_sign:
        res = res * -1

    return res


def surf_normal_sdf(binary, return_sdf=True, smooth_gradient=None, eps=1e-12, norm_vectors=True):
    """Surface normals (gradient) of the signed distance function of a binary, pointing outwards."""
    sdf_vol = _sdf_distance_transform(binary, rev_sign=True)

    sdf_vol_normal = np.array(np.gradient(sdf_vol))
    # smoothing needs to be done before normalization of magnitude.
    if smooth_gradient is not None:
        sdf_vol_normal = np.array(
            [ndimage.gaussian_filter(sdf, sigma=smooth_gradient) for sdf in sdf_vol_normal]
        )

    if norm_vectors:
        sdf_vol_normal = sdf_vol_normal / (np.linalg.norm(sdf_vol_normal, axis=0)[None, :] + eps)

    return sdf_vol_normal, sdf_vol


def _divergence(f):
    """Divergence of the vector field f (one array per dimension): dFx/dx + dFy/dy + ..."""
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def mean_curvature_sdf(sdf_normal):
    """Mean curvature, i.e. half the divergence of the surface normal field."""
    return 0.5 * (_divergence(sdf_normal))


def _initial_gradients_3D(binary, gradient_img, binary_mask_gradient, smooth_gradient, eps):
    """(3,M,N,L) gradients to follow: gradient_img if given, else the binary's SDF normals."""
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3, 0, 1, 2)  # use the supplied gradients!
    else:
        sdf_normals = surf_normal_sdf(
            binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True
        )[0]

    if binary_mask_gradient:
        sdf_normals = sdf_normals * binary[None, ...]

    return sdf_normals


def _curvature_rescale_3D(sdf_normals):
    """Scale gradients by the normalized mean curvature; helps prevent breaking up tube-like shapes."""
    curvature_3D = mean_curvature_sdf(sdf_normals)
    curvature_3D = _normalize99(curvature_3D, lower=0.01, upper=99)  # rescales to between 0-1
    return sdf_normals * curvature_3D[None, ...]


def _neighbour_averaging_matrix(binary):
    """Sparse matrix that averages a per-voxel quantity over each foreground voxel's neighbours."""
    import scipy.sparse as spsparse
    from sklearn.feature_extraction.image import grid_to_graph

    W = grid_to_graph(binary.shape[0], binary.shape[1], binary.shape[2], mask=binary > 0)

    DD = 1.0 / (W.sum(axis=-1))
    DD = spsparse.spdiags(np.squeeze(DD), [0], DD.shape[0], DD.shape[0])
    return DD.dot(W)


def _advect_points_3D(
    binary,
    sdf_normals,
    pts,
    n_iter,
    delta,
    momenta,
    gradient_decay,
    interp,
    renorm_gradient,
    use_connectivity,
    connectivity_alpha,
    progress,
):
    """Move points along sdf_normals with momentum; yields (iteration, positions) every step."""
    if use_connectivity:
        W = _neighbour_averaging_matrix(binary)
        alpha = connectivity_alpha

    pt_ii = pts.copy()
    mu = momenta
    g0 = np.zeros_like(pt_ii)  # previous step direction (momentum)

    iterations = tqdm(np.arange(n_iter)) if progress else np.arange(n_iter)
    for ii in iterations:
        # gradient at the current positions.
        if interp:
            pts_vect_ii = np.array(
                [
                    _interp3(
                        pt_ii,
                        binary.shape,
                        I_ref=sdf_normals[ch],
                        method="linear",
                        cast_uint8=False,
                    )
                    for ch in np.arange(len(sdf_normals))
                ]
            ).T
        else:
            # direct (truncated) lookup with flat indices.
            pts_vect_ii = (
                sdf_normals.reshape(3, -1)[
                    :,
                    np.ravel_multi_index(
                        pt_ii.astype(np.int32).T, sdf_normals.shape[1:], mode="raise", order="C"
                    ),
                ]
            ).T

        if renorm_gradient:
            pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:, None] + 1e-20)

        if use_connectivity:
            pts_vect_ii = alpha * pts_vect_ii + (1 - alpha) * W.dot(pts_vect_ii)

        # momentum update, with step size decaying over iterations.
        pt_ii_next = pt_ii + (delta * pts_vect_ii + mu * g0) * (1.0 / (delta + mu)) * (
            delta / (1.0 + ii * gradient_decay)
        )
        g0 = pts_vect_ii.copy()

        pt_ii_next[:, 0] = np.clip(pt_ii_next[:, 0], 0, binary.shape[0] - 1)
        pt_ii_next[:, 1] = np.clip(pt_ii_next[:, 1], 0, binary.shape[1] - 1)
        pt_ii_next[:, 2] = np.clip(pt_ii_next[:, 2], 0, binary.shape[2] - 1)

        pt_ii = pt_ii_next
        yield ii, pt_ii


def _new_3d_debug_axes():
    import pylab as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")  # this works better!.
    ax.set_box_aspect(aspect=(1, 1, 1))
    return ax


# =============================================================================
# Gradient descent segmentation
# =============================================================================
def gradient_watershed2D_binary(
    binary,
    gradient_img=None,
    momenta=0.0,
    divergence_rescale=True,
    smooth_sigma=1,
    smooth_gradient=1,
    delta=0.5,
    n_iter=10,
    min_area=5,
    gradient_decay=0.0,
    eps=1e-20,
    interp=True,
    thresh_factor=None,
    track_flow=True,  # if track_flow then we record!.
    mask=None,
    binary_mask_gradient=False,
    debug_viz=False,
):
    """parses the instance level segmentation implicitly given as an input binary or a vector field.
    The algorithm works as an inverse watershed.

    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area

    result is an integer image the same size as binary.

    Parameters
    ----------
    binary : (MxN) numpy array
        input binary image defining the pixels that need labeling
    gradient_img :  (MxNx2) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,0) where img is a potential such as a distance transform or probability map.
    divergence_rescale :
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation.
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step.
    n_iter: int
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed.
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis
    track_flow: bool
        if True, return the positions at every iteration, else only the initial and final positions.
    mask: (MxN) numpy array
        optional binary mask to gate the region to parse labels for.
    binary_mask_gradient: bool
        currently has no effect in 2D.
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration.

    Returns
    -------
    cell_seg_connected_original : (MxN)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
    cell_seg_connected : (MxN)
        the labelled clusters at the final point positions.
    tracks : (n_steps, N, 2)
        point positions over the iterations.
    votes_grid_acc : (MxN)
        the density of the final point positions.

    """
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(2, 0, 1)  # use the supplied gradients!
    else:
        sdf_normals = surf_normal_sdf(
            binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True
        )[0]

    if divergence_rescale:
        curvature_2D = mean_curvature_sdf(
            sdf_normals / (np.linalg.norm(sdf_normals, axis=0)[None, ...] + eps)
        )
        curvature_2D = _normalize99(curvature_2D)  # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_2D[None, ...]

    pts = np.argwhere(binary > 0)  # (N,ndim)

    mu = momenta
    g0 = np.zeros_like(pts)  # previous step direction (momentum)

    tracks = [pts]

    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()

        if interp:
            pts_vect_ii = np.array(
                [
                    _interp2(
                        pt_ii,
                        binary.shape,
                        I_ref=sdf_normals[ch],
                        method="linear",
                        cast_uint8=False,
                    )
                    for ch in np.arange(len(sdf_normals))
                ]
            ).T
        else:
            pts_vect_ii = sdf_normals[
                :, np.rint(pt_ii[:, 0]).astype(np.int32), np.rint(pt_ii[:, 1]).astype(np.int32)
            ].T

        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:, None] + eps)
        pt_ii_next = pt_ii + (delta * pts_vect_ii + mu * g0) * (1.0 / (delta + mu)) * (
            delta / (1.0 + ii * gradient_decay)
        )
        g0 = pts_vect_ii.copy()

        pt_ii_next[:, 0] = np.clip(pt_ii_next[:, 0], 0, binary.shape[0] - 1)
        pt_ii_next[:, 1] = np.clip(pt_ii_next[:, 1], 0, binary.shape[1] - 1)

        if track_flow:
            tracks.append(pt_ii_next)
        else:
            tracks[-1] = pt_ii_next.copy()

        if debug_viz:
            import pylab as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(binary)
            plt.plot(pt_ii_next[:, 1], pt_ii_next[:, 0], "r.")
            plt.show(block=False)

    tracks = np.array(tracks)

    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_2D(
        pts=tracks[-1],
        pts0=pts,
        shape=binary.shape[:2],
        smooth_sigma=smooth_sigma,
        thresh_factor=thresh_factor,
        mask=mask,
        min_area=min_area,
    )

    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc


def gradient_watershed3D_binary(
    binary,
    gradient_img=None,
    momenta=0.75,
    divergence_rescale=True,
    smooth_sigma=1,
    smooth_gradient=1,
    delta=1,
    n_iter=100,
    gradient_decay=0.1,
    min_area=5,
    eps=1e-12,
    thresh_factor=None,
    mask=None,
    interp=False,
    use_connectivity=False,
    connectivity_alpha=0.5,
    binary_mask_gradient=False,
    debug_viz=False,
    renorm_gradient=True,
    sampling=1000,
    track_percent=0,
    rand_seed=0,
    ref_initial_color_img=None,
    ref_alpha=0.01,
    saveplotsfolder=None,
    viewinit=(0, 0),
):
    """parses the instance level segmentation implicitly given as an input binary or a vector field.
    The algorithm works as an inverse watershed.

    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area

    result is an integer image the same size as binary.

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
    gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map.
    divergence_rescale :
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation.
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step.
    n_iter: int
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed.
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    use_connectivity: bool
        if True, average each point's gradient with its neighbours' (weight connectivity_alpha for its own).
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration.
    track_percent: float
        fraction of points (randomly chosen with rand_seed) whose positions are recorded every iteration.

    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
    cell_seg_connected : (MxNxL)
        the labelled clusters at the final point positions.
    tracks : (n_steps, n_tracked, 3) array, or [] if track_percent=0
    votes_grid_acc : (MxNxL)
        the density of the final point positions.

    """
    sdf_normals = _initial_gradients_3D(
        binary, gradient_img, binary_mask_gradient, smooth_gradient, eps
    )

    if divergence_rescale:
        sdf_normals = _curvature_rescale_3D(sdf_normals)

    pts = np.argwhere(binary > 0).astype(np.float32)  # (N,ndim)
    pt_ii = pts

    tracks = []

    if track_percent > 0:
        np.random.seed(rand_seed)
        n_samples_track = int(track_percent * len(pts))
        select_pts_track = np.arange(len(pts))
        np.random.shuffle(select_pts_track)

        select_pts_track = select_pts_track[:n_samples_track].copy()
        tracks.append(pts[select_pts_track])

    if debug_viz:
        import pylab as plt

        from .plotting import set_axes_equal

        if ref_initial_color_img is not None:
            ref_pts_color = ref_initial_color_img[
                pts[::sampling, 0].astype(np.int32),
                pts[::sampling, 1].astype(np.int32),
                pts[::sampling, 2].astype(np.int32),
            ].copy()
        else:
            ref_pts_color = "k"

    for ii, pt_ii in _advect_points_3D(
        binary,
        sdf_normals,
        pts,
        n_iter,
        delta,
        momenta,
        gradient_decay,
        interp,
        renorm_gradient,
        use_connectivity,
        connectivity_alpha,
        progress=True,
    ):
        if track_percent > 0:
            tracks.append(pt_ii[select_pts_track, :])

        if debug_viz:
            ax = _new_3d_debug_axes()
            # starting positions (baseline), then current positions.
            ax.scatter(
                pts[::sampling, 0],
                pts[::sampling, 1],
                pts[::sampling, 2],
                c=ref_pts_color,
                s=1,
                alpha=ref_alpha,
                zorder=0,
            )
            ax.scatter(
                pt_ii[::sampling, 0],
                pt_ii[::sampling, 1],
                pt_ii[::sampling, 2],
                c="magenta",
                s=1,
                zorder=1000,
            )
            ax.view_init(viewinit[0], viewinit[1])
            set_axes_equal(ax)

            if saveplotsfolder is not None:
                # set this off for better looking visualization.
                ax.grid("off")
                ax.axis("off")
                plt.savefig(
                    os.path.join(saveplotsfolder, "iter_" + str(ii).zfill(8) + ".png"),
                    dpi=120,
                    bbox_inches="tight",
                )

            plt.show(block=False)

    if len(tracks) > 0:
        tracks = np.array(tracks)

    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D(
        pts=pt_ii,
        pts0=pts,
        shape=binary.shape[:3],
        smooth_sigma=smooth_sigma,
        thresh_factor=thresh_factor,
        mask=mask,
        min_area=min_area,
    )

    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc


def gradient_watershed3D_binary_dynamics(
    binary,
    gradient_img=None,
    momenta=0.75,
    divergence_rescale=False,
    smooth_sigma=1,
    smooth_gradient=1,
    gradient_decay=0.0,
    delta=1,
    n_iter=100,
    min_area=5,
    eps=1e-12,
    thresh_factor=None,
    mask=None,
    interp=False,
    use_connectivity=False,
    connectivity_alpha=0.5,
    debug_viz=False,
    binary_mask_gradient=False,
    renorm_gradient=True,
):
    """Only the point dynamics (step 2) of gradient_watershed3D_binary, without the clustering.

    The parameters are as for gradient_watershed3D_binary; smooth_sigma, min_area,
    thresh_factor and mask are accepted for compatibility but unused.

    Returns
    -------
    (2, N, 3) array of the initial and final point positions.
    """
    sdf_normals = _initial_gradients_3D(
        binary, gradient_img, binary_mask_gradient, smooth_gradient, eps
    )

    if divergence_rescale:
        sdf_normals = _curvature_rescale_3D(sdf_normals)

    pts = np.argwhere(binary > 0).astype(np.float32)  # (N,ndim)
    pt_ii = pts

    for _, pt_ii in _advect_points_3D(
        binary,
        sdf_normals,
        pts,
        n_iter,
        delta,
        momenta,
        gradient_decay,
        interp,
        renorm_gradient,
        use_connectivity,
        connectivity_alpha,
        progress=False,
    ):
        if debug_viz:
            import pylab as plt

            from .plotting import set_axes_equal

            sampling = 1000
            ax = _new_3d_debug_axes()
            ax.scatter(pt_ii[::sampling, 0], pt_ii[::sampling, 1], pt_ii[::sampling, 2], c="r", s=1)
            ax.view_init(0, 180)
            set_axes_equal(ax)
            plt.show(block=False)

    return np.concatenate([pts[None, ...], pt_ii[None, ...]], axis=0)


def _auto_tile_shape(binary_shape, n_procs, tile_aspect, tile_overlap_ratio):
    """Tile shape with the given aspect ratio giving roughly one (overlapping) tile per process."""
    total_vol = np.prod(binary_shape[:3])
    per_vol = total_vol / n_procs

    tile_shape = tuple(
        np.int32(
            np.hstack(
                (per_vol / (float(np.sum(tile_aspect)))) ** (1.0 / 3) * np.hstack(tile_aspect)
            )
        )
    )

    smallest_ind = np.argmin(binary_shape)
    max_correction_factor = np.int32(np.ceil(binary_shape[smallest_ind] / tile_shape[smallest_ind]))

    overlap = np.int32(np.hstack(tile_shape) * tile_overlap_ratio)
    tiler = Tiler(data_shape=binary_shape, tile_shape=tile_shape, overlap=overlap)

    correction_factor = (tiler.n_tiles * 1.0 / tile_overlap_ratio / n_procs) ** (1.0 / 3)
    correction_factor = np.minimum(correction_factor, max_correction_factor)
    return tuple(np.int32(np.hstack(tile_shape) * correction_factor))


def mp_gradient_watershed3D_binary(
    binary,
    gradient_img=None,
    tile_shape=None,
    tile_aspect=(1, 2, 2),
    tile_overlap_ratio=0.25,  # should be bigger than the most movement.
    momenta=0.95,
    divergence_rescale=False,
    gradient_decay=0.0,
    smooth_sigma=1,
    smooth_gradient=1,
    delta=1,
    n_iter=100,
    min_area=5,
    eps=1e-12,
    thresh_factor=None,
    mask=None,
    interp=False,
    use_connectivity=False,
    connectivity_alpha=0.5,
    binary_mask_gradient=False,
    debug_viz=False,
    renorm_gradient=False,
):
    """Tiled, multi-threaded version of gradient_watershed3D_binary.

    The volume is split into overlapping tiles (tile_shape, or inferred from tile_aspect).
    The dynamics are run per tile, keeping only points that start outside the overlap
    margins, then all final positions are clustered together.

    Returns
    -------
    cell_seg_connected_original, cell_seg_connected, final point positions, votes_grid_acc
    """
    n_procs = mp.cpu_count() - 1

    if tile_shape is None:
        tile_shape = _auto_tile_shape(binary.shape, n_procs, tile_aspect, tile_overlap_ratio)
        print("using tile shape: ", tile_shape)

    overlap = np.int32(np.hstack(tile_shape) * tile_overlap_ratio)

    tiler = Tiler(data_shape=binary.shape, tile_shape=tile_shape, overlap=overlap)

    # pad the volume so the tiles cover it exactly.
    new_shape, padding = tiler.calculate_padding()
    print(new_shape, padding)

    offset = np.hstack([p[0] for p in padding])
    tiler.recalculate(data_shape=new_shape)

    shape = binary.shape

    sdf_normals = _initial_gradients_3D(
        binary, gradient_img, binary_mask_gradient, smooth_gradient, eps
    )

    im_binary = np.pad(binary, padding, mode="constant")
    sdf_normals = np.pad(sdf_normals, [(0, 0)] + padding)

    print("using total number of tiles : ", tiler.n_tiles)

    def _run_dynamics(tile_id):
        bbox_low, bbox_high = tiler.get_tile_bbox(tile_id)
        size = np.hstack(bbox_high) - np.hstack(bbox_low)
        tile = (
            slice(bbox_low[0], bbox_high[0]),
            slice(bbox_low[1], bbox_high[1]),
            slice(bbox_low[2], bbox_high[2]),
        )
        res = gradient_watershed3D_binary_dynamics(
            im_binary[tile],
            gradient_img=sdf_normals[(slice(None),) + tile].transpose(1, 2, 3, 0),
            momenta=momenta,
            divergence_rescale=divergence_rescale,
            smooth_sigma=smooth_sigma,
            smooth_gradient=smooth_gradient,
            gradient_decay=gradient_decay,
            delta=delta,
            n_iter=n_iter,
            eps=eps,
            min_area=min_area,
            thresh_factor=None,
            use_connectivity=use_connectivity,
            connectivity_alpha=connectivity_alpha,
            renorm_gradient=renorm_gradient,
            binary_mask_gradient=binary_mask_gradient,
            debug_viz=False,
        )
        res = res.astype(np.float32)

        # keep only the points starting outside the tile's overlap margins.
        keep = np.ones(res[0].shape[:1], dtype=bool)
        print("Tile id", tile_id, len(keep), np.sum(keep))

        for dim in np.arange(res.shape[-1]):
            keep = np.logical_and(
                keep,
                np.logical_and(
                    res[0, ..., dim] >= overlap[dim] // 2,
                    res[0, ..., dim] < size[dim] - overlap[dim] // 2,
                ),
            )

        # tile coordinates -> padded volume coordinates.
        return res[:, keep > 0] + np.hstack(bbox_low)[None, None, :]

    print("start dynamics")
    t1 = time.time()

    with ThreadPool(n_procs) as pool:
        res = pool.map(_run_dynamics, range(0, tiler.n_tiles))
        res = np.concatenate(res, 1)
        res = res - offset[None, None, :]  # remove the padding.

    t2 = time.time()
    print("finished dynamics:, ", t2 - t1)

    # clip the points to the original shape.
    res[..., 0] = np.clip(res[..., 0], 0, shape[0] - 1)
    res[..., 1] = np.clip(res[..., 1], 0, shape[1] - 1)
    res[..., 2] = np.clip(res[..., 2], 0, shape[2] - 1)

    # forward the user mask (same as the serial gradient_watershed3D_binary) so do_mp gives the same threshold.
    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D(
        pts=res[1],
        pts0=res[0],
        shape=shape,
        smooth_sigma=smooth_sigma,
        thresh_factor=thresh_factor,
        mask=mask,
        min_area=min_area,
    )

    return cell_seg_connected_original, cell_seg_connected, res[1], votes_grid_acc
