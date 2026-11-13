# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 03:04:09 2023

@author: fyz11

Module for computing and implementing various flow definitions.

A "flow" is the (normalized) gradient of a distance transform computed per labelled
cell. Available distance transforms (``dtform_method``):

- 'edt': Euclidean distance transform.
- 'cellpose': Cellpose's heat diffusion from the cell centre.
- 'cellpose_improve': heat diffusion from the cell centre, solved exactly (Poisson equation).
- 'fmm': geodesic distance from the cell centre (fast marching method).
- 'cellpose_skel' / 'fmm_skel': as above but from the cell's medial-axis skeleton.
"""

import multiprocessing

import edt  # fast distance transform from seung lab. https://github.com/seung-lab/euclidean-distance-transform-3d
import multiprocess as mp
import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.segmentation as sksegmentation
from numba import int32, njit
from scipy.sparse.linalg import spsolve

from .filters import smooth_vol, var_combine

# Per view: (name, axes putting that view's slicing axis first, axes mapping its (N,3,A,B)
# gradients back to (Z,3,Y,X), channel order putting [0, d_a, d_b] into (dz, dy, dx)).
_VIEWS = (
    ("xy", (0, 1, 2), None, [0, 1, 2]),
    ("xz", (1, 0, 2), (2, 1, 0, 3), [1, 0, 2]),
    ("yz", (2, 0, 1), (2, 1, 3, 0), [1, 2, 0]),
)


# =============================================================================
#  Euclidean distance transform (edt library, faster).
# =============================================================================
def distance_transform_labels_fast(labels, sdf=False, n_threads=16, black_border=False):
    """compute euclidean distance transform for each uniquely labelled cell in a 2D/3D binary image

    3D label volumes are transformed slice-by-slice. If sdf, compute the signed distance function.
    """
    transform = edt.sdf if sdf else edt.edt

    def _dtform(img):
        return transform(img, black_border=black_border, order="C", parallel=n_threads).astype(
            np.float32
        )

    if len(labels.shape) == 2:
        return _dtform(labels)
    return np.array([_dtform(ss) for ss in labels])


# =============================================================================
#   Poisson (heat) distance transform
# =============================================================================
def _laplacian_matrix(n, m, mask=None):
    """Sparse 5-point Laplacian on an n x m grid; rows of masked-out (mask == 0) pixels become identity."""
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    if mask is not None:
        x_range = mask.shape[1]
        zeros = np.argwhere(mask == 0)  # in (y,x)
        k = zeros[:, 1] + zeros[:, 0] * x_range
        mat_A[k, k] = 1
        mat_A[k, k + 1] = 0
        mat_A[k, k - 1] = 0
        mat_A[k, k + x_range] = 0
        mat_A[k, k - x_range] = 0

    return mat_A.tocsc()


def poisson_dist_tform(binary, pt=None):
    """
    Computation for a single binary image.

    Solves the Poisson equation inside the binary, with point sources at pt (N,2) if
    given, else a uniform source.
    """
    mask = np.pad(
        binary, [1, 1], mode="constant", constant_values=1
    )  # ones needed for connectivity

    mat_A = _laplacian_matrix(mask.shape[0], mask.shape[1], mask=mask)

    if pt is not None:
        mask = np.zeros_like(mask)
        mask[pt[:, 0], pt[:, 1]] = 1

        mask_flat = mask.flatten()
        mat_b = np.zeros(len(mask_flat))
        mat_b[mask_flat == 1] = 1

    else:
        # zero the padding
        mask[0, :] = 0
        mask[:, -1] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0

        # solve within the mask: \Delta f = div v = \Delta g
        mask_flat = mask.flatten()
        mat_b = np.ones(len(mask_flat))
        mat_b[mask_flat == 0] = 0

    x = spsolve(mat_A, mat_b, permc_spec="MMD_AT_PLUS_A")
    x = x.reshape(mask.shape)
    x = x[1:-1, 1:-1].copy()
    x = x - x.min()  # solution is only positive!.

    return x


def _values_and_gradient_at(image, coords, zero_gradient_at=None):
    """The image values at coords and its (dy, dx) gradient there, as (values, (2, N))."""
    gradient = np.array(np.gradient(image))
    if zero_gradient_at is not None:
        gradient[:, zero_gradient_at[:, 0], zero_gradient_at[:, 1]] = 0

    dy = gradient[0, coords[:, 0], coords[:, 1]].copy()
    dx = gradient[1, coords[:, 0], coords[:, 1]].copy()

    return image[coords[:, 0], coords[:, 1]], np.stack((dy, dx))


def _coords_to_binary(coords, shape):
    binary = np.zeros(shape, dtype=bool)
    binary[coords[:, 0], coords[:, 1]] = 1
    return binary


def poisson_dist_tform_flow(coords, centroid, shape, power_dist=None, norm_dist=False):
    """
    Solve diffusion for points.

    Heat diffusion from the point source centroid within the object given by coords.
    If power_dist, the (clipped to 0-1) distance is raised to this power to boost the gradients.
    """
    mask = _coords_to_binary(coords, shape)

    poisson = poisson_dist_tform(mask > 0, pt=centroid[None, :])
    poisson = ndimage.gaussian_filter(poisson, sigma=1)

    if power_dist is not None:
        # this is to boost beyond the numerical limits.
        poisson = np.clip(poisson, 0, 1)
        poisson = poisson**power_dist  # valid for 0-1

    return _values_and_gradient_at(poisson, coords)


def poisson_dist_tform_flow_sets(coords, pts, shape, power_dist=None):
    """
    Solve diffusion for points.

    Heat diffusion from the source set pts (e.g. a skeleton) within the object given by coords.
    """
    mask = _coords_to_binary(coords, shape)

    poisson = poisson_dist_tform(mask > 0, pt=pts)
    poisson = ndimage.gaussian_filter(poisson, sigma=1)

    if power_dist is not None:
        # this is to boost beyond the numerical limits.
        poisson = poisson - poisson.min()
        poisson = poisson**power_dist

    return _values_and_gradient_at(poisson, coords, zero_gradient_at=pts)


# =============================================================================
#   Geodesic (fast marching method) distance transform
# =============================================================================
def _fmm_distance_image(coords, shape, source_index):
    """Inverted geodesic distance from the source(s) inside the object; negative distance outside."""
    import skfmm

    binary = _coords_to_binary(coords, shape)

    mask = np.logical_not(binary).copy()  # the masked out region
    m = np.ones_like(binary)
    m[source_index] = 0  # the source
    m = np.ma.masked_array(m, mask.copy())

    dist_image = skfmm.distance(m)
    dist_image = dist_image.max() - dist_image  # invert the image

    # fix up the boundary differentiation (gives a little normal push).
    dist_outer = skfmm.distance(mask) * -1
    dist_image[mask > 0] = dist_outer[mask > 0]

    return dist_image


def fmm_point_source2D(coords, centroid, shape):
    """
    This is a much better solver of the diffusion

    Geodesic distance from the point source centroid within the object given by coords.
    """
    dist_image = _fmm_distance_image(coords, shape, (centroid[0], centroid[1]))
    return _values_and_gradient_at(dist_image, coords)


def fmm_point_source2D_sets(coords, source_pts, shape):
    """
    This is a much better solver of the diffusion

    Geodesic distance from the source set source_pts (e.g. a skeleton) within the object.
    """
    dist_image = _fmm_distance_image(coords, shape, (source_pts[:, 0], source_pts[:, 1]))
    return _values_and_gradient_at(dist_image, coords, zero_gradient_at=source_pts)


def _largest_component_vol(vol_binary, connectivity=1):
    r"""Given a binary segmentation, return only the largest connected component of the given connectivity

    Parameters
    ----------
    vol : array
        input binary image
    connectivity : 1 or 2
        if 1, the local 4-neighbors for 2D or 6-neighbors for 3D.
        if 2, the local 8-neighbors for 2D or 26-neighbors for 3D.

    Returns
    -------
    vol_binary : array
        output binary image same size as input retaining only the largest connected component

    """
    vol_binary_labelled = skmeasure.label(vol_binary, connectivity=connectivity)
    vol_binary_vols = [re.area for re in skmeasure.regionprops(vol_binary_labelled)]

    return vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])


# =============================================================================
#   Replicating cellpose flow
# =============================================================================
@njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

    taken from  cellpose source code.

    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = (
            1
            / 9.0
            * (
                T[y * Lx + x]
                + T[(y - 1) * Lx + x]
                + T[(y + 1) * Lx + x]
                + T[y * Lx + x - 1]
                + T[y * Lx + x + 1]
                + T[(y - 1) * Lx + x - 1]
                + T[(y - 1) * Lx + x + 1]
                + T[(y + 1) * Lx + x - 1]
                + T[(y + 1) * Lx + x + 1]
            )
        )
    return T


def cellpose_diffuse2D(coords, centroid, shape, niter):
    """
    Solve diffusion for points.

    Cellpose's iterative heat diffusion from centroid within the object given by coords.
    """
    ly, lx = shape

    x = coords[:, 1].copy()
    y = coords[:, 0].copy()

    T = np.zeros((ly + 2) * (lx + 2), np.float64)
    T = _extend_centers(
        T,
        y=y.astype(np.int32),
        x=x.astype(np.int32),
        ymed=centroid[0].astype(np.int32),
        xmed=centroid[1].astype(np.int32),
        Lx=int32(lx),
        niter=int32(niter),
    )
    T[(y + 1) * lx + x + 1] = np.log(1.0 + T[(y + 1) * lx + x + 1])
    dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
    dx = T[y * lx + x + 1] - T[y * lx + x - 1]

    return T[(y + 1) * lx + x + 1], np.stack((dy, dx))


# =============================================================================
#   Generic wrapper functions.
# =============================================================================
def distance_tform_labels2D(labelled, dtform_fnc, clip=False):
    """
    this function wraps a distance transform defined for a single binary and generalize to multi-labels.

    If clip, each cell's distance is normalized to 0-1, else only clipped to be >= 0.
    """
    distance = np.zeros(labelled.shape, dtype=np.float32)

    for reg in skmeasure.regionprops(labelled):
        y1, x1, _, _ = reg.bbox  # (min_row, min_col, max_row, max_col)
        coords = reg.coords

        dist = dtform_fnc(reg.image)

        if clip:
            dist = dist / dist.max()
            dist = np.clip(dist, 0, 1)
        else:
            dist = np.clip(dist, 0, np.inf)

        distance[coords[:, 0], coords[:, 1]] = dist[coords[:, 0] - y1, coords[:, 1] - x1].copy()

    return distance


def _medial_skeleton(patch_mask, smooth_skel_sigma):
    """Dilated medial-axis skeleton of a binary; smoothing first removes small branches."""
    if smooth_skel_sigma > 0:
        patch_mask = ndimage.gaussian_filter(patch_mask * 255.0, sigma=smooth_skel_sigma)
        patch_mask = patch_mask / (patch_mask.max() + 1e-12) > 0.5
        skel = skmorph.skeletonize(patch_mask)
        skel_base = skel.copy()
        # smooth and reskeletonize, keeping the largest piece.
        skel = ndimage.gaussian_filter(skel * 255.0, sigma=smooth_skel_sigma)
        skel = skel / (skel.max() + 1e-12)
        skel = skel > np.mean(skel)
        try:
            skel = _largest_component_vol(skel)
            skel = skmorph.skeletonize(skel)
        except:  # noqa: E722
            skel = skel_base.copy()
    else:
        skel = skmorph.skeletonize(patch_mask)

    return skmorph.binary_dilation(skel, skmorph.disk(1))


def distance_centroid_tform_flow_labels2D(
    labelled,
    dtform_method="cellpose",
    guide_img=None,
    fixed_point_percentile=0.9,
    iter_factor=5,
    smooth_skel_sigma=3,
    power_dist=None,
):
    """
    This function wraps a distance transform defined for a single binary with point source and generalize to multi-labels.

    Each cell's source point is the interior point closest to its median coordinate,
    restricted to points with guide_img >= its fixed_point_percentile if guide_img is given.

    Returns
    -------
    (3, M, N) array: the distance and the unit (dy, dx) flow.
    """
    distance = np.zeros(labelled.shape)
    flow = np.zeros((2,) + labelled.shape)

    for reg in skmeasure.regionprops(labelled):
        patch = reg.image
        y1, x1, _, _ = reg.bbox  # (min_row, min_col, max_row, max_col)
        start = np.hstack([y1, x1])
        coords = reg.coords

        if len(coords) <= 8:  # minimum size.
            continue

        centroid = np.nanmedian(coords, axis=0)

        # interior pixels (for stability of the solvers).
        inner = sksegmentation.find_boundaries(patch * 1, mode="inner")
        inner = np.logical_and(patch, np.logical_not(inner))
        inner_coords = np.argwhere(inner > 0)

        if dtform_method == "fmm_skel" or dtform_method == "cellpose_skel":
            skel = _medial_skeleton(inner.copy(), smooth_skel_sigma)

        if len(inner_coords) > 0:
            inner_coords = (inner_coords + start[None, :]).copy()
        else:
            inner_coords = coords.copy()

        if guide_img is not None:
            # use the guide to find an internal fixed point.
            coords_guide = guide_img[inner_coords[:, 0], inner_coords[:, 1]].copy()
            coords_valid = inner_coords[
                coords_guide >= np.percentile(coords_guide, fixed_point_percentile * 100)
            ]
            centroid = coords_valid[
                np.argmin(np.linalg.norm(coords_valid - centroid[None, :], axis=1))
            ]
        else:
            centroid = inner_coords[
                np.argmin(np.linalg.norm(inner_coords - centroid[None, :], axis=1))
            ]

        # solve in the cell's bounding box, with a 1 pixel border.
        local_coords = coords - start[None, :] + 1
        local_centroid = centroid - start + 1
        local_shape = np.hstack(patch.shape) + 1

        if dtform_method == "cellpose":
            niter = iter_factor * np.int32(np.ptp(coords[:, 1]) + np.ptp(coords[:, 0]))
            dist_, flow_ = cellpose_diffuse2D(
                local_coords, local_centroid, shape=local_shape, niter=niter
            )

        if dtform_method == "cellpose_improve":
            dist_, flow_ = poisson_dist_tform_flow(
                local_coords, local_centroid, shape=local_shape, power_dist=power_dist
            )

        if dtform_method == "fmm":
            dist_, flow_ = fmm_point_source2D(local_coords, local_centroid, shape=local_shape)

        if dtform_method == "fmm_skel":
            if np.sum(skel > 0) > 0:
                dist_, flow_ = fmm_point_source2D_sets(
                    local_coords, np.argwhere(skel > 0) + 1, shape=local_shape
                )
            else:
                dist_ = np.zeros(len(coords))
                flow_ = np.zeros((2, len(coords)))

        if dtform_method == "cellpose_skel":
            if np.sum(skel > 0) > 0:
                dist_, flow_ = poisson_dist_tform_flow_sets(
                    local_coords,
                    np.argwhere(skel > 0) + 1,
                    shape=local_shape,
                    power_dist=power_dist,
                )
            else:
                dist_ = np.zeros(len(coords))
                flow_ = np.zeros((2, len(coords)))

        distance[coords[:, 0], coords[:, 1]] = dist_.copy()
        flow[:, coords[:, 0], coords[:, 1]] = flow_.copy()

    # normalise
    flow /= 1e-20 + (flow**2).sum(axis=0) ** 0.5

    return np.concatenate([distance[None, ...], flow], axis=0)


# =============================================================================
# Parallel scripts
# =============================================================================
def _split_into_chunks(array, n_chunks):
    """Split along the first axis into n_chunks pieces; the last also takes the remainder."""
    chunksize = len(array) // n_chunks

    chunks = []
    for i_proc in range(n_chunks):
        chunkstart = i_proc * chunksize
        chunkend = (i_proc + 1) * chunksize if i_proc < n_chunks - 1 else None
        chunks.append(array[slice(chunkstart, chunkend)])

    assert sum(map(len, chunks)) == len(array)
    return chunks


def _distance_centroid_tform_flow_labels2D_chunk(
    labelled_array,
    dtform_method="cellpose",
    guide_img=None,
    fixed_point_percentile=0.9,
    iter_factor=5,
    smooth_skel_sigma=3,
    power_dist=None,
):
    return np.array(
        [
            distance_centroid_tform_flow_labels2D(
                labelled_array[zz],
                dtform_method=dtform_method,
                guide_img=None if guide_img is None else guide_img[zz],
                fixed_point_percentile=fixed_point_percentile,
                iter_factor=iter_factor,
                smooth_skel_sigma=smooth_skel_sigma,
                power_dist=power_dist,
            )
            for zz in np.arange(len(labelled_array))
        ]
    )


def distance_centroid_tform_flow_labels2D_parallel(
    labelled_array,
    dtform_method="cellpose",
    guide_image=None,
    fixed_point_percentile=0.9,
    iter_factor=5,
    n_processes=4,
    smooth_skel_sigma=3,
    power_dist=None,
):
    """distance_centroid_tform_flow_labels2D for each slice of a stack, in n_processes chunks.

    Note: the whole guide_image (not its chunk) is passed to each chunk.
    """
    chunks = _split_into_chunks(labelled_array, n_processes)
    print("starting parallel")

    with mp.Pool(processes=n_processes) as pool:
        proc_results = [
            pool.apply_async(
                _distance_centroid_tform_flow_labels2D_chunk,
                args=(
                    chunk,
                    dtform_method,
                    guide_image,
                    fixed_point_percentile,
                    iter_factor,
                    smooth_skel_sigma,
                    power_dist,
                ),
            )
            for chunk in chunks
        ]
        res = np.vstack([r.get() for r in proc_results])

    return res


def distance_centroid_tform_flow_labels2D_dask(
    labelled_array,
    dtform_method="cellpose_improve",
    guide_image=None,
    fixed_point_percentile=0.9,
    iter_factor=5,
    n_processes=4,
    smooth_skel_sigma=3,
    power_dist=None,
):
    """distance_centroid_tform_flow_labels2D for each slice of a stack, using a local dask cluster.

    Returns (N, 3, A, B) float32.
    """
    import dask
    from dask.distributed import Client

    client = Client(n_workers=n_processes, threads_per_worker=1)

    lazy_flow = dask.delayed(distance_centroid_tform_flow_labels2D)

    kwargs = {
        "dtform_method": dtform_method,
        "fixed_point_percentile": fixed_point_percentile,
        "smooth_skel_sigma": smooth_skel_sigma,
    }
    if dtform_method == "cellpose_improve":
        kwargs["power_dist"] = power_dist
    else:
        kwargs["iter_factor"] = iter_factor
        if guide_image is None:
            # note: power_dist is only forwarded here when there is no guide image.
            kwargs["power_dist"] = power_dist

    res = [
        lazy_flow(
            labelled_array[sli],
            guide_img=None if guide_image is None else guide_image[sli],
            **kwargs,
        )
        for sli in np.arange(len(labelled_array))
    ]
    res = dask.compute(res)
    res = np.array(res[0], dtype=np.float32)

    client.close()

    return res


def _distance_tform_labels2D_chunk(labelled_array, dtform_fnc, clip=False):
    return np.array(
        [
            distance_tform_labels2D(labelled_array[zz], dtform_fnc=dtform_fnc, clip=clip)
            for zz in np.arange(len(labelled_array))
        ]
    )


def distance_tform_labels2D_parallel(labelled_array, dtform_fnc, clip=False, n_processes=4):
    """distance_tform_labels2D for each slice of a stack, in n_processes chunks."""
    chunks = _split_into_chunks(labelled_array, n_processes)

    # note: uses the standard library multiprocessing (dtform_fnc must be picklable).
    with multiprocessing.Pool(processes=n_processes) as pool:
        proc_results = [
            pool.apply_async(_distance_tform_labels2D_chunk, args=(chunk, dtform_fnc, clip))
            for chunk in chunks
        ]
        res = np.vstack([r.get() for r in proc_results])

    return res


def _relabel_slices(labelled, bg_label=0):
    """Offset each slice's labels so that labels are unique across slices."""
    max_ID = 0
    labelled_ = []
    for lab in labelled:
        lab_ = lab.copy()
        lab_[lab > bg_label] = lab[lab > bg_label] + max_ID  # only update the foreground!
        labelled_.append(lab_)
        if np.max(lab) > 0:
            max_ID = np.max(lab) + 1

    return np.array(labelled_)


# =============================================================================
# Flow consistency: remove masks whose re-derived flows disagree with a reference flow
# =============================================================================
def _view_gradients_to_zyx(gradient, view_index):
    """(N,2,A,B) per-slice 2D gradients of a view -> (Z,3,Y,X) with channels (dz, dy, dx)."""
    _, _, back_axes, channel_order = _VIEWS[view_index]
    if back_axes is not None:
        gradient = gradient.transpose(back_axes)
    gradient = np.concatenate([np.zeros_like(gradient[:, 1])[:, None, ...], gradient], axis=1)
    return gradient[:, channel_order, ...]


def _fuse_view_gradients(xy, xz, yz, filter_scale):
    """Combine the (Z,3,Y,X) gradients of the three views component by component."""

    def fuse(a, b):
        return var_combine(
            [ndimage.gaussian_filter(a, sigma=1), ndimage.gaussian_filter(b, sigma=1)],
            ksize=filter_scale,
            alpha=0.5,
        )

    dx = fuse(xy[:, 2], xz[:, 2])
    dy = fuse(xy[:, 1], yz[:, 1])
    dz = fuse(xz[:, 0], yz[:, 0])
    return dz, dy, dx


def _stack_zyx(dz, dy, dx):
    """Stack the components as (3,Z,Y,X)."""
    return np.concatenate([dz[:, None, ...], dy[:, None, ...], dx[:, None, ...]], axis=1).transpose(
        1, 0, 2, 3
    )


def _normalize_vectors(gradients):
    return gradients / (np.linalg.norm(gradients, axis=0)[None, ...] + 1e-20)


def _flow_errors(labels_gradients, flow, mask):
    """Mean squared difference between the two flows, per label 1..mask.max() (0 if absent)."""
    print("finished computing gradients and removing bad flow masks ...")
    flow_errors = np.zeros(mask.max())
    for i in range(labels_gradients.shape[0]):
        flow_errors += ndimage.mean(
            (labels_gradients[i] - flow[i]) ** 2, mask, index=np.arange(1, mask.max() + 1)
        )
    flow_errors[np.isnan(flow_errors)] = 0
    return flow_errors


def _remove_error_labels_by_position(mask, flow_errors, flow_threshold):
    """Zero out labels with flow error > flow_threshold.

    Assumes the labels are 1..N without gaps (regionprops are looked up at label - 1).
    """
    error_labels = flow_errors > flow_threshold

    if np.sum(error_labels) == 0:
        return mask

    reg_error_labels = np.arange(1, mask.max() + 1)[error_labels]

    mask_new = mask.copy()
    mask_regionprops = skmeasure.regionprops(mask)
    for lab in reg_error_labels:
        coords = mask_regionprops[lab - 1].coords
        mask_new[tuple(coords.T)] = 0
    return mask_new


def _check_flow_shape(mask, flow):
    if flow.shape[1:] != mask.shape:
        print("ERROR: input ref flow is not same size as predicted masks")
        return False
    return True


def remove_bad_flow_masks_2D(
    mask,
    flow,
    flow_threshold=0.6,
    dtform_method="cellpose_improve",
    fixed_point_percentile=0.01,
    n_processes=4,
    power_dist=None,
    alpha=0.5,
    filter_scale=1,
):
    """Remove 2D masks whose flows, re-derived with dtform_method, differ from flow by more than flow_threshold (MSE).

    Returns (filtered mask, per-label flow errors, re-derived flows), or None on error.
    """
    if not _check_flow_shape(mask, flow):
        return

    if dtform_method == "cellpose":
        print("ERROR: not yet implemented.")
        return

    label_xy = mask.copy()
    guide_image_xy = distance_transform_labels_fast(label_xy)

    labels_gradients = distance_centroid_tform_flow_labels2D(
        label_xy,
        dtform_method=dtform_method,
        guide_img=guide_image_xy,
        fixed_point_percentile=fixed_point_percentile,
        iter_factor=5,
        power_dist=power_dist,
    )
    labels_gradients = _normalize_vectors(labels_gradients[1:].copy())

    flow_errors = _flow_errors(labels_gradients, flow, mask)
    mask_out = _remove_error_labels_by_position(mask, flow_errors, flow_threshold)
    return mask_out, flow_errors, labels_gradients


def remove_bad_flow_masks_3D(
    mask,
    flow,
    flow_threshold=0.6,
    dtform_method="cellpose_improve",
    fixed_point_percentile=0.01,
    n_processes=4,
    smooth_skel_sigma=3,
    power_dist=None,
    alpha=0.5,
    smooth_sigma=0,
    smooth_ds=1,
    filter_scale=1,
):
    """Remove 3D masks whose flows, re-derived from their xy, xz and yz slices with dtform_method,
    differ from flow by more than flow_threshold (MSE).

    Returns (filtered mask, per-label flow errors, re-derived flows), or None on error.
    """
    if not _check_flow_shape(mask, flow):
        return

    if dtform_method == "cellpose":
        print("ERROR: not yet implemented.")
        return

    view_gradients = []
    for view_index, (view, view_axes, _, _) in enumerate(_VIEWS):
        if view_index == 0:
            print("starting xy")
            labels = mask.copy()
        else:
            labels = mask.transpose(view_axes)
        guide_image = distance_transform_labels_fast(labels)
        gradient = distance_centroid_tform_flow_labels2D_dask(
            labels,
            dtform_method=dtform_method,
            guide_image=guide_image,
            fixed_point_percentile=fixed_point_percentile,
            n_processes=n_processes,
            smooth_skel_sigma=smooth_skel_sigma,
            power_dist=power_dist,
        )
        view_gradients.append(_view_gradients_to_zyx(gradient[:, 1:].copy(), view_index))
        print("done xy " if view_index == 0 else "done " + view)
        del labels, guide_image

    dz, dy, dx = _fuse_view_gradients(*view_gradients, filter_scale)

    dx = ndimage.gaussian_filter(dx, sigma=1.0)
    dy = ndimage.gaussian_filter(dy, sigma=1.0)
    dz = ndimage.gaussian_filter(dz, sigma=1.0)

    labels_gradients = _stack_zyx(dz, dy, dx)
    del dx, dy, dz

    print("computed true gradients", labels_gradients.shape)

    if smooth_sigma > 0:
        for ch_no in np.arange(len(labels_gradients)):
            labels_gradients[ch_no] = smooth_vol(
                labels_gradients[ch_no], ds=smooth_ds, smooth=smooth_sigma
            )

    labels_gradients = _normalize_vectors(labels_gradients)

    flow_errors = _flow_errors(labels_gradients, flow, mask)

    error_labels = flow_errors > flow_threshold
    if np.sum(error_labels) == 0:
        return mask, flow_errors, labels_gradients

    # labels may have gaps: find each error label's position among the present labels,
    # which is its index in regionprops.
    valid_labels = np.setdiff1d(np.unique(mask), 0)
    reg_error_labels = np.arange(1, mask.max() + 1)[error_labels]
    reg_error_labels = np.intersect1d(reg_error_labels, valid_labels)

    reg_error_labels_indices = np.hstack(
        [np.arange(len(valid_labels))[valid_labels == ll] for ll in reg_error_labels]
    )

    mask_new = mask.copy()
    mask_regionprops = skmeasure.regionprops(mask)
    for lab in reg_error_labels_indices:
        coords = mask_regionprops[lab].coords
        mask_new[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
    return mask_new, flow_errors, labels_gradients


def remove_bad_flow_masks_3D_edt_from2D(
    mask,
    flow,
    flow_threshold=0.6,
    fixed_point_percentile=0.01,
    n_processes=4,
    power_dist=None,
    alpha=0.5,
    filter_scale=1,
):
    """As remove_bad_flow_masks_3D, but re-deriving the flows from 2D Euclidean distance transforms."""
    if not _check_flow_shape(mask, flow):
        return

    view_gradients = []
    for view_index, (view, view_axes, _, _) in enumerate(_VIEWS):
        if view_index == 0:
            labels = mask.copy()
        elif view_index == 1:
            labels = mask.transpose(view_axes)
        else:
            labels = mask.transpose(view_axes) * 1

        flows_dist = distance_transform_labels_fast(labels)
        flows_2d = np.array([np.array(np.gradient(dist_slice)) for dist_slice in flows_dist])
        flows_2d = flows_2d / (np.linalg.norm(flows_2d, axis=1)[:, None] + 1e-20)

        view_gradients.append(_view_gradients_to_zyx(flows_2d, view_index).copy())
        print("done xy " if view_index == 0 else "done " + view)
        del labels

    dz, dy, dx = _fuse_view_gradients(*view_gradients, filter_scale)

    labels_gradients = _normalize_vectors(_stack_zyx(dz, dy, dx))
    del dx, dy, dz

    print(labels_gradients.shape)

    flow_errors = _flow_errors(labels_gradients, flow, mask)
    mask_out = _remove_error_labels_by_position(mask, flow_errors, flow_threshold)
    return mask_out, flow_errors, labels_gradients


def remove_bad_flow_masks_3D_edt(
    mask,
    flow,
    flow_threshold=0.6,
    dtform_method="cellpose_improve",
    fixed_point_percentile=0.01,
    n_processes=4,
    power_dist=None,
    alpha=0.5,
    filter_scale=1,
):
    """As remove_bad_flow_masks_3D, but re-deriving the flows from the 3D Euclidean distance transform."""
    if not _check_flow_shape(mask, flow):
        return

    if dtform_method == "cellpose":
        print("ERROR: not yet implemented.")
        return

    edt_3D = edt.edt(mask, black_border=True, order="C", parallel=n_processes).astype(np.float32)
    labels_gradients = _normalize_vectors(np.array(np.gradient(edt_3D)))

    flow_errors = _flow_errors(labels_gradients, flow, mask)
    mask_out = _remove_error_labels_by_position(mask, flow_errors, flow_threshold)
    return mask_out, flow_errors, labels_gradients


def _pool_view_gradients(labels, dtform_method, fixed_point_percentile, power_dist, n_processes):
    """Per-slice 2D flows (N,2,A,B) of a label stack, computed with a process pool."""
    guide = distance_transform_labels_fast(labels, n_threads=n_processes)

    def _compute_cellflow_view(slice_id):
        return distance_centroid_tform_flow_labels2D(
            labels[slice_id],
            dtform_method=dtform_method,
            guide_img=guide[slice_id],
            fixed_point_percentile=fixed_point_percentile,
            power_dist=power_dist,
        )

    with mp.Pool(n_processes) as pool:
        gradient = np.array(pool.map(_compute_cellflow_view, range(0, len(labels))))

    return gradient[:, 1:]


def remove_bad_flow_masks_3D_mp(
    mask,
    flow,
    flow_threshold=0.6,
    dtform_method="cellpose_improve",
    fixed_point_percentile=0.01,
    n_processes=4,
    power_dist=None,
    alpha=0.5,
    filter_scale=1,
):
    """As remove_bad_flow_masks_3D, using a process pool instead of dask.

    Note: sets the multiprocess start method to 'spawn', so can only be called once per session.
    """
    mp.set_start_method("spawn")

    if not _check_flow_shape(mask, flow):
        return

    if dtform_method == "cellpose":
        print("ERROR: not yet implemented.")
        return

    view_gradients = []
    for view_index, (view, view_axes, _, _) in enumerate(_VIEWS):
        labels = mask.transpose(view_axes).copy()
        print("starting " + view)
        gradient = _pool_view_gradients(
            labels, dtform_method, fixed_point_percentile, power_dist, n_processes
        )
        view_gradients.append(_view_gradients_to_zyx(gradient, view_index))
        print("done xy " if view_index == 0 else "done " + view)
        del labels

    dz, dy, dx = _fuse_view_gradients(*view_gradients, filter_scale)

    labels_gradients = _normalize_vectors(_stack_zyx(dz, dy, dx))
    del dx, dy, dz

    flow_errors = _flow_errors(labels_gradients, flow, mask)
    mask_out = _remove_error_labels_by_position(mask, flow_errors, flow_threshold)
    return mask_out, flow_errors, labels_gradients
