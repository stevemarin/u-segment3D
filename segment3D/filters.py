# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper filters.
"""

import warnings

import cc3d
import multiprocess as mp
import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as spsparse
import skimage.exposure as skexposure
import skimage.filters as skfilters
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.segmentation as sksegmentation
import skimage.transform as sktform
from tqdm import tqdm


# =============================================================================
# intensity normalization
# =============================================================================
def fill_array(im, thresh=0, fill_vals=0, method="constant"):
    """Replace values <= thresh by fill_vals ('constant'), or the median / mean of the values above thresh."""
    out = im.copy()

    if method == "constant":
        out[im <= thresh] = fill_vals
    if method == "median":
        out[im <= thresh] = np.nanmedian(im[im > thresh])
    if method == "mean":
        out[im <= thresh] = np.nanmean(im[im > thresh])

    return out


def imadjust(vol, p1, p2):
    """Contrast stretch the intensities between the p1 and p2 percentiles."""
    p1_, p2_ = np.percentile(vol, (p1, p2))
    return skexposure.rescale_intensity(vol, in_range=(p1_, p2_))


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)
    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0), sigma=0, option=1, ploton=False):
    """
    Anisotropic diffusion in 2D

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)  # used as-is if option is not 1 or 2
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation="nearest")
        ih = ax2.imshow(imgout, interpolation="nearest", animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    # note: runs niter - 1 iterations.
    for ii in np.arange(1, niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if 0 < sigma:
            deltaSf = ndimage.gaussian_filter(deltaS, sigma)
            deltaEf = ndimage.gaussian_filter(deltaE, sigma)
        else:
            deltaSf = deltaS
            deltaEf = deltaE

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-((deltaSf / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaEf / kappa) ** 2.0)) / step[1]
        elif option == 2:
            gS = 1.0 / (1.0 + (deltaSf / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaEf / kappa) ** 2.0) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()

    return imgout


def meijering_ridge_filter_2D_multiprocess(
    stack, sigmas, low_contrast_percentile=0.05, black_ridges=False, n_cpu=4
):
    """Slice-by-slice Meijering ridge filter (in parallel). Low-contrast slices give zeros."""
    stack_in = normalize(stack, pmin=0, pmax=100, clip=True)  # ensure 0-1

    def _run_filter(slice_id):
        im_slice = stack_in[slice_id]
        if skexposure.is_low_contrast(im_slice, fraction_threshold=low_contrast_percentile):
            return np.zeros_like(im_slice)
        return skfilters.meijering(im_slice, sigmas=sigmas, black_ridges=black_ridges)

    with mp.Pool(n_cpu) as pool:
        res = pool.map(_run_filter, range(0, len(stack_in)))

    return np.array(res)


# =============================================================================
# for cleaning 2D-slice labels
# =============================================================================
def relabel_slices(labelled, bg_label=0):
    """Offset each slice's labels by the running maximum so labels are unique across slices."""
    max_ID = 0
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        lab_slice[lab_slice > bg_label] = (
            lab[lab > bg_label] + max_ID
        )  # only update the foreground!
        labelled_.append(lab_slice)
        max_ID += np.max(lab_slice)

    return np.array(labelled_)


def connected_component_slices(labelled, bg_label=0):
    """Relabel each slice's foreground by its connected components, unique across slices."""
    max_ID = 0
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        lab_slice_label = skmeasure.label(lab_slice)
        lab_slice[lab_slice > bg_label] = lab_slice_label[lab_slice_label > 0] + max_ID
        labelled_.append(lab_slice)
        max_ID += np.max(lab_slice_label)

    return np.array(labelled_)


def filter_2d_label_slices(labelled, bg_label=0, minsize=10):
    """Per slice: split labels into connected components, drop those <= minsize, relabel 1..N."""
    labelled_ = []
    for lab in labelled:
        lab_slice = lab.copy()
        uniq_labels = np.setdiff1d(np.unique(lab_slice), bg_label)

        max_ID = 0
        label_slice_new = np.zeros_like(lab_slice)

        for ll in uniq_labels:
            lab_binary = skmeasure.label((lab_slice == ll) > 0)

            for component in np.setdiff1d(np.unique(lab_binary), 0):
                if np.sum(lab_binary == component) > minsize:
                    label_slice_new[lab_binary == component] = max_ID + 1
                    max_ID += 1

        labelled_.append(label_slice_new)

    return np.array(labelled_)


def detect_and_relabel_multi_component_labels(labels, min_size=100):
    """Split 3D labels made of several connected pieces into separate labels (pieces < min_size are dropped)."""
    labels_filt = np.zeros_like(labels)

    max_id = 1

    for re in skmeasure.regionprops(labels):
        bbox = re.bbox
        coords = re.coords

        x1, y1, z1 = bbox[:3]
        bbox_size = np.hstack(bbox[3:]) - np.hstack(bbox[:3])

        mask = np.zeros(bbox_size, dtype=bool)
        mask[coords[:, 0] - x1, coords[:, 1] - y1, coords[:, 2] - z1] = 1

        mask = skmorph.remove_small_objects(mask, min_size=min_size)
        label_mask = skmeasure.label(mask > 0)

        for ll in np.setdiff1d(np.unique(label_mask), 0):
            coords_ll = np.argwhere(label_mask == ll) + np.hstack(bbox[:3])

            labels_filt[coords_ll[:, 0], coords_ll[:, 1], coords_ll[:, 2]] = max_id
            max_id += 1

    return labels_filt


def filter_segmentations_axis(labels, window=3, min_count=1):
    """according to the persistence along 1 direction. - this works and is relatively fast.

    A label in slice i is kept if it overlaps (>= min_count pixels) labels in a long enough
    contiguous run of the reflect-padded slices i, ..., i + 2 * (window // 2) - 1.
    """
    labels_filt = []
    offset = window // 2

    labels_pad = np.pad(labels, [[offset, offset], [0, 0], [0, 0]], mode="reflect")

    for ss_ii in np.arange(len(labels)):
        ss = labels_pad[ss_ii + offset].copy()
        ss_copy = ss.copy()

        for reg in np.setdiff1d(np.unique(ss), 0):
            mask = ss == reg

            # does the region overlap something in each neighbouring slice?
            checks = np.zeros(2 * offset)
            for cnt in range(2 * offset):
                mask_ii = labels_pad[ss_ii + cnt, mask > 0].copy()
                if np.sum(mask_ii > 0) >= min_count:
                    checks[cnt] += 1

            # check contiguity!
            contigs = skmeasure.label(checks)

            if contigs.max() > 0:
                uniq_contig = np.setdiff1d(np.unique(contigs), 0)
                lengths = [np.sum(contigs == cc) for cc in uniq_contig]
                mask = contigs == uniq_contig[np.argmax(lengths)]

                if np.sum(mask) >= window // 2 + 1:
                    # sufficiently long, but must cover one of the two middle neighbours.
                    if mask[offset - 1] == 0 and mask[offset] == 0:
                        ss_copy[ss == reg] = 0
                else:
                    ss_copy[ss == reg] = 0
            else:
                ss_copy[ss == reg] = 0

        labels_filt.append(ss_copy)

    return np.array(labels_filt)


def filter_segmentations_axis_IoU(labels, window=3, iou_var=0.1):
    """according to the persistence along 1 direction of the IoU with respect to the middle reference shape.

    A label in slice i is removed if the std of its IoU with the largest overlapping label in
    the reflect-padded slices i, ..., i + 2 * (window // 2) - 1 is > iou_var.
    """
    labels_filt = []
    offset = window // 2

    labels_pad = np.pad(labels, [[offset, offset], [0, 0], [0, 0]], mode="reflect")

    for ss_ii in np.arange(len(labels)):
        ss = labels_pad[ss_ii + offset].copy()
        ss_copy = ss.copy()

        for reg in np.setdiff1d(np.unique(ss), 0):
            mask = ss == reg

            iou_checks = np.zeros(2 * offset)
            for cnt in range(2 * offset):
                neighbour = labels_pad[ss_ii + cnt]
                mask_ii = labels_pad[ss_ii + cnt, mask > 0].copy()

                if mask_ii.max() > 0:
                    # IoU with the largest overlapping region.
                    uniq_regions_mask_ii = np.setdiff1d(np.unique(mask_ii), 0)
                    largest = uniq_regions_mask_ii[
                        np.argmax([np.sum(neighbour == rr) for rr in uniq_regions_mask_ii])
                    ]
                    largest_mask = neighbour == largest
                    overlap = np.sum(np.logical_and(largest_mask, mask))
                    iou_checks[cnt] = overlap / (np.sum(mask) + np.sum(largest_mask) - overlap)
                else:
                    iou_checks[cnt] = 0

            if np.std(iou_checks) > iou_var:
                ss_copy[ss == reg] = 0

        labels_filt.append(ss_copy)

    return np.array(labels_filt)


# =============================================================================
# smoothing
# =============================================================================
def smooth_vol(vol_binary, ds=4, smooth=5):
    """Fast Gaussian smoothing: downsample by ds, smooth, then upsample back."""
    small = sktform.resize(vol_binary, np.array(vol_binary.shape) // ds, preserve_range=True)
    small = ndimage.gaussian_filter(small, sigma=smooth)

    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


def smooth_vol_anisotropic(vol_binary, ds=4, smooth=[5, 5, 5]):
    """As smooth_vol, with a separate smoothing sigma per axis."""
    small = sktform.resize(vol_binary, np.array(vol_binary.shape) // ds, preserve_range=True)
    for axis in np.arange(len(smooth)):
        small = ndimage.gaussian_filter1d(small, sigma=smooth[axis], axis=axis)

    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


# =============================================================================
# for cleaning labels
# =============================================================================
def remove_small_labels(labels, min_size=64):
    """Set labels with fewer than min_size pixels to background."""
    if labels.max() == 0:
        return labels

    uniq_regions = np.setdiff1d(np.unique(labels), 0)
    areas = np.hstack([re.area for re in skmeasure.regionprops(labels)])
    regions_remove = uniq_regions[areas < min_size]

    labels_new = labels.copy()
    for reg in regions_remove:
        labels_new[labels == reg] = 0

    return labels_new


def intensity_filter_labels(labels, intensity_img, threshold=None, auto_thresh_method="Otsu"):
    """Remove 3D labels whose mean intensity is below threshold (automatic if None: 'Otsu', 'Mean' or 'Median')."""
    labels_filt = labels.copy()
    regionprops = skmeasure.regionprops(labels)

    all_region_I = []
    all_region_I_inds = []

    for ind, reg in enumerate(regionprops):
        coords = reg.coords
        all_region_I.append(np.nanmean(intensity_img[coords[:, 0], coords[:, 1], coords[:, 2]]))
        all_region_I_inds.append(ind)

    if len(all_region_I) > 0:
        all_region_I = np.hstack(all_region_I)
        all_region_I_inds = np.hstack(all_region_I_inds)

        if threshold is None:
            if auto_thresh_method == "Mean":
                threshold = np.nanmean(all_region_I)
            elif auto_thresh_method == "Median":
                threshold = np.nanmedian(all_region_I)
            else:  # 'Otsu' (the default) and any other value
                threshold = skfilters.threshold_otsu(all_region_I)

        for rr in all_region_I_inds[all_region_I < threshold]:
            coords = regionprops[rr].coords
            labels_filt[coords[:, 0], coords[:, 1], coords[:, 2]] = 0

    return labels_filt


def expand_masks(label_seeds, binary, dist_tform=None):
    """Grow the seed labels to fill the binary, by watershed on the distance transform."""
    if dist_tform is None:
        from .flows import distance_transform_labels_fast

        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2))

    seeds = label_seeds * binary

    # note: parts of the binary not reached from a seed are left unlabelled.
    return sksegmentation.watershed(-dist_tform, seeds, mask=binary)


def expand_masks2D(label_seeds, binary, dist_tform=None):
    """As expand_masks but slice-by-slice; unreached parts get new labels (offset by label_seeds.max())."""
    if dist_tform is None:
        from .flows import distance_transform_labels_fast

        # this is running 2D slice by slice!.
        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2))

    seeds = label_seeds * binary

    labels_refine = []

    for dd in np.arange(len(binary)):
        labels_refine_slice = sksegmentation.watershed(-dist_tform[dd], seeds[dd], mask=binary[dd])

        # the remainder we label as new objects.
        remainder = np.logical_and(binary[dd] > 0, labels_refine_slice == 0)
        remainder_label = skmeasure.label(remainder)
        remainder_label[remainder > 0] = remainder_label[remainder > 0] + label_seeds.max()

        labels_refine_slice[remainder > 0] = remainder_label[remainder > 0].copy()
        labels_refine.append(labels_refine_slice)

    return np.array(labels_refine)


def remove_eccentric_shapes(labels, min_size=20, stretch_cutoff=15):
    """Remove labels <= min_size pixels or whose covariance eigenvalue ratio exceeds stretch_cutoff.

    Returns the filtered labels and (areas, stretch ratios) of every label.
    """
    labels_filter = labels.copy()

    obj_area = []
    obj_ecc = []

    for lab in np.setdiff1d(np.unique(labels), 0):
        mask = labels == lab
        obj_area.append(np.nansum(mask))

        pts_mask = np.vstack(np.argwhere(mask > 0))

        if len(pts_mask) > min_size:
            cov = np.cov((pts_mask - pts_mask.mean(axis=0)[None, :]).T)
            eigs = np.linalg.eigvalsh(cov)

            stretch_ratio = np.max(np.max(np.abs(eigs)) / np.abs(eigs))

            if stretch_ratio > stretch_cutoff:
                labels_filter[mask] = 0

            obj_ecc.append(stretch_ratio)
        else:
            obj_ecc.append(np.nan)
            labels_filter[mask] = 0

    return labels_filter, (obj_area, obj_ecc)


# =============================================================================
# for fusion
# =============================================================================
def entropy_mean(arrays, ksize=3, alpha=0.5, eps=1e-20):
    """Average the arrays weighted by 1 / (local entropy + alpha).

    Note: uses the ``selem`` keyword, which recent scikit-image versions no longer accept.
    """
    import skimage.filters.rank as skfilters_rank

    ents = np.array(
        [
            1.0 / (skfilters_rank.entropy(array, selem=skmorph.ball(ksize)) + alpha)
            for array in arrays
        ]
    )

    return np.sum(ents * arrays, axis=0) / (ents.sum(axis=0) + eps)


def var_filter(img, ksize):
    """Local variance in a ksize window."""
    win_mean = ndimage.uniform_filter(img, ksize)
    win_sqr_mean = ndimage.uniform_filter(img**2, ksize)
    return win_sqr_mean - win_mean**2


def fast_dog_filter(img, ksize, norm=True):
    """Difference of box filters (size 1 and ksize), optionally z-score normalized."""
    out = ndimage.uniform_filter(img, size=1) - ndimage.uniform_filter(img, size=ksize)

    if norm:
        out = (out - np.mean(out)) / np.std(out)

    return out


def var_combine(arrays, ksize=3, alpha=1.0, eps=1e-20):
    """Average the arrays weighted by 1 / (local variance + alpha): smoother inputs count more."""
    weights = np.array([1.0 / (var_filter(array, ksize=ksize) + alpha) for array in arrays])

    return np.sum(weights * arrays, axis=0) / (weights.sum(axis=0) + eps)


def dog_combine(arrays, ksize=3, eps=1e-20):
    """Average the arrays weighted by their (clipped) difference-of-box-filter response."""
    weights = np.array(
        [np.clip(fast_dog_filter(array, ksize=ksize, norm=True), 0, 4) for array in arrays]
    )

    return np.sum(weights * arrays, axis=0) / (weights.sum(axis=0) + eps)


# =============================================================================
# 3D guided filter
# =============================================================================
def guidedfilter(I, G, radius, eps):
    """n-dimensional guided filter.

    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    # step 1
    meanI = ndimage.uniform_filter(G, size=radius)
    meanp = ndimage.uniform_filter(I, size=radius)
    corrI = ndimage.uniform_filter(G * G, size=radius)
    corrIp = ndimage.uniform_filter(I * G, size=radius)

    # step 2
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanp

    # step 3
    a = covIp / (varI + eps)
    b = meanp - a * meanI
    # step 4 (note: this uses size=eps, not radius, as in the original implementation)
    meana = ndimage.uniform_filter(a, size=eps)
    meanb = ndimage.uniform_filter(b, size=eps)
    # step 5
    return meana * G + meanb


def largest_component_vol(vol_binary, connectivity=1):
    r"""Given a binary segmentation, return only the largest connected component of the given connectivity

    connectivity 1: 4-neighbors (2D) / 6-neighbors (3D); 2: 8-neighbors (2D) / 26-neighbors (3D).
    """
    vol_binary_labelled = skmeasure.label(vol_binary, connectivity=connectivity)
    vol_binary_vols = [re.area for re in skmeasure.regionprops(vol_binary_labelled)]

    return vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])


def largest_component_vol_labels(vol_labels, connectivity=1):
    """Keep only the largest connected component of each label."""
    vol_labels_new = np.zeros_like(vol_labels)

    for lab in np.setdiff1d(np.unique(vol_labels), 0):
        mask = largest_component_vol(vol_labels == lab, connectivity=connectivity)
        vol_labels_new[mask > 0] = lab

    return vol_labels_new


def largest_component_vol_labels_fast(vol_labels, connectivity=1):
    """Keep only the largest connected component of each 3D label (computed in its bounding box).

    Labels are renumbered 1..N in regionprops order. connectivity 1 uses 18-, 2 uses 26-connectivity.
    """
    vol_labels_clean = np.zeros_like(vol_labels)

    regions = skmeasure.regionprops(vol_labels)
    print(len(regions))

    for reg_ii, reg in enumerate(regions):
        bbox = reg.bbox
        bbox_binary = reg.image
        if connectivity == 1:
            bbox_label = cc3d.connected_components(bbox_binary, connectivity=18)
            bbox_label, N = cc3d.largest_k(bbox_label, k=1, connectivity=18, return_N=True)
        if connectivity == 2:
            bbox_label = cc3d.connected_components(bbox_binary, connectivity=26)
            bbox_label, N = cc3d.largest_k(bbox_label, k=1, connectivity=26, return_N=True)

        coords = np.argwhere(bbox_label > 0)

        if len(coords) > 0:
            coords = coords + bbox[:3]
            vol_labels_clean[coords[:, 0], coords[:, 1], coords[:, 2]] = reg_ii + 1

    return vol_labels_clean


# =============================================================================
# label diffusion
# =============================================================================
def _distance_to_affinity_matrix(Dmatrix, gamma, kernel):
    """Apply kernel(distances, sigma) to a sparse distance matrix and set the diagonal to 1.

    sigma is the mean distance, or gamma if provided.
    """
    l = Dmatrix.shape[0]  # noqa: E741
    A = Dmatrix.copy()
    sigma_D = np.mean(A.data) if gamma is None else gamma
    np.exp(kernel(A.data, sigma_D), out=A.data)
    A = A + spsparse.diags(np.ones(l), 0)  # diagonal is 1 by definition.

    return A.tocsr()


def _distance_to_heat_affinity_matrix(Dmatrix, gamma=None):
    r"""Convert any distance matrix to an affinity matrix by applying a heat kernel.

    .. math::
        A = \exp^{\left(\frac{-D^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.

    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    return _distance_to_affinity_matrix(Dmatrix, gamma, lambda d, sigma: -(d**2) / (2 * (sigma**2)))


def _distance_to_laplace_affinity_matrix(Dmatrix, gamma=None):
    r"""Convert any distance matrix to an affinity matrix by applying a Laplace kernel.

    .. math::
        A = \exp^{\left(\frac{-|D|}{\sigma}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.

    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    return _distance_to_affinity_matrix(Dmatrix, gamma, lambda d, sigma: -np.abs(d) / sigma)


def _diffuse_labels(
    labels_in, guide, ndim, clamp, n_iter, noprogress, alpha, affinity_type, reinitialize_freq
):
    """Label propagation on a graph mixing guide-intensity affinity (weight alpha) and spatial adjacency."""
    from sklearn.feature_extraction.image import grid_to_graph, img_to_graph

    # fwd: original -> new, bwd: new -> original
    labels_in_relabel, fwd, bwd = sksegmentation.relabel_sequential(labels_in)

    graph = img_to_graph(guide)  # intensity differences between neighbours
    if affinity_type == "heat":
        affinity = _distance_to_heat_affinity_matrix(graph, gamma=None)
    elif affinity_type == "laplace":
        affinity = _distance_to_laplace_affinity_matrix(graph, gamma=None)
    else:
        print("not valid")

    graph_laplacian = grid_to_graph(*guide.shape[:ndim])  # spatial adjacency
    if affinity_type == "heat":
        affinity_laplacian = _distance_to_heat_affinity_matrix(graph_laplacian * 1.0, gamma=None)
    elif affinity_type == "laplace":
        affinity_laplacian = _distance_to_laplace_affinity_matrix(graph_laplacian * 1.0, gamma=None)
    else:
        print("not valid")

    affinity = alpha * affinity + (1.0 - alpha) * affinity_laplacian

    # one-hot labels, including background.
    n_labels = np.max(labels_in_relabel) + 1
    labels = np.zeros((np.prod(labels_in_relabel.shape[:ndim]), n_labels), dtype=np.float32)
    labels[np.arange(len(labels_in_relabel.ravel())), labels_in_relabel.ravel()] = 1

    init_matrix = np.zeros_like(labels)

    for ii in tqdm(np.arange(n_iter), disable=noprogress):
        init_matrix = (1.0 - clamp) * affinity.dot(init_matrix) + clamp * labels

        if reinitialize_freq is not None:  # rescale, as it can saturate.
            init_matrix = init_matrix / np.nanmean(init_matrix)

    # most likely label per pixel.
    z = np.nansum(init_matrix, axis=1)
    z[z == 0] += 1  # Avoid division by 0
    z = ((init_matrix.T) / z).T
    z_label = np.argmax(z, axis=1)
    z_label = z_label.reshape(labels_in_relabel.shape)

    # map back.
    return bwd[z_label]


def diffuse_labels2D(
    labels_in,
    guide,
    clamp=0.99,
    n_iter=10,
    noprogress=True,
    alpha=0.8,
    affinity_type="heat",
    reinitialize_freq=None,
):
    """Refine 2D labels by label propagation guided by the guide image.

    Parameters
    ----------
    labels_in : (M,N) integer array
        the labels to refine.
    guide : (M,N) array
        guide image (intensity edges stop the propagation).
    clamp : float
        how strongly the input labels are enforced (1: fully). The default is 0.99.
    n_iter : int
        number of propagation iterations. The default is 10.
    noprogress : bool
        if True, hide the progress bar. The default is True.
    alpha : float
        weight of the guide-intensity affinity versus the spatial adjacency. The default is 0.8.
    affinity_type : 'heat' or 'laplace'
        the kernel turning distances into affinities.
    reinitialize_freq :
        not used in 2D.

    Returns
    -------
    z_label : (M,N) integer array
        the refined labels.

    """
    return _diffuse_labels(
        labels_in, guide, 2, clamp, n_iter, noprogress, alpha, affinity_type, reinitialize_freq=None
    )


def diffuse_labels3D(
    labels_in,
    guide,
    clamp=0.99,
    n_iter=10,
    noprogress=True,
    alpha=0.8,
    affinity_type="heat",
    reinitialize_freq=None,
):
    """Refine 3D labels by label propagation guided by the guide image.

    Parameters are as for diffuse_labels2D. If reinitialize_freq is not None, the label
    scores are rescaled by their mean after every iteration.

    Returns
    -------
    z_label : (M,N,L) integer array
        the refined labels.

    """
    return _diffuse_labels(
        labels_in, guide, 3, clamp, n_iter, noprogress, alpha, affinity_type, reinitialize_freq
    )
