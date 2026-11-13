# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:35:23 2023

@author: fyz11

High-level u-Segment3D pipeline:

1. ``preprocess_imgs``: rescale, illumination-correct and normalize the volume.
2. ``Cellpose2D_model_auto``: run Cellpose slice-by-slice in one view (xy, xz or yz).
3. ``aggregate_2D_to_3D_segmentation_direct_method`` / ``..._indirect_method``:
   fuse the 2D views into one 3D foreground and one 3D gradient field, then
   group voxels into cells by gradient descent.
4. ``postprocess_3D_cell_segmentation``, ``label_diffuse_3D_cell_segmentation_MP``,
   ``guided_filter_3D_cell_segmentation_MP``: optional clean-up and refinement.
"""

import gc
import os

import multiprocess as mp
import numpy as np
import scipy.io as spio
import scipy.ndimage as ndimage
import skimage.filters as skfilters
import skimage.measure as skmeasure
import skimage.morphology as skmorph

from segment3D import file_io as usegment3D_fio
from segment3D import filters as usegment3D_filters
from segment3D import flows as usegment3D_flows
from segment3D import gpu as usegment3D_gpu
from segment3D import segmentation as usegment3D_segment
from segment3D import watershed as usegment3D_watershed

# For each view (0: xy, 1: xz, 2: yz), a 2D gradient has 2 channels. After prepending a
# zero channel, this index order puts the channels into (dz, dy, dx) order.
_VIEW_GRADIENT_CHANNEL_ORDER = ([0, 1, 2], [1, 0, 2], [1, 2, 0])

# Axis order that maps a view's (2, N, A, B) gradient stack back to (2, Z, Y, X).
_VIEW_TO_ZYX_AXES = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 2, 3, 1))


# =============================================================================
# 1. Preprocessing
# =============================================================================
def _rescale_channels(rescale_fn, imgs, zoom, **kwargs):
    """Median-fill zero pixels and rescale each channel."""
    return np.array(
        [
            rescale_fn(usegment3D_filters.fill_array(img, method="median"), zoom, **kwargs)
            for img in imgs
        ],
        dtype=np.float32,
    )


def _normalize_channels(imgs, pmin, pmax, bg_fn=None, **bg_kwargs):
    """Optionally background-correct each channel with bg_fn, then percentile-normalize."""
    return np.array(
        [
            usegment3D_gpu.num_normalize(
                img if bg_fn is None else bg_fn(img, **bg_kwargs), pmin=pmin, pmax=pmax, clip=True
            )
            for img in imgs
        ]
    )


def preprocess_imgs(img, params):
    r"""wrapper function for running preprocessing steps.

    Parameters
    ----------
    img : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. preprocessing will be applied to each channel separately.
    params : Python dict
        Dictionary of parameters as specified by segment3D.parameters.get_preprocess_params

    Returns
    -------
    imgs_out : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. preprocessing will be applied to each channel separately. If do_img_avg == True in the parameters, channels will be averaged to return a single (M,N,L) volume image

    """
    # check dimensions
    img_dim = len(img.shape)

    if img_dim > 3:
        imgs_out = img.copy()
    elif img_dim == 3:
        imgs_out = [img]
    else:
        print(
            "error: Image must be a grayscale volume image of 3 spatial dimensions, or be 4D array where the first dimension, is the number of images or channels."
        )

    # 1. resize and fill in gaps with median. Try CUDA, then torch, then CPU.
    zoom = [params["factor"] * p for p in params["voxel_res"]]

    try:
        imgs_out = _rescale_channels(
            usegment3D_gpu.dask_cuda_rescale, imgs_out, zoom, order=1, mode="reflect"
        )
    except Exception as e:
        try:
            print("no CUDA. trying torch for resizing", "| error:", repr(e))
            imgs_out = _rescale_channels(usegment3D_gpu.zoom_3d_pytorch, imgs_out, zoom)
        except Exception as e:
            print("no gpu. falling back to CPU for resizing", "| error:", repr(e))
            imgs_out = _rescale_channels(
                usegment3D_gpu.dask_cpu_rescale, imgs_out, zoom, order=1, mode="reflect"
            )

    # 2. background correct illumination, then normalize intensities.
    pmin = params["normalize_min"]
    pmax = params["normalize_max"]
    bg_kwargs = {"bg_ds": params["bg_ds"], "bg_sigma": params["bg_sigma"]}

    if params["do_bg_correction"]:
        try:
            # gpu.bg_normalize uses cucim which somehow needs module load cuda.
            imgs_out = _normalize_channels(
                imgs_out, pmin, pmax, bg_fn=usegment3D_gpu.bg_normalize, **bg_kwargs
            )
        except Exception as e:
            try:
                print("no CUDA. trying torch for normalizing", "| error:", repr(e))
                imgs_out = _normalize_channels(
                    imgs_out, pmin, pmax, bg_fn=usegment3D_gpu.bg_normalize_torch, **bg_kwargs
                )
            except Exception as e:
                print("no gpu. falling back to CPU for normalizing", "| error:", repr(e))
                imgs_out = _normalize_channels(
                    imgs_out, pmin, pmax, bg_fn=usegment3D_gpu.bg_normalize_cpu, **bg_kwargs
                )
    else:
        try:
            imgs_out = _normalize_channels(imgs_out, pmin, pmax)
        except Exception as e:
            print("no gpu. falling back to CPU for normalizing", "| error:", repr(e))
            imgs_out = _normalize_channels(imgs_out, pmin, pmax)

    # 3. combine channels if needed.
    if params["do_avg_imgs"]:
        avg_func = params["avg_func_imgs"]
        imgs_out = avg_func(imgs_out, axis=0)

    return imgs_out


# =============================================================================
# 2. Cellpose 2D segmentation of one view
# =============================================================================
def Cellpose2D_model_auto(img, view, params, basename=None, savefolder=None):
    r"""wrapper function for running Cellpose2D with diameter autotuning.

    Parameters
    ----------
    img : numpy array
        (M,N,L) grayscale 3D volume image or (C,M,N,L) multichannel 3D volumes where C=# of channels. This image should be provided in the same order as the input i.e. NO TRANSPOSE needed
    view : one of 'xy', 'xz', or 'yz'
        specifies the view.
    params : Python dict
        Dictionary of parameters as specified by segment3D.parameters.get_preprocess_params

    Returns
    -------
    auto_diam : Python dict
        (diam_range, diam_score, best_diam, test_slice
    all_probs : numpy array
        Cellpose cell probability (logits), transposed back to the input (Z,Y,X) orientation.
    all_flows : numpy array
        Cellpose 2D flows, shape (2,Z,Y,X), transposed back to the input orientation.
    all_styles : numpy array
        Cellpose style vectors, one per slice.

    """
    import cellpose
    from cellpose import models

    if int(cellpose.version.split(".")[0]) < 4:
        model = models.Cellpose(model_type=params["cellpose_modelname"], gpu=params["gpu"])
    else:
        model = models.CellposeModel(
            pretrained_model=params["cellpose_modelname"], gpu=params["gpu"]
        )

    grayscale_bool = params["cellpose_channels"] == "grayscale"
    if grayscale_bool:
        channels = [0, 0]
    else:
        channels = [2, 1]  # cytoplasm = 'Green' and nuclei='Red' channel.

    # bring the requested view's slicing axis to the front.
    if view == "xy":
        order = (0, 1, 2, 3)
    elif view == "xz":
        order = (1, 0, 2, 3)
    elif view == "yz":
        order = (2, 0, 1, 3)
    else:
        print("error: view parameter must be one of the strings, 'xy', 'xz', or 'yz' ")

    if grayscale_bool:
        input_im = img.transpose(order)[..., 0]
    else:
        input_im = img.transpose(order)

    auto_diam, (all_probs, all_flows, all_styles) = usegment3D_segment.apply_cellpose_model_2D_prob(
        input_im,
        model,
        model_channels=channels,
        best_diam=params["best_diam"],
        use_Cellpose_auto_diameter=params["use_Cellpose_auto_diameter"],
        model_invert=params["model_invert"],
        test_slice=params["test_slice"],
        hist_norm=params["hist_norm"],
        diam_range=params["diam_range"],
        ksize=params["ksize"],
        smoothwinsize=params["smoothwinsize"],
        kernel_size=params["histnorm_kernel_size"],
        clip_limit=params["histnorm_clip_limit"],
        use_edge=params["use_edge"],
        show_img_bounds=params["show_img_bounds"],
        saveplotsfolder=params["saveplotsfolder"],
        use_prob_weighted_score=params["use_prob_weighted_score"],
        debug_viz=params["debug_viz"],
    )

    all_probs = np.squeeze(all_probs).astype(np.float32)
    all_flows = np.squeeze(all_flows).astype(np.float32)

    # (N, 2, A, B) -> (2, N, A, B), then transpose the view back to (Z, Y, X).
    all_flows = all_flows.transpose(1, 0, 2, 3)

    if view == "xz":
        all_probs = all_probs.transpose(1, 0, 2)
        all_flows = all_flows.transpose(0, 2, 1, 3)
    if view == "yz":
        all_probs = all_probs.transpose(1, 2, 0)
        all_flows = all_flows.transpose(0, 2, 3, 1)

    if (savefolder is not None) and (basename is not None):
        usegment3D_fio.write_pickle(
            os.path.join(savefolder, basename + "_cellpose_flows_%s.pkl" % (str(view))),
            {"flow": all_flows.astype(np.float32)},
        )
        usegment3D_fio.write_pickle(
            os.path.join(savefolder, basename + "_cellpose_probs_%s.pkl" % (str(view))),
            {"prob": all_probs.astype(np.float32)},
        )
        spio.savemat(
            os.path.join(savefolder, basename + "_cellpose_diam-styles_%s.mat" % (str(view))),
            {
                "style": all_styles,
                "diam_range": auto_diam,
                "diam_score": auto_diam[1],
                "best_diam": auto_diam[2],
            },
            do_compression=True,
        )

    return auto_diam, all_probs, all_flows, all_styles


# =============================================================================
# 3. Shared steps of the 2D-to-3D aggregation (direct and indirect methods)
# =============================================================================
def _threshold_combined_probability(mask_combine, params):
    """Binarize the combined 3D probability with the manual or automatic (multi-Otsu) threshold."""
    prob_params = params["combine_cell_probs"]

    if prob_params["prob_thresh"] is not None:
        return mask_combine >= prob_params["prob_thresh"]

    prob_thresh_raw = skfilters.threshold_multiotsu(
        mask_combine, prob_params["threshold_n_levels"]
    )[prob_params["threshold_level"]]

    # additive offset on the automatic threshold (continuous in the Otsu value, unlike the legacy 1 d.p. floor).
    prob_thresh_offset = prob_params.get("prob_thresh_offset", 0.0)
    prob_thresh = prob_thresh_raw + prob_thresh_offset

    if prob_params["apply_one_d_p_thresh"] > 0:
        # legacy: floor to 1 decimal point.
        prob_thresh = int(prob_thresh * 10) / 10.0

    # enforce a minimum since statistically we can be skewed ...
    prob_thresh = np.maximum(prob_params["min_prob_thresh"], prob_thresh)
    print(
        "Automatic binary thresholding: ",
        prob_thresh,
        "(multi-Otsu: %.4f, offset: %s, 1 d.p. floor: %s)"
        % (prob_thresh_raw, prob_thresh_offset, bool(prob_params["apply_one_d_p_thresh"])),
    )

    return mask_combine >= prob_thresh


def _postprocess_binary(mask_binary, params):
    """Morphological clean-up of the thresholded 3D foreground."""
    post = params["postprocess_binary"]

    if post["binary_closing"] > 0:
        mask_binary = skmorph.binary_closing(mask_binary, skmorph.ball(post["binary_closing"]))
    if post["remove_small_objects"] > 0:
        mask_binary = skmorph.remove_small_objects(
            mask_binary, min_size=post["remove_small_objects"], connectivity=2
        )
    if post["binary_dilation"] > 0:
        mask_binary = skmorph.binary_dilation(mask_binary, skmorph.ball(post["binary_dilation"]))
    if post["binary_fill_holes"] > 0:
        mask_binary = ndimage.binary_fill_holes(mask_binary)
    if post["extra_erode"] > 0:
        mask_binary = skmorph.binary_erosion(mask_binary, skmorph.ball(post["extra_erode"]))

    return mask_binary


def _binarize_combined_probability(mask_combine, params):
    """Threshold then clean up the combined 3D probability."""
    return _postprocess_binary(_threshold_combined_probability(mask_combine, params), params)


def _view_gradient_to_zyx(gradient, view_index, shape):
    """Turn a view's 2-channel gradient (already in Z,Y,X orientation) into 3 channels (dz,dy,dx).

    The axis the view does not see gets a zero channel.
    """
    gradient = np.concatenate([np.zeros(shape, dtype=np.float32)[None, ...], gradient], axis=0)
    return gradient[_VIEW_GRADIENT_CHANNEL_ORDER[view_index], ...].copy()


def _fuse_two_views(first, second, gradient_params, shape, paired_sigma=None):
    """Fuse one gradient component seen by two views. Either view may be missing (None).

    Each view is Gaussian smoothed then combined with an inverse local-variance weighting.
    ``paired_sigma`` overrides the smoothing sigma when both views are present.
    """
    sigma = gradient_params["smooth_sigma"]
    if paired_sigma is None:
        paired_sigma = sigma

    if first is not None and second is not None:
        return usegment3D_filters.var_combine(
            [
                ndimage.gaussian_filter(first, sigma=paired_sigma),
                ndimage.gaussian_filter(second, sigma=paired_sigma),
            ],
            ksize=gradient_params["ksize"],
            alpha=gradient_params["alpha"],
        )
    if second is not None:
        return ndimage.gaussian_filter(second, sigma=sigma)
    if first is not None:
        return ndimage.gaussian_filter(first, sigma=sigma)
    return np.zeros(shape, dtype=np.float32)


def _smooth_stack_and_normalize(dz, dy, dx):
    """Smooth each component, stack as (3,Z,Y,X) and normalize to unit vectors."""
    dx = ndimage.gaussian_filter(dx, sigma=1.0)
    dy = ndimage.gaussian_filter(dy, sigma=1.0)
    dz = ndimage.gaussian_filter(dz, sigma=1.0)
    gradients = np.concatenate([dz[None, ...], dy[None, ...], dx[None, ...]], axis=0)
    return gradients / (np.linalg.norm(gradients, axis=0)[None, ...] + 1e-20)


def _combine_view_gradients(view_gradients, params, shape):
    """Fuse the (dz,dy,dx) gradients of the xy, xz and yz views into one 3D unit-vector field.

    view_gradients : list of 3 arrays of shape (3,Z,Y,X), or [] for a missing view.
    Returns (Z,Y,X,3) float32, the layout expected by the gradient descent.
    """
    xy, xz, yz = [g if len(g) > 0 else None for g in view_gradients]

    def component(gradient, channel):
        return None if gradient is None else gradient[channel]

    gradient_params = params["combine_cell_gradients"]
    dx = _fuse_two_views(component(xy, 2), component(xz, 2), gradient_params, shape)
    dy = _fuse_two_views(component(xy, 1), component(yz, 1), gradient_params, shape)
    # note: dz uses a fixed sigma of 1 (not smooth_sigma) when both views are present.
    dz = _fuse_two_views(component(xz, 0), component(yz, 0), gradient_params, shape, paired_sigma=1)

    labels_gradients = _smooth_stack_and_normalize(dz, dy, dx)
    return labels_gradients.transpose(1, 2, 3, 0).astype(np.float32)


def _save_combined_outputs(savefolder, basename, mask_binary, labels_gradients, mask_combine):
    if (savefolder is not None) and (basename is not None):
        usegment3D_fio.write_pickle(
            os.path.join(savefolder, basename + "_combined2D_3D_probs_gradients_outputs.pkl"),
            {"binary": mask_binary, "flow": labels_gradients, "prob": mask_combine},
        )


def _save_aggregated_segmentation(savefolder, basename, labels_3D):
    if (savefolder is not None) and (basename is not None):
        # note: both the direct and indirect methods save under this '_direct_method' name.
        usegment3D_fio.save_segmentation(
            os.path.join(
                savefolder, basename + "_2D-to-3D_aggregated_u-Segment3D_direct_method.tif"
            ),
            labels_3D,
        )


def _gradient_descent_segment(mask_binary, labels_gradients, params):
    """Group foreground voxels into cells by following the 3D gradients (serial or tiled MP)."""
    gd = params["gradient_descent"]
    cc = params["connected_component"]

    common_kwargs = {
        "gradient_img": labels_gradients,
        "momenta": gd["momenta"],
        "divergence_rescale": False,
        "smooth_sigma": cc["smooth_sigma"],  # use a larger value for weird shapes.
        "smooth_gradient": 1.0,  # this has no effect unless divergence_rescale=True
        "delta": gd["delta"],
        "n_iter": gd["n_iter"],
        "gradient_decay": gd["gradient_decay"],
        "interp": gd["interp"],
        "binary_mask_gradient": gd["binary_mask_gradient"],
        "eps": gd["eps"],
        "min_area": cc["min_area"],
        "thresh_factor": cc["thresh_factor"],
        "mask": None,
        "use_connectivity": gd["use_connectivity"],
        "connectivity_alpha": gd["connectivity_alpha"],
        "renorm_gradient": gd["renorm_gradient"],
    }

    if gd["do_mp"]:
        labels_3D = usegment3D_watershed.mp_gradient_watershed3D_binary(
            mask_binary > 0,
            tile_shape=gd["tile_shape"],
            tile_aspect=gd["tile_aspect"],
            tile_overlap_ratio=gd["tile_overlap_ratio"],
            debug_viz=False,  # can't visualize.
            **common_kwargs,
        )[0]
    else:
        labels_3D = usegment3D_watershed.gradient_watershed3D_binary(
            mask_binary > 0,
            debug_viz=gd["debug_viz"],
            sampling=gd["sampling"],
            track_percent=gd["track_percent"],
            rand_seed=gd["rand_seed"],
            ref_initial_color_img=gd["ref_initial_color_img"],
            ref_alpha=gd["ref_alpha"],
            saveplotsfolder=gd["saveplotsfolder"],
            viewinit=gd["viewinit"],
            **common_kwargs,
        )[0]

    return labels_3D


def _unknown_probability(shape):
    """Placeholder probability (all NaN) used when a precomputed binary is supplied."""
    mask_combine = np.zeros(shape, dtype=np.float32)
    mask_combine[:] = np.nan
    return mask_combine


# =============================================================================
# 3a. Direct method: Cellpose probabilities + gradients as input
# =============================================================================
def aggregate_2D_to_3D_segmentation_direct_method(
    probs,
    gradients,
    params,
    precombined_binary=None,  # this is to allow for using externally computed / combined versions.
    precombined_gradients=None,
    savefolder=None,
    basename=None,
):
    r"""Aggregate per-view 2D cell probabilities and gradients into a 3D instance segmentation.

    Parameters
    ----------
    probs : list of 3
        (Z,Y,X) cell probability logits for the xy, xz and yz views ([] for a missing view).
    gradients : list of 3
        (2,Z,Y,X) 2D gradients for the xy, xz and yz views ([] for a missing view).
    params : dict
        as given by segment3D.parameters.get_2D_to_3D_aggregation_params
    precombined_binary : (Z,Y,X) array, optional
        use this foreground instead of combining ``probs``.
    precombined_gradients : (3,Z,Y,X) or (Z,Y,X,3) array, optional
        use these 3D gradients instead of combining ``gradients``.

    Returns
    -------
    labels_3D : (Z,Y,X) integer array
    (mask_combine, labels_gradients) : the combined probability and the (Z,Y,X,3) gradients.
    """
    assert len(probs) == 3
    assert len(gradients) == 3

    # 1. combine the per-view probabilities into a 3D foreground.
    if precombined_binary is None:
        masks = [np.clip(mask, -88.72, 88.72) for mask in probs if len(mask) > 0]

        if params["combine_cell_probs"]["cellpose_prob_mask"]:
            masks = [1.0 / (1.0 + np.exp(-mask)) for mask in masks]  # sigmoid

        mask_combine = usegment3D_filters.var_combine(
            masks,
            ksize=params["combine_cell_probs"]["ksize"],
            alpha=params["combine_cell_probs"]["alpha"],
            eps=params["combine_cell_probs"]["eps"],
        )
        del masks

        if params["combine_cell_probs"]["smooth_sigma"] > 0:
            mask_combine = ndimage.gaussian_filter(
                mask_combine, sigma=params["combine_cell_probs"]["smooth_sigma"]
            )

        mask_binary = _binarize_combined_probability(mask_combine, params)
    else:
        mask_binary = precombined_binary.copy()
        mask_combine = _unknown_probability(mask_binary.shape)

    # 2. combine the per-view 2D gradients into 3D gradients.
    im_shape = mask_binary.shape

    if precombined_gradients is None:
        view_gradients = [
            _view_gradient_to_zyx(gradient, view_index, im_shape) if len(gradient) > 0 else []
            for view_index, gradient in enumerate(gradients)
        ]
        labels_gradients = _combine_view_gradients(view_gradients, params, im_shape)
        del view_gradients
    else:
        labels_gradients = precombined_gradients.copy()

        if len(labels_gradients) == 3:  # (3,Z,Y,X)
            labels_gradients = labels_gradients / (
                np.linalg.norm(labels_gradients, axis=0)[None, ...] + 1e-20
            )
            labels_gradients = labels_gradients.transpose(1, 2, 3, 0).astype(np.float32)
        else:  # (Z,Y,X,3)
            labels_gradients = labels_gradients / (
                np.linalg.norm(labels_gradients, axis=-1)[..., None] + 1e-20
            )

    _save_combined_outputs(savefolder, basename, mask_binary, labels_gradients, mask_combine)

    # 3. gradient descent to group voxels into cells.
    labels_3D_watershed = _gradient_descent_segment(mask_binary, labels_gradients, params)

    _save_aggregated_segmentation(savefolder, basename, labels_3D_watershed)

    return labels_3D_watershed, (mask_combine, labels_gradients)


# =============================================================================
# 3b. Indirect method: 2D instance segmentation masks as input
# =============================================================================
def _view_shape(img_xy_shape, view_index):
    """Shape of the volume as seen (slice axis first) from view 0: xy, 1: xz or 2: yz."""
    if view_index == 0:
        return img_xy_shape
    if view_index == 1:
        return np.hstack(img_xy_shape)[[1, 0, 2]]
    return np.hstack(img_xy_shape)[[2, 0, 1]]


def _label_gradients_2D(mask, params):
    """Per-slice 2D gradients of a view's label stack, as (2, N, A, B)."""
    indirect = params["indirect_method"]

    if indirect["dtform_method"] == "edt":
        # euclidean distance transform: just take its gradient.
        gradient = usegment3D_flows.distance_transform_labels_fast(mask * 1)
        gradient = np.array(
            [np.array(np.gradient(dist_slice)) for dist_slice in gradient]
        ).transpose(1, 0, 2, 3)
        return gradient / (np.linalg.norm(gradient, axis=0)[None, ...] + 1e-20)

    skel_guide_image = usegment3D_flows.distance_transform_labels_fast(mask * 1)

    n_processes = indirect["n_cpu"]
    if n_processes is None:
        n_processes = (mp.cpu_count() - 1) // 2

    gradient = usegment3D_flows.distance_centroid_tform_flow_labels2D_dask(
        mask * 1,
        dtform_method=indirect["dtform_method"],
        guide_image=skel_guide_image,
        fixed_point_percentile=indirect["edt_fixed_point_percentile"],
        n_processes=n_processes,
        iter_factor=indirect["iter_factor"],
        smooth_skel_sigma=indirect["smooth_skel_sigma"],
        power_dist=indirect["power_dist"],
    )
    # drop the distance channel, keep the 2 gradient channels.
    return gradient[:, 1:].transpose(1, 0, 2, 3).copy()


def aggregate_2D_to_3D_segmentation_indirect_method(
    segmentations,
    params,
    img_xy_shape,
    precomputed_binary=None,  # this is to allow for using externally computed / combined versions.
    savefolder=None,
    basename=None,
):
    r"""Aggregate per-view 2D instance segmentations into a 3D instance segmentation.

    Gradients are computed from each 2D label slice with the distance transform given by
    ``params['indirect_method']['dtform_method']``; the rest is as the direct method.

    Parameters
    ----------
    segmentations : list of 3
        2D-slice label stacks for the xy (Z,Y,X), xz (Y,Z,X) and yz (X,Z,Y) views ([] for a missing view).
    params : dict
        as given by segment3D.parameters.get_2D_to_3D_aggregation_params
    img_xy_shape : tuple
        (Z,Y,X) shape of the volume.
    precomputed_binary : (Z,Y,X) array, optional
        use this foreground instead of combining the segmentations.

    Returns
    -------
    labels_3D : (Z,Y,X) integer array
    (mask_combine, labels_gradients) : the combined foreground probability and the (Z,Y,X,3) gradients.
    """
    assert len(segmentations) == 3

    # missing views are treated as empty segmentations.
    masks = [
        mask if len(mask) > 0 else np.zeros(_view_shape(img_xy_shape, view_index), dtype=np.uint16)
        for view_index, mask in enumerate(segmentations)
    ]

    # 1. combine the per-view foregrounds into a 3D foreground.
    if precomputed_binary is None:
        sigma = params["indirect_method"]["smooth_binary"]
        binaries = [
            (masks[0] > 0) * 1.0,
            (masks[1] > 0).transpose(1, 0, 2) * 1.0,
            (masks[2] > 0).transpose(1, 2, 0) * 1.0,
        ]
        if sigma > 0:
            binaries = [ndimage.gaussian_filter(binary, sigma=sigma) for binary in binaries]

        mask_combine = usegment3D_filters.var_combine(
            binaries,
            ksize=params["combine_cell_probs"]["ksize"],
            alpha=params["combine_cell_probs"]["alpha"],
            eps=params["combine_cell_probs"]["eps"],
        )
        del binaries

        # rescale to 0-1 (to compensate for having only partial views).
        mask_combine = np.clip(mask_combine / (np.max(mask_combine) + 1e-12), 0, 1)

        if params["combine_cell_probs"]["smooth_sigma"] > 0:
            mask_combine = ndimage.gaussian_filter(
                mask_combine, sigma=params["combine_cell_probs"]["smooth_sigma"]
            )

        mask_binary = _binarize_combined_probability(mask_combine, params)
    else:
        mask_binary = precomputed_binary.copy()
        mask_combine = _unknown_probability(mask_binary.shape)

    # 2. compute 2D gradients from the label masks and combine into 3D gradients.
    view_gradients = []
    for view_index, mask in enumerate(masks):
        if mask.max() > 0:
            gradient = _label_gradients_2D(mask, params)
            gradient = gradient.transpose(_VIEW_TO_ZYX_AXES[view_index]).astype(np.float32)
            view_gradients.append(_view_gradient_to_zyx(gradient, view_index, img_xy_shape))
        else:
            view_gradients.append([])  # empty view: nothing to transpose and flip.

    labels_gradients = _combine_view_gradients(view_gradients, params, img_xy_shape)
    del view_gradients

    _save_combined_outputs(savefolder, basename, mask_binary, labels_gradients, mask_combine)

    # 3. gradient descent to group voxels into cells (identical to the direct method).
    labels_3D_watershed = _gradient_descent_segment(mask_binary, labels_gradients, params)

    _save_aggregated_segmentation(savefolder, basename, labels_3D_watershed)

    return labels_3D_watershed, (mask_combine, labels_gradients)


# =============================================================================
# 4. Postprocessing
# =============================================================================
def _remove_large_labels(segmentation, max_size=None, max_size_factor=None):
    """Zero out (in place) labels larger than max_size, or larger than mean + max_size_factor * std."""
    uniq_labels = np.setdiff1d(np.unique(segmentation), 0)
    regions = skmeasure.regionprops(segmentation)
    areas = np.hstack([re.area for re in regions])

    if max_size is not None:
        area_thresh = max_size
    else:
        area_thresh = np.nanmean(areas) + max_size_factor * np.nanstd(areas)

    if np.sum(areas > area_thresh):
        for rr in np.arange(len(uniq_labels))[areas > area_thresh]:
            coords = regions[rr].coords
            segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] = 0


def _consistency_slice_gradients(stack, flow_params, n_cpu):
    """Per-slice 2D gradients (N, 2, A, B) of a label stack, for the flow consistency check."""
    dtform_method = flow_params["dtform_method"]

    if dtform_method == "edt":
        gradient = usegment3D_flows.distance_transform_labels_fast(stack * 1, n_threads=n_cpu)
        gradient = np.array([np.array(np.gradient(dist_slice)) for dist_slice in gradient])
        return gradient / (np.linalg.norm(gradient, axis=1)[:, None] + 1e-20)

    guide = usegment3D_flows.distance_transform_labels_fast(stack, n_threads=n_cpu)

    def _compute_cellflow_view(slice_id):
        return usegment3D_flows.distance_centroid_tform_flow_labels2D(
            stack[slice_id],
            dtform_method=dtform_method,
            guide_img=guide[slice_id],
            fixed_point_percentile=flow_params["edt_fixed_point_percentile"],
            iter_factor=flow_params["iter_factor"],
            smooth_skel_sigma=flow_params["smooth_skel_sigma"],
            power_dist=flow_params["power_dist"],
        )

    with mp.Pool(n_cpu) as pool:
        gradient = pool.map(_compute_cellflow_view, range(0, len(stack)))
    # drop the distance channel, keep the 2 gradient channels.
    return np.array(gradient, dtype=np.float32)[:, 1:]


def _reconstruct_cell_gradients(segmentation, aggregation_params, flow_params, n_cpu):
    """Recompute 3D (3,Z,Y,X) unit gradients from a 3D segmentation, as the indirect method would."""
    dtform_method = flow_params["dtform_method"]

    view_gradients = []
    for view_index, (view, view_axes, _, _) in enumerate(usegment3D_flows._VIEWS):
        stack = segmentation.transpose(view_axes).copy()
        if dtform_method != "edt":
            print("start " + view)
        gradient = _consistency_slice_gradients(stack, flow_params, n_cpu)
        view_gradients.append(usegment3D_flows._view_gradients_to_zyx(gradient, view_index))
        print("finish " + view)
        del stack

    gc.collect()

    gradient_xy, gradient_xz, gradient_yz = view_gradients
    gradient_params = aggregation_params["combine_cell_gradients"]
    shape = segmentation.shape
    dx = _fuse_two_views(gradient_xy[:, 2], gradient_xz[:, 2], gradient_params, shape)
    dy = _fuse_two_views(gradient_xy[:, 1], gradient_yz[:, 1], gradient_params, shape)
    dz = _fuse_two_views(gradient_xz[:, 0], gradient_yz[:, 0], gradient_params, shape)

    return _smooth_stack_and_normalize(dz, dy, dx)


def postprocess_3D_cell_segmentation(
    segmentation,
    aggregation_params,
    postprocess_params,
    cell_gradients=None,
    savefolder=None,
    basename=None,
):
    r"""Clean up a 3D segmentation.

    1. size filters (minimum size, optional maximum size), keeping the largest component per label.
    2. flow consistency: remove cells whose re-derived gradients disagree with ``cell_gradients``.
    3. statistical size filter (remove cells larger than mean + max_size_factor * std).

    Parameters
    ----------
    segmentation : (Z,Y,X) integer array
    aggregation_params : dict
        the parameters used for the aggregation (get_2D_to_3D_aggregation_params).
    postprocess_params : dict
        as given by segment3D.parameters.get_postprocess_segmentation_params
    cell_gradients : (Z,Y,X,3) array, optional
        the combined 3D gradients used to create the segmentation. Required for step 2.

    Returns
    -------
    segmentation_filt : (Z,Y,X) integer array
    flow_consistency_outputs : list
        [flow_errors, reconstructed gradients (Z,Y,X,3)] if any cells were removed in step 2.
    """
    from cellpose.utils import fill_holes_and_remove_small_masks

    params = postprocess_params
    flow_consistency_outputs = []

    # 1. size filters.
    segmentation_filt = segmentation.copy()

    if params["size_filters"]["min_size"] > 0:
        segmentation_filt = fill_holes_and_remove_small_masks(
            segmentation, min_size=params["size_filters"]["min_size"]
        )
    if params["size_filters"]["max_size"] is not None:
        _remove_large_labels(segmentation_filt, max_size=params["size_filters"]["max_size"])

    segmentation_filt = usegment3D_filters.largest_component_vol_labels_fast(
        segmentation_filt, connectivity=1
    )

    # 2. flow consistency filter.
    if params["flow_consistency"]["do_flow_remove"] and cell_gradients is not None:
        n_cpu = params["flow_consistency"]["n_cpu"]
        if n_cpu is None:
            n_cpu = np.maximum(mp.cpu_count() - 1, 1)

        dzyx = _reconstruct_cell_gradients(
            segmentation_filt, aggregation_params, params["flow_consistency"], n_cpu
        )

        # mean squared error between the reconstructed and given gradients, per label.
        n_labels = segmentation_filt.max()
        flow_errors = usegment3D_flows._flow_errors(
            dzyx, cell_gradients.transpose(3, 0, 1, 2), segmentation_filt
        )

        error_labels = flow_errors > params["flow_consistency"]["flow_threshold"]

        if np.sum(error_labels) > 0:
            valid_labels = np.setdiff1d(np.unique(segmentation_filt), 0)
            reg_error_labels = np.arange(1, n_labels + 1)[error_labels]
            reg_error_labels = np.intersect1d(reg_error_labels, valid_labels)

            # regionprops are in label order, so index them by position in valid_labels.
            reg_error_labels_indices = np.hstack(
                [np.arange(len(valid_labels))[valid_labels == ll] for ll in reg_error_labels]
            )

            mask_new = segmentation_filt.copy()
            mask_regionprops = skmeasure.regionprops(segmentation_filt)

            for lab in reg_error_labels_indices:
                coords = mask_regionprops[lab].coords
                mask_new[coords[:, 0], coords[:, 1], coords[:, 2]] = 0

            segmentation_filt = mask_new.copy()
            # the transpose returns the gradients in the same layout as cell_gradients.
            flow_consistency_outputs.append([flow_errors, dzyx.transpose(1, 2, 3, 0)])

            del mask_new

    # 3. statistical size filter.
    if params["size_filters"]["do_stats_filter"]:
        _remove_large_labels(
            segmentation_filt, max_size_factor=params["size_filters"]["max_size_factor"]
        )

    return segmentation_filt, flow_consistency_outputs


# =============================================================================
# 5. Per-cell refinement: label diffusion and guided filtering
# =============================================================================
def _crop_around(img_pad, bbox, pad_size):
    """Crop a padded image around a (z0,y0,x0,z1,y1,x1) bounding box, extended by pad_size."""
    return img_pad[
        bbox[0] - pad_size : bbox[3] + pad_size,
        bbox[1] - pad_size : bbox[4] + pad_size,
        bbox[2] - pad_size : bbox[5] + pad_size,
    ]


def _assemble_refined_cells(cell_coords, cells_id, segmentation_pad, pad_size, out_shape):
    """Paint each cell's refined (padded-image) coordinates with its label, then remove the padding."""
    segmentation_pad_clean = np.zeros_like(segmentation_pad)

    for coords, cell_id in zip(cell_coords, cells_id):
        if len(coords) > 0:
            segmentation_pad_clean[coords[:, 0], coords[:, 1], coords[:, 2]] = cell_id

    return segmentation_pad_clean[
        pad_size : pad_size + out_shape[0],
        pad_size : pad_size + out_shape[1],
        pad_size : pad_size + out_shape[2],
    ]


def label_diffuse_3D_cell_segmentation_MP(segmentation, guide_image, params):
    r"""Refine each cell's boundary by label diffusion on an affinity graph built from the guide image.

    Each cell is processed in its own (padded) bounding-box crop, in parallel.

    Parameters
    ----------
    segmentation : (Z,Y,X) integer array
    guide_image : (Z,Y,X) array
    params : dict
        as given by segment3D.parameters.get_label_diffusion_params

    Returns
    -------
    (Z,Y,X) integer array
    """
    n_cpu = params["diffusion"]["n_cpu"]
    if n_cpu is None:
        n_cpu = np.maximum(mp.cpu_count() - 1, 1)

    # guide_img must be in range 0, 1, enforce here.
    guide_img = usegment3D_filters.normalize(
        guide_image, pmin=params["guide_img"]["pmin"], pmax=params["guide_img"]["pmax"], clip=True
    )

    refine_clamp = params["diffusion"]["refine_clamp"]
    refine_iters = params["diffusion"]["refine_iters"]
    refine_alpha = params["diffusion"]["refine_alpha"]
    noprogress_bool = params["diffusion"]["noprogress_bool"]
    affinity_type = params["diffusion"]["affinity_type"]
    pad_size = params["diffusion"]["pad_size"]

    padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
    guide_img_pad = np.pad(guide_img, padding, mode="reflect")
    segmentation_pad = np.pad(segmentation, padding, mode="constant") * 1

    cells = skmeasure.regionprops(segmentation_pad)
    cells_id = np.setdiff1d(np.unique(segmentation_pad), 0)  # exclude the background label
    bboxes = [cell.bbox for cell in cells]

    def _refine_single_obj(obj_id):
        bbox = bboxes[obj_id]
        label_bbox = _crop_around(segmentation_pad, bbox, pad_size)
        label_img = _crop_around(guide_img_pad, bbox, pad_size)

        label_bbox_refine = usegment3D_filters.diffuse_labels3D(
            (label_bbox == cells_id[obj_id]) * 1,
            label_img,
            clamp=refine_clamp,
            n_iter=refine_iters,
            alpha=refine_alpha,
            noprogress=noprogress_bool,
            affinity_type=affinity_type,
        )
        label_coords = np.argwhere(label_bbox_refine > 0)

        if len(label_coords) > 0:
            # crop coordinates -> padded image coordinates.
            return label_coords + (np.asarray(bbox[:3]) - pad_size)[None, :]
        return []

    with mp.Pool(n_cpu) as pool:
        res = pool.map(_refine_single_obj, range(0, len(cells)))

    return _assemble_refined_cells(res, cells_id, segmentation_pad, pad_size, segmentation.shape)


def _ridge_enhanced_guide(guide_img, params):
    """Mix the guide image with its (normalized) Meijering ridge filter response."""
    ridge = params["ridge_filter"]

    if ridge["do_multiprocess_2D"]:
        n_cpu_ridge = ridge["n_cpu"]
        if n_cpu_ridge is None:
            n_cpu_ridge = np.maximum(mp.cpu_count() // 2 - 1, 1)

        guide_img_edge = usegment3D_filters.meijering_ridge_filter_2D_multiprocess(
            guide_img,
            sigmas=ridge["sigmas"],
            low_contrast_percentile=ridge["low_contrast_fraction"],
            black_ridges=ridge["black_ridges"],
            n_cpu=n_cpu_ridge,
        )
    else:
        guide_img_edge = skfilters.meijering(
            guide_img, sigmas=ridge["sigmas"], black_ridges=ridge["black_ridges"]
        )

    print(guide_img_edge.shape)

    return (1.0 - ridge["mix_ratio"]) * guide_img + (
        ridge["mix_ratio"]
    ) * usegment3D_filters.normalize(
        guide_img_edge, pmin=ridge["pmin"], pmax=ridge["pmax"], clip=True
    )


def guided_filter_3D_cell_segmentation_MP(segmentation, guide_image, params):
    r"""Refine each cell (e.g. recover thin protrusions) with a guided filter on the guide image.

    Each cell is processed in its own (padded) bounding-box crop, in parallel.

    Parameters
    ----------
    segmentation : (Z,Y,X) integer array
    guide_image : (Z,Y,X) array
    params : dict
        as given by segment3D.parameters.get_guided_filter_params

    Returns
    -------
    segmentation_refined : (Z,Y,X) integer array
    guide_img : the normalized (and optionally ridge-enhanced) guide image used.
    """
    gf = params["guide_filter"]

    n_cpu = gf["n_cpu"]
    if n_cpu is None:
        n_cpu = np.maximum(mp.cpu_count() - 1, 1)

    # guide_img must be in range 0, 1, enforce here.
    guide_img = usegment3D_filters.normalize(
        guide_image, pmin=params["guide_img"]["pmin"], pmax=params["guide_img"]["pmax"], clip=True
    )

    if params["ridge_filter"]["do_ridge_enhance"]:
        guide_img = _ridge_enhanced_guide(guide_img, params)

    pad_size = gf["pad_size"]
    # scales the guided filter radius by this * mean bounding box diameter; increase to recover longer protrusions.
    size_factor = gf["size_factor"]
    use_adaptive_radius = gf["adaptive_radius_bool"]
    min_protrusion_size = gf["min_protrusion_size"]  # the minimum adaptive radius.
    mode = gf["mode"]  # 'additive' is useful when cells are hollow (membrane marker).
    base_dilate = gf["base_dilate"]
    # eroding the input (additive mode) helps if it is too fat, but too much may leave it hollow.
    base_erode = gf["base_erode"]
    guide_filter_eps = gf["eps"]  # lower for more effect without changing radius.
    guide_radius = gf["radius"]
    threshold_level = gf["threshold_level"]
    n_thresholds = gf["threshold_nlevels"]
    use_int = gf["use_int"]

    padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
    guide_img_pad = np.pad(guide_img, padding, mode="reflect")
    segmentation_pad = np.pad(segmentation, padding, mode="constant")

    cells = skmeasure.regionprops(segmentation_pad)
    cells_id = np.setdiff1d(np.unique(segmentation_pad), 0)  # exclude the background label
    bboxes = [cell.bbox for cell in cells]

    def _refine_single_obj_guidedfilter(obj_id):
        bbox = bboxes[obj_id]
        label_bbox = _crop_around(segmentation_pad, bbox, pad_size)
        label_img = _crop_around(guide_img_pad, bbox, pad_size)
        is_cell = label_bbox == cells_id[obj_id]

        label_binary = is_cell * 1 if use_int else is_cell * 1.0

        if use_adaptive_radius:  # generally better for many heterogeneous cells.
            mean_size = np.prod(label_bbox.shape) ** (1.0 / 3)
            radius = np.maximum(min_protrusion_size, mean_size * size_factor)
        else:
            radius = guide_radius

        label_bbox_refine_guided = usegment3D_filters.guidedfilter(
            label_binary, label_img, radius=radius, eps=guide_filter_eps
        )

        # rebinarize and keep the largest component.
        label_bbox_refine = (
            label_bbox_refine_guided
            >= skfilters.threshold_multiotsu(label_bbox_refine_guided, n_thresholds)[
                threshold_level
            ]
        )
        label_bbox_refine = usegment3D_filters.largest_component_vol(label_bbox_refine)

        if mode == "additive":
            # add the (optionally dilated or eroded) input cell. Other modes replace it.
            base = is_cell.copy()
            if base_dilate > 0:
                base = skmorph.binary_dilation(base, skmorph.ball(base_dilate))
            elif base_erode > 0:
                base = skmorph.binary_erosion(base, skmorph.ball(base_erode))
            label_bbox_refine = np.logical_or(label_bbox_refine, base)

        if gf["collision_erode"] > 0:
            # don't grow into other cells.
            label_bbox_refine = (
                np.logical_or(
                    is_cell,
                    skmorph.binary_erosion(label_bbox == 0, skmorph.ball(gf["collision_erode"])),
                )
                * label_bbox_refine
            )
        if gf["collision_close"] > 0:
            label_bbox_refine = skmorph.binary_closing(
                label_bbox_refine, skmorph.ball(gf["collision_close"])
            )
        if gf["collision_dilate"] > 0:
            label_bbox_refine = skmorph.binary_dilation(
                label_bbox_refine, skmorph.ball(gf["collision_dilate"])
            )
        if gf["collision_fill_holes"] > 0:
            label_bbox_refine = ndimage.binary_fill_holes(label_bbox_refine)

        label_bbox_refine = usegment3D_filters.largest_component_vol(label_bbox_refine)

        label_coords = np.argwhere(label_bbox_refine > 0)
        if len(label_coords) > 0:
            # crop coordinates -> padded image coordinates.
            label_coords = label_coords + (np.asarray(bbox[:3]) - pad_size)[None, :]

        return label_coords

    with mp.Pool(n_cpu) as pool:
        res = pool.map(_refine_single_obj_guidedfilter, range(0, len(cells)))

    segmentation_refined = _assemble_refined_cells(
        res, cells_id, segmentation_pad, pad_size, segmentation.shape
    )
    return segmentation_refined, guide_img
