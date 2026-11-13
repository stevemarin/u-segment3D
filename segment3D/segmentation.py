# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper functions for applying several standard DL segmentation models.
"""

import os

import numpy as np
import skimage.exposure as skexposure
import skimage.filters as skfilters
import skimage.measure as skmeasure
from tqdm import tqdm

from .filters import var_filter


def reorder_labels_2D(labels, bg=0):
    """Renumber labels 1..N in (row, column) order of their centroids."""
    regprops = skmeasure.regionprops(labels)

    reg_labels = np.setdiff1d(np.unique(labels), bg)
    centroids = np.vstack([re.centroid for re in regprops])

    ind = np.lexsort((centroids[:, 1], centroids[:, 0]))
    labels_out = np.zeros_like(labels)

    for ii, lab in enumerate(ind):
        labels_out[labels == reg_labels[lab]] = ii + 1

    return labels_out


def reorder_labels_3D(labels, bg=0):
    """Renumber labels 1..N in (z, y, x) order of their centroids."""
    regprops = skmeasure.regionprops(labels)

    reg_labels = np.setdiff1d(np.unique(labels), bg)
    centroids = np.vstack([re.centroid for re in regprops])

    ind = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    labels_out = np.zeros_like(labels)

    for ii, lab in enumerate(ind):
        labels_out[labels == reg_labels[lab]] = ii + 1

    return labels_out


def _BackgroundRemoval(im):
    # given an image, utilises the pixels to subtract the background in the image
    # remembering images are represented by a matrix with x axis = cols, y axis = rows,
    # The algorithm carries out the following steps:
    # 1) normalize the input image
    # 2) fits a 2nd order background model
    # 3) subtracts this from the input image, then makes everything positive to be a proper image,
    # 4) renormalises the image,
    nrows, ncols = im.shape
    N = nrows * ncols

    xx, yy = np.meshgrid(np.arange(0, ncols), np.arange(0, nrows))
    xx = xx.ravel()
    yy = yy.ravel()

    X = np.column_stack((np.ones(N), xx, yy, np.square(xx), np.multiply(xx, yy), np.square(yy)))
    p = np.linalg.lstsq(X, im.ravel())[0]  # X\im(:)

    imgInit = (im.ravel() - np.dot(X, p)).reshape(nrows, ncols)

    if np.min(imgInit) == np.max(imgInit):
        imgInit = np.zeros(im.shape)
    else:
        imgInit = (imgInit - np.min(imgInit)) / (np.max(imgInit) - np.min(imgInit))

    # imgInit is the subtracted background (flattened image), normalised to between [0,1]
    return imgInit


def _ma_average(sig, winsize=3, mode="reflect", avg_func=np.nanmean):
    """Moving average with a (padded) window of winsize."""
    sig_ = np.pad(sig, [winsize // 2, winsize // 2], mode=mode)
    sig_out = []
    for ii in np.arange(len(sig)):
        sig_out.append(avg_func(sig_[ii : ii + winsize]))
    return np.hstack(sig_out)


def _histeq_low_contrast(im, kernel_size=(256, 256), clip_limit=0.01, fraction_threshold=0.1):
    """Adaptive histogram equalization (GPU if available), unless the image is low contrast."""
    if skexposure.is_low_contrast(im, fraction_threshold):
        return im
    else:
        try:
            from .gpu import cuda_equalize_adapthist

            return cuda_equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)
        except:  # noqa: E722
            return skexposure.equalize_adapthist(im, kernel_size=kernel_size, clip_limit=clip_limit)


# =============================================================================
# Cellpose helpers
# =============================================================================
def _is_legacy_cellpose():
    """True for Cellpose < 4 (the model is called via model.cp.eval and accepts invert)."""
    import cellpose

    return int(cellpose.version.split(".")[0]) < 4


def _model_eval_fn(model):
    return model.cp.eval if _is_legacy_cellpose() else model.eval


def _cellpose_eval(model_eval, img, model_channels, model_invert, v4_channels=True, **kwargs):
    """Run Cellpose on one image.

    Cellpose < 4 is passed channels and invert; Cellpose >= 4 only channels, if v4_channels.
    """
    if _is_legacy_cellpose():
        return model_eval([img], channels=model_channels, invert=model_invert, **kwargs)
    if v4_channels:
        return model_eval([img], channels=model_channels, **kwargs)
    return model_eval([img], **kwargs)


def _pick_test_slice(im_stack, use_edge):
    """The slice with the most signal (mean intensity, or mean Sobel edge strength if use_edge)."""
    if len(im_stack.shape) > 3:  # multichannel: use the max over channels
        if use_edge:
            signal_stack = [
                np.nanmean(skfilters.sobel(np.max(im_slice, axis=-1))) for im_slice in im_stack
            ]
        else:
            signal_stack = [np.nanmean(np.max(im_slice, axis=-1)) for im_slice in im_stack]
    else:
        if use_edge:
            print("single channel")
            signal_stack = [np.nanmean(skfilters.sobel(im_slice)) for im_slice in im_stack]
        else:
            signal_stack = [np.nanmean(im_slice) for im_slice in im_stack]
    test_slice = np.argmax(signal_stack)
    print("test_slice", test_slice, len(im_stack))
    return test_slice


def _background_removal_channels(img):
    if len(img.shape) > 2:
        return np.dstack([_BackgroundRemoval(img[..., ch]) for ch in np.arange(img.shape[-1])])
    return _BackgroundRemoval(img)


def _prepare_image(img, hist_norm, bg_remove, kernel_size, clip_limit, fraction_threshold):
    """Optional histogram equalization and background removal, then intensity rescaling."""
    if hist_norm:
        img = _histeq_low_contrast(
            img,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            fraction_threshold=fraction_threshold,
        )
        if bg_remove:
            img = _background_removal_channels(img)
        img = skexposure.rescale_intensity(img)
    elif bg_remove:
        print("bg remove")
        if len(img.shape) > 2:
            print(img.shape)
        img = _background_removal_channels(img)
        img = skexposure.rescale_intensity(img)
    return img


def _contrast_score(flow, ksize, use_prob_weighted_score, clip_prob=False):
    """Mean local variance of the predicted flows (optionally weighted by cell probability).

    Returns (score, cell probability).
    """
    prob = flow[0][2]
    if clip_prob:
        prob = np.clip(prob, -88.72, 88.72)
    prob = 1.0 / (1 + np.exp(-prob))

    if use_prob_weighted_score:
        prob_score = np.nanmean(
            var_filter(flow[0][1][0], ksize=ksize) * prob
            + var_filter(flow[0][1][1], ksize=ksize) * prob
        )
    else:
        prob_score = np.nanmean(
            var_filter(flow[0][1][0], ksize=ksize) + var_filter(flow[0][1][1], ksize=ksize)
        )
    return prob_score, prob


def _plot_diameter_response(
    img, flow, prob, diam, prob_score, ksize, show_img_bounds, saveplotsfolder
):
    """Debug figure: image, cell probability, flow variance and flow field for one test diameter."""
    import cellpose.plot
    import pylab as plt

    def _crop(x):
        return x[: show_img_bounds[0], : show_img_bounds[1]] if show_img_bounds is not None else x

    plt.figure(figsize=(10, 10))
    plt.subplot(411)
    plt.title(str(diam) + " " + str(prob_score))
    plt.imshow(_crop(img), cmap="gray")
    plt.subplot(412)
    plt.title(np.mean(flow[0][2]))
    if show_img_bounds is not None:
        plt.imshow(_crop(prob), cmap="magma", vmin=0, vmax=1)
    else:
        plt.imshow(prob, cmap="magma_r", vmin=0, vmax=1)
    plt.subplot(413)  # show the highlighted edges.
    plt.imshow(_crop(var_filter(flow[0][2], ksize=ksize)), cmap="coolwarm")
    plt.subplot(414)  # show the flow field.
    plt.imshow(_crop(cellpose.plot.dx_to_circ(flow[0][1])))

    if saveplotsfolder is not None:
        plt.savefig(
            os.path.join(saveplotsfolder, "diameter_response_%.5f.svg" % (diam)),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show(block=False)
    plt.close("all")  # too many figures is generated therefore force close.


def _run_on_stack(
    im_stack,
    model_eval,
    model_channels,
    model_invert,
    diameter,
    hist_norm,
    kernel_size,
    clip_limit,
    fraction_threshold,
    v4_channels,
):
    """Run Cellpose on every slice. Returns float32 (probs, flows, styles)."""
    all_probs = []
    all_flows = []
    all_styles = []

    for zz in tqdm(np.arange(len(im_stack))):
        img = _prepare_image(
            im_stack[zz].copy(), hist_norm, False, kernel_size, clip_limit, fraction_threshold
        )
        _, flow, style = _cellpose_eval(
            model_eval,
            img,
            model_channels,
            model_invert,
            v4_channels=v4_channels,
            batch_size=32,
            diameter=diameter,
            compute_masks=False,
        )
        all_probs.append(flow[0][2])
        all_flows.append(flow[0][1])
        all_styles.append(style[0])

    return (
        np.array(all_probs, dtype=np.float32),
        np.array(all_flows, dtype=np.float32),
        np.array(all_styles, dtype=np.float32),
    )


# =============================================================================
# Applying Cellpose with automatic diameter selection
# =============================================================================
def apply_cellpose_model_2D_prob_slice(
    im_slice,
    model,
    model_channels,
    best_diam=None,
    model_invert=False,
    test_slice=None,
    diam_range=np.arange(15, 51, 5),
    ksize=25,
    smoothwinsize=5,
    hist_norm=True,
    kernel_size=(256, 256),
    clip_limit=0.01,
    fraction_threshold=0.1,
    n_proc=48,
    bg_remove=False,
    use_edge=False,
    use_prob_weighted_score=True,
):
    """Run Cellpose on a single 2D image, choosing the diameter that maximizes the contrast score.

    Returns (diam_range, diam_score, best_diam), (probs, flows, style).
    """
    model_eval = _model_eval_fn(model)

    if best_diam is None:
        diam_score = []

        for diam in diam_range[:]:
            img = im_slice.copy()
            _, flow, style = _cellpose_eval(
                model_eval,
                img,
                model_channels,
                model_invert,
                batch_size=32,
                do_3D=False,
                flow_threshold=0.6,
                cellprob_threshold=-np.inf,
                diameter=diam,
                compute_masks=False,
            )
            diam_score.append(_contrast_score(flow, ksize, use_prob_weighted_score)[0])

        diam_score = _ma_average(np.hstack(diam_score), winsize=smoothwinsize)
        best_diam = diam_range[np.argmax(diam_score)]

    else:
        diam_score = []

    # note: this uses img from the diameter scan (it is undefined if best_diam was given).
    _, flow, style = _cellpose_eval(
        model_eval,
        img,
        model_channels,
        model_invert,
        batch_size=32,
        diameter=best_diam,
        compute_masks=False,
    )

    (all_probs, all_flows, all_styles) = (flow[0][2], flow[0][1], style[0])

    return (diam_range, diam_score, best_diam), (all_probs, all_flows, all_styles)


def apply_cellpose_model_2D_prob(
    im_stack,
    model,
    model_channels,
    best_diam=None,
    use_Cellpose_auto_diameter=False,
    model_invert=False,
    test_slice=None,
    diam_range=np.arange(15, 51, 5),
    ksize=25,
    smoothwinsize=5,
    hist_norm=True,
    kernel_size=(256, 256),
    clip_limit=0.01,
    fraction_threshold=0.1,
    n_proc=48,
    bg_remove=False,
    use_edge=False,
    show_img_bounds=(1024, 1024),
    debug_viz=False,
    saveplotsfolder=None,
    use_prob_weighted_score=False,
):
    """Run Cellpose slice-by-slice on a stack, with automatic diameter selection.

    If best_diam is None, the diameter is chosen on test_slice (default: the slice with
    the most signal) either by Cellpose's own estimate (use_Cellpose_auto_diameter) or as
    the diameter in diam_range maximizing the (smoothed) contrast score.

    Returns
    -------
    (diam_range, diam_score, best_diam, test_slice), (all_probs, all_flows, all_styles)
    """
    model_eval = _model_eval_fn(model)

    if best_diam is None:
        if test_slice is None:
            test_slice = _pick_test_slice(im_stack, use_edge)

        if use_Cellpose_auto_diameter:
            diam_score = []
            diam_range = []

            img = _prepare_image(
                im_stack[test_slice].copy(),
                hist_norm,
                bg_remove,
                kernel_size,
                clip_limit,
                fraction_threshold,
            )
            _, _, _, best_diam = _cellpose_eval(
                model_eval,
                img,
                model_channels,
                model_invert,
                v4_channels=False,
                batch_size=32,
                do_3D=False,
                flow_threshold=0.6,
                diameter=None,
                compute_masks=False,
            )

        else:
            diam_range = np.hstack(diam_range)
            diam_score = []

            for diam in diam_range[:]:
                img = _prepare_image(
                    im_stack[test_slice].copy(),
                    hist_norm,
                    bg_remove,
                    kernel_size,
                    clip_limit,
                    fraction_threshold,
                )
                _, flow, style = _cellpose_eval(
                    model_eval,
                    img,
                    model_channels,
                    model_invert,
                    v4_channels=False,
                    batch_size=32,
                    do_3D=False,
                    flow_threshold=0.6,
                    diameter=diam,
                    compute_masks=False,
                )

                prob_score, prob = _contrast_score(
                    flow, ksize, use_prob_weighted_score, clip_prob=True
                )
                diam_score.append(prob_score)

                if debug_viz:
                    _plot_diameter_response(
                        img, flow, prob, diam, prob_score, ksize, show_img_bounds, saveplotsfolder
                    )

            diam_score = _ma_average(np.hstack(diam_score), winsize=smoothwinsize)
            best_diam = diam_range[np.argmax(diam_score)]

            if debug_viz:
                import pylab as plt

                plt.figure(figsize=(10, 10))
                plt.plot(diam_range, diam_score, "o-")
                plt.vlines(
                    best_diam,
                    np.min(diam_score),
                    np.max(diam_score),
                    color="k",
                    linestyles="dashed",
                )
                plt.xlabel("Cellpose Diameter [px]")
                plt.ylabel("Contrast Score")
                if saveplotsfolder is not None:
                    plt.savefig(
                        os.path.join(saveplotsfolder, "contrast-score_vs_diameter_plot.svg"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.show(block=False)
                plt.close("all")

    else:
        diam_score = []

    print("auto determine cell diameter: ", best_diam)

    all_probs, all_flows, all_styles = _run_on_stack(
        im_stack,
        model_eval,
        model_channels,
        model_invert,
        best_diam,
        hist_norm,
        kernel_size,
        clip_limit,
        fraction_threshold,
        v4_channels=False,
    )

    return (diam_range, diam_score, best_diam, test_slice), (all_probs, all_flows, all_styles)


def apply_cellpose_model_2D_prob_multiscale(
    im_stack,
    model,
    model_channels,
    best_diam=None,
    model_invert=False,
    test_slice=None,
    diam_range=np.arange(15, 51, 5),
    ksize=25,
    smoothwinsize=5,
    peak_prominence=0.25,
    peak_min_distance=None,
    peak_height=0.1,
    peak_winsize_factor=4,
    hist_norm=True,
    kernel_size=(256, 256),
    clip_limit=0.01,
    fraction_threshold=0.1,
    n_proc=48,
    bg_remove=False,
    use_edge=False,
    show_img_bounds=(1024, 1024),
    include_cellpose_auto_diam=True,
    use_prob_weighted_score=True,
    saveplotsfolder=None,
):
    """As apply_cellpose_model_2D_prob, but running Cellpose at several diameters.

    The diameters are the best contrast score, the peaks of the (baseline-subtracted)
    contrast score and, optionally, Cellpose's own estimate. Always plots the diameter scan.
    Note: best_diam is not used, and giving it raises an error.

    Returns
    -------
    (diam_range, diam_score, best_diams, test_slice), (probs, flows, styles) stacked over the diameters.
    """
    import pylab as plt
    from scipy.signal import find_peaks

    model_eval = _model_eval_fn(model)

    if best_diam is None:
        diam_range = np.hstack(diam_range)

        if test_slice is None:
            test_slice = _pick_test_slice(im_stack, use_edge)

        diam_score = []

        for diam in diam_range[:]:
            img = _prepare_image(
                im_stack[test_slice].copy(),
                hist_norm,
                bg_remove,
                kernel_size,
                clip_limit,
                fraction_threshold,
            )
            _, flow, style = _cellpose_eval(
                model_eval,
                img,
                model_channels,
                model_invert,
                batch_size=32,
                do_3D=False,
                flow_threshold=0.6,
                diameter=diam,
                compute_masks=False,
            )

            prob_score, prob = _contrast_score(flow, ksize, use_prob_weighted_score)
            diam_score.append(prob_score)

            _plot_diameter_response(
                img, flow, prob, diam, prob_score, ksize, show_img_bounds, saveplotsfolder
            )

        diam_score = _ma_average(np.hstack(diam_score), winsize=smoothwinsize)

        # find peaks of the score above its smooth baseline.
        diam_score_base = _ma_average(diam_score, winsize=len(diam_score) // peak_winsize_factor)
        diam_score_ = diam_score - diam_score_base

        plt.figure(figsize=(5, 5))
        plt.plot(diam_score_)
        plt.show(block=False)

        # pad either side to handle boundary conditions.
        diam_score_pad = np.pad(
            diam_score_, pad_width=[smoothwinsize, smoothwinsize], mode="reflect"
        )

        best_diams = find_peaks(
            diam_score_pad,
            prominence=peak_prominence,
            distance=peak_min_distance,
            height=peak_height,
        )

        print(best_diams)

        if len(best_diams[0]) > 0:
            print("testing.")
            peak_pos = best_diams[0] - smoothwinsize
            peak_pos = np.clip(peak_pos, 0, len(diam_score) - 1)
            best_diams = np.hstack([np.argmax(diam_score), peak_pos])
            best_diams = diam_range[best_diams]
            best_diams = np.unique(best_diams)

            if include_cellpose_auto_diam:
                _, _, _, best_cp_diam = _cellpose_eval(
                    model_eval,
                    img,
                    model_channels,
                    model_invert,
                    batch_size=32,
                    do_3D=False,
                    flow_threshold=0.6,
                    diameter=None,
                    compute_masks=False,
                )

                print(best_cp_diam)
                best_diams = np.hstack([best_diams, best_cp_diam])  # in actual dimension space.
        else:
            best_diams = [diam_range[np.argmax(diam_score)]]

        plt.figure()
        plt.plot(diam_range, diam_score, "o-")
        plt.show(block=False)
        plt.close("all")

    else:
        diam_score = []

    print("auto determine cell diameter: ", best_diams)

    all_probs_ms = []
    all_flows_ms = []
    all_styles_ms = []

    for best_diam in best_diams:
        all_probs, all_flows, all_styles = _run_on_stack(
            im_stack,
            model_eval,
            model_channels,
            model_invert,
            best_diam,
            hist_norm,
            kernel_size,
            clip_limit,
            fraction_threshold,
            v4_channels=True,
        )
        all_probs_ms.append(all_probs)
        all_flows_ms.append(all_flows)
        all_styles_ms.append(all_styles)

    return (diam_range, diam_score, best_diams, test_slice), (
        np.array(all_probs_ms),
        np.array(all_flows_ms),
        np.array(all_styles_ms),
    )
