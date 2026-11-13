"""Behaviour snapshot for the segment3D package (normally run via check.py).

Usage: python -I run_cases.py <package_root> <fake_cellpose_dir> <out.pkl> [case_filter]

Runs every case on small synthetic data, recording (stdout, result-or-exception, files
written) so that two versions of the package can be compared bit-for-bit with compare.py.
case_filter: only run cases whose name contains one of the '|' separated strings.
"""

import contextlib
import copy
import hashlib
import io
import os
import pickle
import shutil
import sys
import tempfile
import traceback

PKG_ROOT, FAKE_CP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FILTER = sys.argv[4] if len(sys.argv) > 4 else None
sys.path[:0] = [PKG_ROOT, FAKE_CP]
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import scipy.ndimage as ndimage  # noqa: E402

import cellpose  # noqa: E402  (fake)
from segment3D import file_io, filters, flows, gpu, meshtools, metrics, parameters, plotting  # noqa: E402
from segment3D import segmentation as seg  # noqa: E402
from segment3D import usegment3d as u3d  # noqa: E402
from segment3D import watershed  # noqa: E402

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def each(*thunks):
    out = []
    for t in thunks:
        try:
            out.append(("ok", t()))
        except Exception as e:
            out.append(("exc", type(e).__name__, str(e)))
    return out


# ----------------------------------------------------------------------------- data
def make_cells(shape=(20, 36, 36), n=6, radius=8.0, seed=0):
    rng = np.random.RandomState(seed)
    lo = np.array([4, 7, 7])
    hi = np.array(shape) - lo
    centers = rng.uniform(lo, hi, size=(n, 3))
    zz, yy, xx = np.indices(shape)
    grid = np.stack([zz, yy, xx], -1).astype(np.float64)
    d = np.linalg.norm(grid[..., None, :] - centers[None, None, None], axis=-1)
    lab = np.argmin(d, axis=-1) + 1
    lab[np.min(d, axis=-1) > radius] = 0
    return lab.astype(np.int32), centers


LABELS, CENTERS = make_cells()
SHAPE = LABELS.shape
EDT = ndimage.distance_transform_edt(LABELS > 0)
RNG = np.random.RandomState(1)
IMG = (
    ndimage.gaussian_filter((LABELS > 0) * 1.0 + 0.5 * (EDT > 2), 1.0) + 0.05 * RNG.rand(*SHAPE)
).astype(np.float32)
IMGN = IMG / IMG.max()
GZ, GY, GX = np.gradient(ndimage.gaussian_filter(EDT, 1.5))


def view_gradients(noise=0.1, seed=2):
    r = np.random.RandomState(seed)
    nz = lambda: noise * r.randn(*SHAPE)  # noqa: E731
    return [
        np.array([GY + nz(), GX + nz()], dtype=np.float32),  # xy -> (dy, dx)
        np.array([GZ + nz(), GX + nz()], dtype=np.float32),  # xz -> (dz, dx)
        np.array([GZ + nz(), GY + nz()], dtype=np.float32),  # yz -> (dz, dy)
    ]


def view_probs(seed=3):
    r = np.random.RandomState(seed)
    return [((EDT - 1.5) * 2 + 0.3 * r.randn(*SHAPE)).astype(np.float32) for _ in range(3)]


def view_segmentations():
    return [LABELS.copy(), LABELS.transpose(1, 0, 2).copy(), LABELS.transpose(2, 0, 1).copy()]


def agg_params(**over):
    p = parameters.get_2D_to_3D_aggregation_params()
    p["gradient_descent"]["n_iter"] = 40
    p["postprocess_binary"]["remove_small_objects"] = 50
    p["indirect_method"]["n_cpu"] = 2
    for k, v in over.items():
        a, b = k.split("__")
        p[a][b] = v
    return p


def labels_2d():
    return LABELS[SHAPE[0] // 2].copy()


# ----------------------------------------------------------------------------- usegment3d
@case
def preprocess_bg():
    p = parameters.get_preprocess_params()
    p["bg_ds"] = 4
    return u3d.preprocess_imgs(IMG, p)


@case
def preprocess_nobg_multichannel_avg():
    p = parameters.get_preprocess_params()
    p["do_bg_correction"] = False
    p["do_avg_imgs"] = True
    p["voxel_res"] = [2, 1, 1]
    return u3d.preprocess_imgs(np.stack([IMG, IMG**2]), p)


@case
def preprocess_bad_dims():
    return u3d.preprocess_imgs(IMG[0], parameters.get_preprocess_params())


def _cp_params(**over):
    p = parameters.get_Cellpose_autotune_params()
    p["gpu"] = False
    p["diam_range"] = np.arange(8, 20, 4)
    p["ksize"] = 5
    p["smoothwinsize"] = 3
    p.update(over)
    return p


for _view in ("xy", "xz", "yz"):
    for _over_name, _over in (
        ("auto", {}),
        ("cpauto", {"use_Cellpose_auto_diameter": True, "hist_norm": True}),
        ("fixed", {"best_diam": 12}),
    ):

        def _mk(view=_view, over=_over):
            def f():
                cellpose.CALL_LOG.clear()
                d = tempfile.mkdtemp()
                out = each(
                    lambda: u3d.Cellpose2D_model_auto(
                        IMGN[..., None], view, _cp_params(**over), basename="b", savefolder=d
                    ),
                    lambda: u3d.Cellpose2D_model_auto(IMGN[..., None], view, _cp_params(**over)),
                )
                return out, list(cellpose.CALL_LOG), d

            return f

        _f = _mk()
        _f.__name__ = f"cellpose_auto_{_view}_{_over_name}"
        case(_f)


@case
def cellpose_auto_debugviz_color():
    cellpose.CALL_LOG.clear()
    d = tempfile.mkdtemp()
    p = _cp_params(
        debug_viz=True, saveplotsfolder=d, cellpose_channels="color", use_edge=True, hist_norm=False
    )
    out = u3d.Cellpose2D_model_auto(np.stack([IMGN, IMGN * 0.5, IMGN * 0.25], -1), "xy", p)
    return out, list(cellpose.CALL_LOG), d


@case
def seg_slice_and_multiscale():
    cellpose.CALL_LOG.clear()
    m = cellpose.models.Cellpose()
    a = each(
        lambda: seg.apply_cellpose_model_2D_prob_slice(
            IMGN[5], m, [0, 0], diam_range=np.arange(8, 20, 4), ksize=5, smoothwinsize=3
        )
    )
    b = each(
        lambda: seg.apply_cellpose_model_2D_prob_slice(
            IMGN[5],
            m,
            [0, 0],
            diam_range=np.arange(8, 20, 4),
            ksize=5,
            smoothwinsize=3,
            use_prob_weighted_score=False,
        )
    )
    d = tempfile.mkdtemp()
    c = each(
        lambda: seg.apply_cellpose_model_2D_prob_multiscale(
            IMGN,
            m,
            [0, 0],
            diam_range=np.arange(6, 30, 2),
            ksize=5,
            smoothwinsize=3,
            saveplotsfolder=d,
        )
    )
    e = each(
        lambda: seg.apply_cellpose_model_2D_prob_multiscale(
            IMGN,
            m,
            [0, 0],
            diam_range=np.arange(6, 30, 2),
            ksize=5,
            smoothwinsize=3,
            include_cellpose_auto_diam=False,
            hist_norm=False,
            use_edge=True,
        )
    )
    f = each(lambda: seg.apply_cellpose_model_2D_prob_multiscale(IMGN, m, [0, 0], best_diam=10))
    return a, b, c, e, f, list(cellpose.CALL_LOG), d


@case
def seg_helpers():
    return each(
        lambda: seg.reorder_labels_2D(labels_2d() * 3),
        lambda: seg.reorder_labels_3D(LABELS * 2),
        lambda: seg._BackgroundRemoval(IMGN[5]),
        lambda: seg._ma_average(np.arange(10.0), winsize=3),
        lambda: seg._histeq_low_contrast(IMGN[5], kernel_size=(8, 8)),
        lambda: seg._histeq_low_contrast(np.zeros((10, 10)), kernel_size=(8, 8)),
    )


@case
def direct_all_views():
    d = tempfile.mkdtemp()
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(
        view_probs(), view_gradients(), agg_params(), savefolder=d, basename="x"
    ), d


@case
def direct_options():
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(
        view_probs(),
        view_gradients(),
        agg_params(
            combine_cell_probs__smooth_sigma=1,
            combine_cell_probs__apply_one_d_p_thresh=True,
            combine_cell_probs__cellpose_prob_mask=False,
            postprocess_binary__binary_fill_holes=True,
            postprocess_binary__extra_erode=1,
            combine_cell_gradients__smooth_sigma=2,
        ),
    )


@case
def direct_manual_thresh():
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(
        view_probs(), view_gradients(), agg_params(combine_cell_probs__prob_thresh=0.6)
    )


for _missing in ([0], [1], [2], [0, 1], [0, 2], [1, 2]):

    def _mk(missing=_missing):
        def f():
            probs, grads = view_probs(), view_gradients()
            for m in missing:
                probs[m] = []
                grads[m] = []
            p = agg_params(combine_cell_gradients__smooth_sigma=2)
            return u3d.aggregate_2D_to_3D_segmentation_direct_method(probs, grads, p)

        return f

    _f = _mk()
    _f.__name__ = "direct_missing_" + "".join(map(str, _missing))
    case(_f)


@case
def direct_all_gradients_missing():
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(
        view_probs(), [[], [], []], agg_params()
    )


@case
def direct_precombined():
    g = np.array([GZ, GY, GX], dtype=np.float32)
    a = each(
        lambda: u3d.aggregate_2D_to_3D_segmentation_direct_method(
            [[]] * 3, [[]] * 3, agg_params(), precombined_binary=LABELS > 0, precombined_gradients=g
        )
    )[0]
    b = u3d.aggregate_2D_to_3D_segmentation_direct_method(
        [[]] * 3,
        [[]] * 3,
        agg_params(),
        precombined_binary=LABELS > 0,
        precombined_gradients=g.transpose(1, 2, 3, 0).copy(),
    )
    return a, b


@case
def direct_mp():
    p = agg_params(gradient_descent__do_mp=True, gradient_descent__tile_shape=(16, 24, 24))
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(view_probs(), view_gradients(), p)


@case
def direct_debugviz():
    d = tempfile.mkdtemp()
    p = agg_params(
        gradient_descent__debug_viz=True,
        gradient_descent__saveplotsfolder=d,
        gradient_descent__n_iter=3,
        gradient_descent__track_percent=0.1,
    )
    return u3d.aggregate_2D_to_3D_segmentation_direct_method(view_probs(), view_gradients(), p), d


for _method in ("edt", "cellpose_improve", "fmm", "fmm_skel", "cellpose_skel"):

    def _mk(method=_method):
        def f():
            d = tempfile.mkdtemp()
            p = agg_params(indirect_method__dtform_method=method)
            return u3d.aggregate_2D_to_3D_segmentation_indirect_method(
                view_segmentations(), p, SHAPE, savefolder=d, basename="i"
            ), d

        return f

    _f = _mk()
    _f.__name__ = "indirect_" + _method
    case(_f)


for _missing in ([0], [1], [2], [0, 1]):

    def _mk(missing=_missing):
        def f():
            s = view_segmentations()
            for m in missing:
                s[m] = []
            return u3d.aggregate_2D_to_3D_segmentation_indirect_method(
                s, agg_params(indirect_method__dtform_method="edt"), SHAPE
            )

        return f

    _f = _mk()
    _f.__name__ = "indirect_missing_" + "".join(map(str, _missing))
    case(_f)


@case
def indirect_nosmooth_precomputed_powerdist():
    a = each(
        lambda: u3d.aggregate_2D_to_3D_segmentation_indirect_method(
            view_segmentations(),
            agg_params(indirect_method__smooth_binary=0, indirect_method__dtform_method="edt"),
            SHAPE,
        )
    )[0]
    b = u3d.aggregate_2D_to_3D_segmentation_indirect_method(
        view_segmentations(),
        agg_params(indirect_method__power_dist=0.5),
        SHAPE,
        precomputed_binary=LABELS > 0,
    )
    c = each(
        lambda: u3d.aggregate_2D_to_3D_segmentation_indirect_method(
            view_segmentations(),
            agg_params(indirect_method__n_cpu=None, indirect_method__dtform_method="fmm"),
            SHAPE,
        )
    )[0]
    return a, b, c


def _post_params(**over):
    p = parameters.get_postprocess_segmentation_params()
    p["flow_consistency"]["n_cpu"] = 2
    p["size_filters"]["min_size"] = 20
    for k, v in over.items():
        a, b = k.split("__")
        p[a][b] = v
    return p


def _cell_grads():
    g = np.array([GZ, GY, GX]) + 0.3 * np.random.RandomState(5).randn(3, *SHAPE)
    g = g / (np.linalg.norm(g, axis=0) + 1e-20)
    g[:, LABELS == 3] *= -1  # make one cell inconsistent
    return g.transpose(1, 2, 3, 0).astype(np.float32)


for _method in ("edt", "cellpose_improve", "fmm"):

    def _mk(method=_method):
        def f():
            return u3d.postprocess_3D_cell_segmentation(
                LABELS.copy(),
                agg_params(),
                _post_params(
                    flow_consistency__dtform_method=method, flow_consistency__flow_threshold=0.5
                ),
                cell_gradients=_cell_grads(),
            )

        return f

    _f = _mk()
    _f.__name__ = "postprocess_" + _method
    case(_f)


@case
def postprocess_sizes():
    a = each(
        lambda: u3d.postprocess_3D_cell_segmentation(
            LABELS.copy(),
            agg_params(),
            _post_params(size_filters__max_size=600, size_filters__max_size_factor=0.5),
        )
    )[0]
    b = u3d.postprocess_3D_cell_segmentation(
        LABELS.copy(),
        agg_params(),
        _post_params(size_filters__min_size=0, size_filters__do_stats_filter=False),
    )
    c = each(
        lambda: u3d.postprocess_3D_cell_segmentation(
            LABELS.copy(),
            agg_params(),
            _post_params(
                flow_consistency__n_cpu=None,
                flow_consistency__dtform_method="edt",
                flow_consistency__flow_threshold=10,
            ),
            cell_gradients=_cell_grads(),
        )
    )[0]
    return a, b, c


@case
def label_diffuse():
    p = parameters.get_label_diffusion_params()
    p["diffusion"]["n_cpu"] = 2
    p["diffusion"]["refine_iters"] = 5
    p["diffusion"]["pad_size"] = 4
    a = each(lambda: u3d.label_diffuse_3D_cell_segmentation_MP(LABELS.copy(), IMG, p))[0]
    p["diffusion"]["n_cpu"] = None
    p["diffusion"]["affinity_type"] = "laplace"
    b = each(lambda: u3d.label_diffuse_3D_cell_segmentation_MP(LABELS.astype(np.uint16), IMG, p))[0]
    return a, b


def _gf_params(**over):
    p = parameters.get_guided_filter_params()
    p["guide_filter"]["n_cpu"] = 2
    p["guide_filter"]["pad_size"] = 4
    p["guide_filter"]["radius"] = 3
    p["ridge_filter"]["n_cpu"] = 2
    for k, v in over.items():
        a, b = k.split("__")
        p[a][b] = v
    return p


@case
def guided_filter_variants():
    out = []
    for over in (
        {},
        {"guide_filter__mode": "additive"},
        {"guide_filter__mode": "additive", "guide_filter__base_dilate": 1},
        {"guide_filter__mode": "additive", "guide_filter__base_erode": 0},
        {
            "guide_filter__adaptive_radius_bool": True,
            "guide_filter__use_int": True,
            "guide_filter__collision_dilate": 1,
        },
        {
            "guide_filter__collision_erode": 0,
            "guide_filter__collision_close": 0,
            "guide_filter__collision_fill_holes": False,
            "guide_filter__n_cpu": None,
        },
        {"ridge_filter__do_ridge_enhance": True},
        {"ridge_filter__do_ridge_enhance": True, "ridge_filter__do_multiprocess_2D": False},
        {"ridge_filter__do_ridge_enhance": True, "ridge_filter__n_cpu": None},
    ):
        out.append(
            u3d.guided_filter_3D_cell_segmentation_MP(LABELS.copy(), IMG, _gf_params(**over))
        )
    return out


# ----------------------------------------------------------------------------- watershed
@case
def ws_2d():
    lab = labels_2d()
    e = ndimage.distance_transform_edt(lab > 0)
    g = np.array(np.gradient(e))
    out = [watershed.gradient_watershed2D_binary(lab > 0, n_iter=10)]
    out.append(
        watershed.gradient_watershed2D_binary(
            lab > 0,
            gradient_img=g.transpose(1, 2, 0),
            n_iter=10,
            momenta=0.5,
            interp=False,
            divergence_rescale=False,
            thresh_factor=0.1,
            mask=lab > 0,
            binary_mask_gradient=True,
        )
    )
    out.append(
        watershed.gradient_watershed2D_binary(
            lab > 0, n_iter=3, debug_viz=True, track_flow=False, gradient_decay=0.1
        )
    )
    return out


@case
def ws_3d_variants():
    g = np.array([GZ, GY, GX]).transpose(1, 2, 3, 0).astype(np.float32)
    b = LABELS > 0
    return each(
        lambda: watershed.gradient_watershed3D_binary(b, n_iter=10),
        lambda: watershed.gradient_watershed3D_binary(
            b,
            gradient_img=g,
            n_iter=10,
            interp=True,
            binary_mask_gradient=True,
            thresh_factor=0.2,
            mask=b,
            track_percent=0.2,
        ),
        lambda: watershed.gradient_watershed3D_binary(
            b, gradient_img=g, n_iter=5, use_connectivity=True, renorm_gradient=False
        ),
        lambda: watershed.gradient_watershed3D_binary_dynamics(b, gradient_img=g, n_iter=10),
        lambda: watershed.gradient_watershed3D_binary_dynamics(
            b,
            n_iter=5,
            divergence_rescale=True,
            interp=True,
            use_connectivity=True,
            binary_mask_gradient=True,
        ),
        lambda: watershed.mp_gradient_watershed3D_binary(
            b, gradient_img=g, n_iter=10, tile_shape=(12, 20, 20)
        ),
        lambda: watershed.mp_gradient_watershed3D_binary(
            b, n_iter=5, tile_shape=None, divergence_rescale=True, thresh_factor=0.0, mask=b
        ),
    )


@case
def ws_helpers():
    pts = np.argwhere(LABELS > 0).astype(np.float32)
    return each(
        lambda: watershed._normalize99(IMG),
        lambda: watershed._interp2(pts[:50, 1:] + 0.3, SHAPE[1:], IMG[5]),
        lambda: watershed._interp3(pts[:50] + 0.3, SHAPE, IMG),
        lambda: watershed.connected_components_pts_3D(pts, pts, SHAPE),
        lambda: watershed.connected_components_pts_2D(
            pts[:, 1:], pts[:, 1:], SHAPE[1:], thresh_factor=0.5, mask=LABELS[5] > 0
        ),
        lambda: watershed._sdf_distance_transform(LABELS > 0),
        lambda: watershed.surf_normal_sdf(LABELS > 0, smooth_gradient=1),
        lambda: watershed.mean_curvature_sdf(watershed.surf_normal_sdf(LABELS > 0)[0]),
    )


# ----------------------------------------------------------------------------- flows
@case
def flows_dtforms_2d():
    lab = labels_2d()
    out = [
        flows.distance_transform_labels_fast(LABELS),
        flows.distance_transform_labels_fast(LABELS, sdf=True, black_border=True),
    ]
    for m in ("cellpose", "cellpose_improve", "fmm", "fmm_skel", "cellpose_skel", "edt", "bogus"):
        try:
            out.append(
                flows.distance_centroid_tform_flow_labels2D(
                    lab,
                    dtform_method=m,
                    guide_img=ndimage.distance_transform_edt(lab > 0),
                    fixed_point_percentile=0.01,
                )
            )
        except Exception as e:
            out.append(("exc", type(e).__name__, str(e)))
    out.append(
        flows.distance_centroid_tform_flow_labels2D(
            lab, dtform_method="cellpose_improve", power_dist=0.5
        )
    )
    out.append(
        flows.distance_centroid_tform_flow_labels2D(lab, dtform_method="fmm", guide_img=None)
    )
    coords = np.argwhere(lab == lab.max())
    c = coords.mean(0)
    out += each(
        lambda: flows.poisson_dist_tform(lab > 0),
        lambda: flows.poisson_dist_tform_flow(coords, c, lab.shape),
        lambda: flows.poisson_dist_tform_flow(coords, c, lab.shape, power_dist=0.5, norm_dist=True),
        lambda: flows.poisson_dist_tform_flow_sets(coords, coords[:3], lab.shape),
        lambda: flows.fmm_point_source2D(coords, c, lab.shape),
        lambda: flows.fmm_point_source2D_sets(coords, coords[:3], lab.shape),
        lambda: flows.cellpose_diffuse2D(coords, c, lab.shape, 20),
        lambda: flows.distance_tform_labels2D(lab, ndimage.distance_transform_edt),
        lambda: flows.distance_tform_labels2D(lab, ndimage.distance_transform_edt, clip=True),
        lambda: flows._relabel_slices(LABELS),
        lambda: flows.poisson_dist_tform_flow(coords, coords[len(coords) // 2], lab.shape),
        lambda: flows.poisson_dist_tform_flow(
            coords, coords[len(coords) // 2], lab.shape, power_dist=0.5
        ),
        lambda: flows.poisson_dist_tform_flow_sets(coords, coords[:3], lab.shape, power_dist=0.5),
        lambda: flows.fmm_point_source2D(coords, coords[len(coords) // 2], lab.shape),
        lambda: flows.cellpose_diffuse2D(
            coords + 1, coords[len(coords) // 2] + 1, np.array(lab.shape) + 2, 20
        ),
        lambda: flows.distance_tform_labels2D(lab, ndimage.distance_transform_edt, clip=True),
        lambda: flows._largest_component_vol(LABELS > 0),
    )
    return out


@case
def flows_parallel():
    return each(
        lambda: flows.distance_centroid_tform_flow_labels2D_parallel(
            LABELS[:6], dtform_method="cellpose_improve", n_processes=2
        ),
        lambda: flows.distance_centroid_tform_flow_labels2D_parallel(
            LABELS[:6], dtform_method="fmm", n_processes=2, guide_image=EDT[:6]
        ),
        lambda: flows.distance_centroid_tform_flow_labels2D_dask(
            LABELS[:6], dtform_method="cellpose_skel", n_processes=2, guide_image=EDT[:6]
        ),
        lambda: flows.distance_centroid_tform_flow_labels2D_dask(
            LABELS[:6], dtform_method="edt", n_processes=2
        ),
        lambda: flows.distance_tform_labels2D_parallel(
            LABELS[:6], ndimage.distance_transform_edt, n_processes=2
        ),
    )


@case
def flows_remove_bad_mp_fresh():
    """remove_bad_flow_masks_3D_mp sets the spawn start method: run on its own (fresh process)."""
    flow3 = _cell_grads().transpose(3, 0, 1, 2)
    return flows.remove_bad_flow_masks_3D_mp(LABELS, flow3, n_processes=2, flow_threshold=0.5)


@case
def flows_remove_bad():
    flow3 = _cell_grads().transpose(3, 0, 1, 2)
    lab = labels_2d()
    f2 = np.array(np.gradient(ndimage.distance_transform_edt(lab > 0)))
    f2[:, lab == lab.max()] *= -1
    out = []
    for fn, args in (
        (flows.remove_bad_flow_masks_2D, (lab, f2)),
        (flows.remove_bad_flow_masks_3D, (LABELS, flow3)),
        (flows.remove_bad_flow_masks_3D_edt_from2D, (LABELS, flow3)),
        (flows.remove_bad_flow_masks_3D_edt, (LABELS, flow3)),
        (flows.remove_bad_flow_masks_3D_mp, (LABELS, flow3)),
    ):
        try:
            out.append(fn(*args, n_processes=2, flow_threshold=0.5))
        except Exception as e:
            out.append(("exc", fn.__name__, type(e).__name__, str(e)))
    try:
        out.append(
            flows.remove_bad_flow_masks_3D(
                LABELS, flow3, n_processes=2, smooth_sigma=1, smooth_ds=2, dtform_method="edt"
            )
        )
    except Exception as e:
        out.append(("exc", type(e).__name__, str(e)))
    try:
        out.append(
            flows.remove_bad_flow_masks_3D(
                LABELS,
                flow3,
                n_processes=2,
                smooth_sigma=1,
                smooth_ds=2,
                dtform_method="cellpose_improve",
                flow_threshold=0.5,
            )
        )
    except Exception as e:
        out.append(("exc", type(e).__name__, str(e)))
    for fn in (
        flows.remove_bad_flow_masks_2D,
        flows.remove_bad_flow_masks_3D,
        flows.remove_bad_flow_masks_3D_edt,
    ):
        try:
            out.append(fn(LABELS, flow3, dtform_method="cellpose"))
        except Exception as e:
            out.append(("exc", type(e).__name__, str(e)))
    try:
        out.append(flows.remove_bad_flow_masks_3D_edt_from2D(LABELS, flow3[:2]))
    except Exception as e:
        out.append(("exc", type(e).__name__, str(e)))
    return out


# ----------------------------------------------------------------------------- filters
@case
def filters_basic():
    lab = LABELS.copy()
    lab2 = lab.copy()
    lab2[5:7, 5:8, 5:8] = 99  # small stray component
    return each(
        lambda: filters.fill_array(IMG, thresh=0.2),
        lambda: filters.fill_array(IMG, thresh=0.2, method="median"),
        lambda: filters.fill_array(IMG, thresh=0.2, method="mean"),
        lambda: filters.imadjust(IMG, 2, 98),
        lambda: filters.normalize(IMG, clip=True),
        lambda: filters.normalize(IMG, axis=(1, 2)),
        lambda: filters.normalize_mi_ma(IMG, 0.1, 0.5),
        lambda: filters.anisodiff(IMG[5], niter=3),
        lambda: filters.anisodiff(IMG[5], niter=3, option=2, sigma=1),
        lambda: filters.meijering_ridge_filter_2D_multiprocess(IMG[:4], sigmas=[1], n_cpu=2),
        lambda: filters.relabel_slices(lab),
        lambda: filters.connected_component_slices(lab),
        lambda: filters.filter_2d_label_slices(lab, minsize=20),
        lambda: filters.detect_and_relabel_multi_component_labels(lab2, min_size=5),
        lambda: filters.filter_segmentations_axis(lab, window=3, min_count=1),
        lambda: filters.filter_segmentations_axis_IoU(lab, window=3),
        lambda: filters.smooth_vol(lab > 0, ds=2, smooth=1),
        lambda: filters.smooth_vol_anisotropic(lab > 0, ds=2, smooth=[1, 2, 2]),
        lambda: filters.remove_small_labels(lab2, min_size=30),
        lambda: filters.intensity_filter_labels(lab, IMG),
        lambda: filters.intensity_filter_labels(lab, IMG, threshold=0.6),
        lambda: filters.expand_masks(lab * (EDT > 3), lab > 0),
        lambda: filters.expand_masks2D(labels_2d() * (EDT[SHAPE[0] // 2] > 2), labels_2d() > 0),
        lambda: filters.remove_eccentric_shapes(lab, min_size=20, stretch_cutoff=2),
        lambda: filters.var_filter(IMG, 3),
        lambda: filters.fast_dog_filter(IMG, 3),
        lambda: filters.var_combine([IMG, IMG**2], ksize=3),
        lambda: filters.dog_combine([IMG, IMG**2], ksize=3),
        lambda: filters.guidedfilter((lab == 1) * 1.0, IMG, 2, 1e-3),
        lambda: filters.largest_component_vol(lab > 0),
        lambda: filters.largest_component_vol_labels(lab2),
        lambda: filters.largest_component_vol_labels_fast(lab2),
    )


@case
def filters_entropy():
    return filters.entropy_mean([(IMG * 255).astype(np.uint8), (IMG**2 * 255).astype(np.uint8)])


@case
def filters_diffuse():
    lab = labels_2d()
    out = []
    for kw in ({}, {"affinity_type": "laplace"}, {"reinitialize_freq": 2}):
        out += each(
            lambda: filters.diffuse_labels2D(lab, IMG[SHAPE[0] // 2], n_iter=4, **kw),
            lambda: filters.diffuse_labels2D(
                lab, IMG[SHAPE[0] // 2] * 1.0, n_iter=4, noprogress=False, **kw
            ),
            lambda: filters.diffuse_labels3D((LABELS[:6] == 1) * 1, IMG[:6], n_iter=4, **kw),
        )
    out += each(
        lambda: filters._distance_to_heat_affinity_matrix(
            np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0)))
        ),
        lambda: filters._distance_to_laplace_affinity_matrix(
            np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0))), gamma=0.5
        ),
        lambda: filters._distance_to_heat_affinity_matrix(
            __import__("scipy.sparse").sparse.csr_matrix(
                np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0)))
            )
        ),
    )
    return out


# ----------------------------------------------------------------------------- gpu (CPU paths)
@case
def gpu_cpu_paths():
    out = [
        gpu.dask_cpu_rescale(IMG, [1, 0.5, 0.5]),
        gpu.num_bg_correct(IMG, IMG * 0.5 + 0.1),
        gpu.num_normalize(IMG),
        gpu._smooth_vol(IMG, ds=2, sigma=1),
        gpu.bg_normalize_cpu(IMG, bg_ds=2, bg_sigma=1),
    ]
    for fn in (
        lambda: gpu.bg_normalize(IMG, bg_ds=2, bg_sigma=1),
        lambda: gpu.bg_normalize_torch(IMG, 2, 1),
        lambda: gpu.dask_cuda_rescale(IMG, [1, 1, 1]),
        lambda: gpu.zoom_3d_pytorch(IMG, [1, 1, 1]),
    ):
        try:
            out.append(fn())
        except Exception as e:
            out.append(("exc", type(e).__name__))
    return out


@case
def gpu_torch_cupy():
    small = IMG[:16, :32, :32].copy()
    return each(
        lambda: gpu.zoom_3d_pytorch(small, [0.5, 0.5, 0.5]),
        lambda: gpu.zoom_3d_pytorch(small, 2),
        lambda: gpu.zoom_3d_pytorch(small, 1.5, interp_mode="nearest"),
        lambda: gpu.zoom_3d_pytorch(small, [1, 1, 1], output_size=(20, 30, 40)),
        lambda: gpu.zoom_3d_pytorch(
            small, [1, 1, 1], interp_mode="nearest", output_size=np.array([20, 30, 40])
        ),
        lambda: gpu.zoom_3d_pytorch(small, [1, 1, 1], interp_mode="area", output_size=(8, 16, 16)),
        lambda: gpu.zoom_3d_pytorch(small[0], [1, 1]),
        lambda: gpu.torch_smooth_vol(small, ds=2, sigma=1),
        lambda: gpu.bg_normalize_torch(small, bg_ds=2, bg_sigma=1),
        lambda: gpu.cuda_rescale(small, [0.5, 1, 2]),
        lambda: gpu.cuda_rescale(small, 0.5, order=0, mode="constant"),
        lambda: gpu.cuda_smooth_vol(small, ds=2, sigma=1),
        lambda: gpu.bg_normalize(small, bg_ds=2, bg_sigma=1),
        lambda: gpu.cuda_normalize(small, cast_numpy=True),
        lambda: gpu.cuda_normalize(small, clip=True, axis=0, cast_numpy=True),
        lambda: type(gpu.cuda_normalize(small)).__module__,
        lambda: gpu.dask_cuda_rescale(small, [1, 0.5, 0.5], chunksize=(8, 16, 16)),
        lambda: gpu.dask_cuda_bg(IMG, bg_ds=2, bg_sigma=1, chunksize=(10, 18, 18)),
        lambda: gpu.cuda_equalize_adapthist(IMGN[5], kernel_size=(8, 8)),
        lambda: seg._histeq_low_contrast(IMGN[5], kernel_size=(8, 8)),
    )


@case
def preprocess_torch_fallback():
    """Force the cupy paths to fail so preprocess_imgs falls back to torch."""

    def fail(*a, **k):
        raise RuntimeError("forced cupy failure")

    saved = gpu.dask_cuda_rescale, gpu.bg_normalize
    gpu.dask_cuda_rescale, gpu.bg_normalize = fail, fail
    try:
        p = parameters.get_preprocess_params()
        p["bg_ds"] = 4
        p["voxel_res"] = [2, 1, 1]
        a = each(lambda: u3d.preprocess_imgs(IMG, p))
        p["do_bg_correction"] = False
        b = each(lambda: u3d.preprocess_imgs(np.stack([IMG, IMG**2]), p))
    finally:
        gpu.dask_cuda_rescale, gpu.bg_normalize = saved
    return a, b


# ----------------------------------------------------------------------------- metrics / plotting / mesh / io
@case
def metrics_all():
    pred = np.roll(LABELS, 1, axis=2)
    pred[pred == 2] = 0
    t2, p2 = LABELS[10], pred[10]
    out = []
    for fn in (
        lambda: metrics.metrics_np((LABELS > 0)[..., None] * 1, (pred > 0)[..., None] * 1, "iou"),
        lambda: metrics.metrics_np(
            (LABELS > 0)[..., None] * 1,
            (pred > 0)[..., None] * 1,
            "dice",
            metric_type="naive",
            drop_last=False,
            mean_per_class=True,
            verbose=True,
        ),
        lambda: metrics.mean_iou_np((LABELS > 0)[..., None] * 1, (pred > 0)[..., None] * 1),
        lambda: metrics.mean_dice_np((LABELS > 0)[..., None] * 1, (pred > 0)[..., None] * 1),
        lambda: metrics.remove_small_labelled_objects(LABELS, minsize=500),
        lambda: metrics.get_valid_eval_region_segmentation(LABELS, pred),
        lambda: metrics.compute_metrics_cells(LABELS, pred, K=3),
        lambda: metrics.compute_metrics_cells(t2, p2, K=3, iou_thresh=0.3),
        lambda: metrics.mask_ious(t2, p2),
        lambda: metrics.aggregated_jaccard_index([t2], [p2]),
        lambda: metrics.average_precision([t2, t2], [p2, t2]),
        lambda: metrics.average_precision(t2, p2),
        lambda: metrics._intersection_over_union(t2, p2),
        lambda: metrics._true_positive(metrics._intersection_over_union(t2, p2), 0.5),
    ):
        try:
            out.append(fn())
        except Exception as e:
            out.append(("exc", type(e).__name__, str(e)))
    return out


@case
def plotting_all():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0, 1], [0, 2], [0, 3])
    plotting.set_axes_equal(ax)
    lims = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    fig2, ax2 = plt.subplots()
    tracks = np.random.RandomState(0).rand(3, 40, 2) * 30
    plotting.plot_border_tracks_2D(tracks, labels_2d(), ax2, samples=2)
    n_lines = len(ax2.lines)
    plt.close("all")
    return each(
        lambda: lims,
        lambda: n_lines,
        lambda: plotting.relabel_ground_truth_labels(LABELS),
        lambda: plotting.relabel_ground_truth_labels(LABELS, reorder=False, minsize=200),
        lambda: plotting.color_segmentation(LABELS, "Spectral"),
        lambda: plotting.get_colors(np.arange(5.0), plt.cm.viridis),
        lambda: plotting.get_colors(np.arange(5.0), plt.cm.viridis, vmin=1, vmax=3, bg_label=0),
    )


@case
def meshtools_all():
    m = meshtools.marching_cubes_mesh_binary(LABELS == 1)
    m2 = meshtools.marching_cubes_mesh_binary(LABELS > 0, presmooth=0, keep_largest_only=False)
    d = tempfile.mkdtemp()
    m.export(os.path.join(d, "m.obj"))
    r = meshtools.read_mesh(os.path.join(d, "m.obj"), keep_largest_only=True)
    c = meshtools.create_mesh(m.vertices, m.faces, vertex_colors=np.ones((len(m.vertices), 4)))
    return each(
        lambda: (m.vertices, m.faces),
        lambda: (m2.vertices, m2.faces),
        lambda: (r.vertices, r.faces),
        lambda: (c.vertices, c.faces),
        lambda: meshtools.measure_props_trimesh(m),
        lambda: meshtools.measure_props_trimesh(m2, main_component=False, clean=False),
    )


@case
def file_io_all():
    d = tempfile.mkdtemp()
    file_io.mkdir(os.path.join(d, "a", "b"))
    file_io.write_pickle(os.path.join(d, "p.pkl"), {"x": 1})
    r = file_io.read_pickle(os.path.join(d, "p.pkl"))
    file_io.save_segmentation(os.path.join(d, "s.tif"), LABELS)
    file_io.save_segmentation(os.path.join(d, "t.tif"), LABELS * 1000)
    return r, d


@case
def parameters_all():
    return [
        parameters.get_preprocess_params(),
        parameters.get_Cellpose_autotune_params(),
        parameters.get_2D_to_3D_aggregation_params(),
        parameters.get_postprocess_segmentation_params(),
        parameters.get_label_diffusion_params(),
        parameters.get_guided_filter_params(),
    ]


# ----------------------------------------------------------------------------- runner
def snapshot_dir(d):
    """Replace a temp dir by {relative path: sha256} of its files."""
    out = {}
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            data = open(p, "rb").read()
            if f.endswith(".pkl"):
                obj = pickle.loads(data)
                out[os.path.relpath(p, d)] = ("pkl", obj)
            elif f.endswith(".svg"):
                out[os.path.relpath(p, d)] = ("svg", None)  # svg embeds timestamps/ids
            elif f.endswith(".mat"):
                import scipy.io as spio

                m = {k: v for k, v in spio.loadmat(p).items() if not k.startswith("__")}
                out[os.path.relpath(p, d)] = ("mat", m)
            elif f.endswith(".obj"):
                out[os.path.relpath(p, d)] = ("obj", None)
            else:
                out[os.path.relpath(p, d)] = ("sha", hashlib.sha256(data).hexdigest())
    shutil.rmtree(d, ignore_errors=True)
    return out


def scrub(x):
    """Turn results into comparable plain data (dirs -> file snapshots)."""
    if isinstance(x, str) and x.startswith(tempfile.gettempdir()) and os.path.isdir(x):
        return ("DIR", snapshot_dir(x))
    if isinstance(x, (list, tuple)):
        return (
            type(x)(scrub(v) for v in x)
            if not hasattr(x, "_fields")
            else tuple(scrub(v) for v in x)
        )
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if callable(x) and not isinstance(x, np.ndarray):
        return ("callable", getattr(x, "__name__", repr(type(x))))
    if isinstance(x, (np.ndarray, np.generic, int, float, bool, str, bytes, type(None), complex)):
        return x
    if hasattr(x, "__dict__") and type(x).__module__.startswith("skimage"):
        return ("obj", type(x).__name__)
    return ("repr", type(x).__name__, repr(x)[:200])


def main():
    results = {}
    for fn in CASES:
        if FILTER and not any(f in fn.__name__ for f in FILTER.split("|")):
            continue
        if fn.__name__.endswith("_fresh") and FILTER != fn.__name__:
            continue
        np.random.seed(0)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = fn()
            res = ("ok", scrub(copy.copy(res)))
        except Exception as e:
            res = ("exc", type(e).__name__, str(e), traceback.format_exc().splitlines()[-3:-1])
        results[fn.__name__] = (buf.getvalue(), res)
        status = res[0] if res[0] == "ok" else f"EXC {res[1]}: {res[2][:120]}"
        print(f"{fn.__name__:45s} {status}", file=sys.__stderr__, flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
