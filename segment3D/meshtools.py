# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:25:24 2018

@author: felix

Mesh helpers (trimesh).
"""

import numpy as np
import trimesh
from skimage.filters import gaussian


def _largest_component(mesh):
    """The connected component of the mesh with the most vertices."""
    mesh_comps = mesh.split(only_watertight=False)
    return mesh_comps[np.argmax([len(cc.vertices) for cc in mesh_comps])]


def read_mesh(meshfile, process=False, validate=False, keep_largest_only=False):
    """Load a mesh file, optionally keeping only its largest connected component."""
    mesh = trimesh.load_mesh(meshfile, validate=validate, process=process)
    if keep_largest_only:
        mesh = _largest_component(mesh)

    return mesh


def create_mesh(vertices, faces, vertex_colors=None, face_colors=None):
    """Create a trimesh mesh without any processing or validation."""
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
        vertex_colors=vertex_colors,
        face_colors=face_colors,
    )

    return mesh


def marching_cubes_mesh_binary(vol, presmooth=1.0, contourlevel=0.5, keep_largest_only=True):
    """Mesh the surface of a binary volume (optionally Gaussian presmoothed) with marching cubes."""
    try:
        from skimage.measure import marching_cubes_lewiner  # scikit-image < 0.19
    except:  # noqa: E722
        from skimage.measure import marching_cubes

    if presmooth is not None:
        img = gaussian(vol, sigma=presmooth, preserve_range=True)
        img = img / img.max()  # do this.
    else:
        img = vol.copy()

    try:
        V, F, _, _ = marching_cubes_lewiner(img, level=contourlevel, allow_degenerate=False)
    except:  # noqa: E722
        V, F, _, _ = marching_cubes(img, level=contourlevel, method="lewiner")
    mesh = trimesh.Trimesh(V, F, validate=True)

    if keep_largest_only:
        mesh = _largest_component(mesh)

    return mesh


def measure_props_trimesh(mesh, main_component=True, clean=True):
    r"""Compute basic statistics and properties of a given mesh

    - is Convex : Yes/No
    - is Volume : Yes/No - is it closed such that a volume can be computed
    - is Watertight : Yes/No - is it closed such that a volume can be computed
    - orientability : Yes/No - can all faces be oriented the same way. Mobius strips and Klein bottles are non-orientable
    - Euler number : or Euler characteristic, :math:`\chi` #vertices - #edges + #faces
    - Genus : :math:`(2-2\chi)/2` if orientable or :math:`2-\chi` if nonorientable

    Parameters
    ----------
    mesh : trimesh.Trimesh
        input mesh
    main_component : bool
        if True, get the largest mesh component and compute statistics on this
    clean : bool
        if True, removes NaN and infs and degenerate and duplicate faces which may affect the computation of some of these statistics

    Returns
    -------
    props : dict
        A dictionary containing the metrics

        'convex' : bool

        'volume' : bool

        'watertight' : bool

        'orientability' : bool

        'euler_number' : scalar

        'genus' : scalar

    """
    if clean:
        mesh_ = trimesh.Trimesh(mesh.vertices, mesh.faces, validate=True, process=True)
    else:
        mesh_ = mesh.copy()
    # note: the largest component is always used, whatever main_component is.
    main_mesh = _largest_component(mesh_)

    props = {}
    props["convex"] = main_mesh.is_convex
    props["volume"] = main_mesh.is_volume
    props["watertight"] = main_mesh.is_watertight
    props["orientability"] = main_mesh.is_winding_consistent
    props["euler_number"] = main_mesh.euler_number

    if main_mesh.is_winding_consistent:
        # if orientable we can use the euler_number computation. see wolfram mathworld!.
        genus = (2.0 - props["euler_number"]) / 2.0  # euler = 2-2g
    else:
        genus = 2.0 - props["euler_number"]
    props["genus"] = genus

    return props
