"""Visualisations of molecular structures and volumetric data."""

import numpy as np
import py3Dmol
from more_itertools import chunked
from numpy.typing import NDArray

from mess.structure import Structure
from mess.types import MeshAxes
from mess.units import to_angstrom


def plot_molecule(view: py3Dmol.view, structure: Structure) -> py3Dmol.view:
    """Plots molecular structure.

    Args:
        view (py3Dmol.view): py3DMol View to which to add visualizer
        structure (Structure): molecular structure

    Returns:
        py3DMol View object

    """
    xyz = f"{structure.num_atoms}\n\n"
    sym = structure.atomic_symbol
    pos = to_angstrom(structure.position)

    for i in range(structure.num_atoms):
        r = np.array2string(pos[i, :], separator="\t")[1:-1]
        xyz += f"{sym[i]}\t{r}\n"

    view.addModel(xyz)
    style = "stick" if structure.num_atoms > 1 else "sphere"
    view.setStyle({style: {"radius": 0.1}})
    return view


def plot_volume(view: py3Dmol.view, value: NDArray, axes: MeshAxes):
    """Plots volumetric data value with molecular structure.

    Volumetric render using https://3dmol.csb.pitt.edu/doc/VolumetricRendererSpec.html

    Args:
        view (py3Dmol.view): py3DMol View to which to add visualizer
        value (NDArray): the volume data to render
        axes (MeshAxes): the axes over which the data was sampled.

    Returns:
        py3DMol View object

    """

    s = cube_data(value, axes)
    view.addVolumetricData(s, "cube", build_transferfn(value))
    return view


def plot_isosurfaces(
    view: py3Dmol.view, value: NDArray, axes: MeshAxes, percentiles=[95, 75]
):
    """Plots volumetric data value with molecular structure.

    IsoSurface render using https://3dmol.csb.pitt.edu/doc/IsoSurfaceSpec.html

    Args:
        view (py3Dmol.view): py3DMol View to which to add visualizer
        value (NDArray): the volume data to render
        axes (MeshAxes): the axes over which the data was sampled.
        percentiles (tuple): percentiles at which to draw isosurfaces

    Returns:
        py3DMol View object

    Note:
        3Dmol does not currently implement full transparency, so only two
        percentiles are accepted, with the inner one being rendered with full opacity.
        - https://github.com/3dmol/3Dmol.js/issues/224
    """
    assert len(percentiles) == 2

    voldata_str = cube_data(value, axes)

    v = np.percentile(np.abs(value), tuple(reversed(sorted(percentiles))))
    for sign in [-1, 1]:
        for isovalind in (0, 1):
            isoval = sign * v[isovalind]
            tf = {
                "isoval": isoval,
                "smoothness": 2,
                "opacity": 0.9 if isovalind == 1 else 1.0,
                "voldata": voldata_str,
                "volformat": "cube",
                "volscheme": {"gradient": "rwb", "min": -v[0], "max": v[0]},
            }
            view.addVolumetricData(voldata_str, "cube", tf)

    return view


def cube_data(value: NDArray, axes: MeshAxes) -> str:
    """Generate the cube file format as a string.  See:

      https://paulbourke.net/dataformats/cube/

    Args:
        value (NDArray): the volume data to serialise in the cube format
        axes (MeshAxes): the axes over which the data was sampled

    Returns:
        str: cube format representation of the volumetric data.
    """
    # The first two lines of the header are comments, they are generally ignored by
    # parsing packages or used as two default labels.
    fmt = "cube format\n\n"

    axes = [to_angstrom(ax) for ax in axes]
    x, y, z = axes
    # The third line has the number of atoms included in the file followed by the
    # position of the origin of the volumetric data.
    fmt += f"0 {cube_format_vec([x[0], y[0], z[0]])}\n"

    # The next three lines give the number of voxels along each axis (x, y, z)
    # followed by the axis vector. Note this means the volume need not be aligned
    # with the coordinate axis, indeed it also means it may be sheared although most
    # volumetric packages won't support that.
    # The length of each vector is the length of the side of the voxel thus allowing
    # non cubic volumes.
    # If the sign of the number of voxels in a dimension is positive then the
    # units are Bohr, if negative then Angstroms.
    nx, ny, nz = [ax.shape[0] for ax in axes]
    dx, dy, dz = [ax[1] - ax[0] for ax in axes]
    fmt += f"{nx} {cube_format_vec([dx, 0.0, 0.0])}\n"
    fmt += f"{ny} {cube_format_vec([0.0, dy, 0.0])}\n"
    fmt += f"{nz} {cube_format_vec([0.0, 0.0, dz])}\n"

    # The last section in the header is one line for each atom consisting of 5
    # numbers, the first is the atom number, the second is the charge, and the last
    # three are the x,y,z coordinates of the atom center.
    pass  # Number of atoms = 0 above

    # The volumetric data is straightforward, one floating point number for each
    # volumetric element.  Traditionally the grid is arranged with the x axis as
    # the outer loop and the z axis as the inner loop
    for vals in chunked(value, 6):
        fmt += f"{cube_format_vec(vals)}\n"

    return fmt


def cube_format_vec(vals):
    """
    From https://paulbourke.net/dataformats/cube, floats are formatted 12.6
    """
    # Benchmarks showed this is 4x faster than numpy.printoptions...
    return " ".join([f"{v:12.6f}" for v in vals])


def build_transferfn(value: NDArray) -> dict:
    """Generate the 3dmol.js transferfn argument for a particular value.

    Tries to set isovalues to capture main features of the volume data.

    Args:
        value (NDArray): the volume data.

    Returns:
        dict: containing transferfn
    """
    v = np.percentile(value, [99.9, 75])
    a = [0.02, 0.0005]
    return {
        "transferfn": [
            {"color": "blue", "opacity": a[0], "value": -v[0]},
            {"color": "blue", "opacity": a[1], "value": -v[1]},
            {"color": "white", "opacity": 0.0, "value": 0.0},
            {"color": "red", "opacity": a[1], "value": v[1]},
            {"color": "red", "opacity": a[0], "value": v[0]},
        ]
    }
