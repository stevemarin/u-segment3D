# Regression check

Checks that changes to `segment3D/` do not change its behavior: the current working tree
and a reference git revision are run on the same small synthetic 3D data, and every result
is compared bit-for-bit (return values, dtypes and shapes, exceptions, printed output, and
files written). It is meant for refactoring and clean-up changes, where the expected result
is "identical".

```bash
python tests/regression/check.py                 # compare with master
python tests/regression/check.py --ref upstream/master
python tests/regression/check.py --filter direct # only cases whose name contains "direct"
```

- `run_cases.py` holds the cases (about 60, covering every public function) and the synthetic data.
- `compare.py` compares two result files.
- `fake_cellpose/` is a small deterministic stand-in for Cellpose that also records the
  arguments of every `model.eval` call; the Cellpose cases run with both a v3 and a v4 version
  string, so both code paths are checked.

Requirements: the package dependencies (numpy, scipy, scikit-image, scikit-learn, dask with
distributed, numba, multiprocess, tiler, connected-components-3d, edt, trimesh, seaborn,
scikit-fmm). Cellpose is not needed. GPU code runs if cupy / cucim / torch are installed;
otherwise both versions raise the same errors and those are compared instead.

Notes:
- Both versions run in the same Python environment, so results do not depend on library versions.
- Printed lines are compared without regard to order (parallel workers print in any order),
  and printed timings are ignored.
- This does not test real Cellpose models, real microscopy data, or the MATLAB interface.
