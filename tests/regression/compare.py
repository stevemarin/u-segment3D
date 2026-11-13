"""Compare two snapshot pickles bit-for-bit (dtype, shape, values incl. NaN positions)."""

import pickle, re, sys


def norm_out(t):
    return sorted(re.sub(r"(finished dynamics:,\s+)[\d.e-]+", r"\1<t>", t).splitlines())


import numpy as np


def diff(a, b, path):
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: array {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
        if a.dtype == object:
            return [
                d
                for i, (x, y) in enumerate(zip(a.ravel(), b.ravel()))
                for d in diff(x, y, f"{path}[{i}]")
            ]
        eq = np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        if not eq:
            n = np.sum(
                ~((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind in "fc" else (a == b))
            )
            return [
                f"{path}: {n} differing elements (max abs diff {np.nanmax(np.abs(a.astype(float) - b.astype(float))) if a.dtype.kind in 'fciub' else '?'})"
            ]
        return []
    if isinstance(a, np.generic):
        same = a == b or (np.issubdtype(type(a), np.floating) and np.isnan(a) and np.isnan(b))
        return [] if same else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(map(str, set(a) ^ set(b)))}"]
        return [d for k in a for d in diff(a[k], b[k], f"{path}[{k!r}]")]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [f"{path}: {str(a)[:150]!r} != {str(b)[:150]!r}"]


A = pickle.load(open(sys.argv[1], "rb"))
B = pickle.load(open(sys.argv[2], "rb"))
bad = 0
for k in A:
    if k not in B:
        print(f"MISSING {k}")
        bad += 1
        continue
    ra, rb = A[k][1], B[k][1]
    if ra[0] == "exc":
        ra = ra[:3]
    if rb[0] == "exc":
        rb = rb[:3]
    A[k] = (A[k][0], ra)
    B[k] = (B[k][0], rb)
    d = diff(norm_out(A[k][0]), norm_out(B[k][0]), "stdout(sorted lines)") + diff(
        A[k][1], B[k][1], "result"
    )
    if d:
        bad += 1
        print(f"DIFF {k}")
        [print("   ", x) for x in d[:8]]
print(f"{len(A) - bad}/{len(A)} cases identical")
sys.exit(1 if bad else 0)
