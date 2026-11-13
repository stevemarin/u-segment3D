"""Check that the working tree's segment3D behaves exactly like a reference git revision.

Usage:
    python tests/regression/check.py [--ref REV] [--filter NAMES]

The reference revision's ``segment3D/`` is extracted with ``git archive`` and both
versions are run through ``run_cases.py`` in the current Python environment. The
results are compared bit-for-bit with ``compare.py`` (return values, dtypes, shapes,
exceptions, stdout and files written). Exits with status 1 if anything differs.

Cellpose is replaced by the deterministic fake in ``fake_cellpose/``; the Cellpose
cases are run once with a v3 and once with a v4 version string. Cases ending in
``_fresh`` need a fresh interpreter and are run on their own.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FAKE_CELLPOSE = os.path.join(HERE, "fake_cellpose")

# (label, case filter, extra environment)
RUNS = [
    ("all cases (Cellpose v3)", "", {}),
    ("Cellpose v4 code paths", "cellpose|seg_", {"FAKE_CELLPOSE_VERSION": "4.0.5"}),
    ("fresh-process cases", "flows_remove_bad_mp_fresh", {}),
]


def run_cases(pkg_root, out, case_filter, env_extra):
    env = dict(os.environ, MPLBACKEND="Agg", **env_extra)
    cmd = [sys.executable, "-I", os.path.join(HERE, "run_cases.py"), pkg_root, FAKE_CELLPOSE, out]
    if case_filter:
        cmd.append(case_filter)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"run_cases.py failed for {pkg_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ref", default="master", help="git revision to compare against (default: master)"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="only run cases whose name contains one of these ('|' separated)",
    )
    args = parser.parse_args()

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        ref_root = os.path.join(tmp, "ref")
        os.makedirs(ref_root)
        archive = subprocess.run(
            ["git", "-C", REPO, "archive", args.ref, "segment3D"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess.run(["tar", "-x", "-C", ref_root], input=archive.stdout, check=True)

        runs = RUNS if args.filter is None else [("selected cases", args.filter, {})]
        for label, case_filter, env_extra in runs:
            print(f"--- {label}", flush=True)
            ref_out = os.path.join(tmp, "ref.pkl")
            new_out = os.path.join(tmp, "new.pkl")
            run_cases(ref_root, ref_out, case_filter, env_extra)
            run_cases(REPO, new_out, case_filter, env_extra)
            result = subprocess.run(
                [sys.executable, os.path.join(HERE, "compare.py"), ref_out, new_out]
            )
            failed |= result.returncode != 0

    print(f"\nworking tree vs {args.ref}: " + ("DIFFERENCES FOUND" if failed else "identical"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
