""" write the golden snapshot (test/golden/*.json)

Run with the reference implementation only:
    python test/make_golden.py
Then check the snapshot with test/test_golden.py.
"""

import tempfile
import golden_lib


def main():
    with tempfile.TemporaryDirectory() as tmp:
        paths = golden_lib.make_fixtures(tmp)
        snapshots = golden_lib.snapshot(paths)
    golden_lib.save(snapshots)
    for name, snap in snapshots.items():
        print(f"{name}: {snap['len']} frames")


if __name__ == "__main__":
    main()
