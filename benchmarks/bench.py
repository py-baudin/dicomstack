""" dicomstack benchmark (not run in CI)

    python benchmarks/make_data.py           # once
    python benchmarks/bench.py [data] [--scenario NAME ...]

Each scenario runs in a fresh process: it times the load, then the operation,
and reports the process's peak memory.
For cold-cache numbers on Linux, run `sync; echo 3 > /proc/sys/vm/drop_caches` (root)
before each scenario.
"""

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = [
    "EchoTime", "RepetitionTime", "SeriesNumber", "InstanceNumber",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "Rows", "Columns", "StudyDate", "StudyTime",
    "SeriesDescription", "PatientID", "Modality", "Manufacturer",
    "FlipAngle", "SOPInstanceUID", "SeriesInstanceUID", "AcquisitionTime",
]


def op_first_access(stack):
    stack["EchoTime"]


def op_20_fields(stack):
    for field in FIELDS:
        stack.get_field_values(field, ignore_missing=True)


def op_filter(stack):
    stack(SeriesNumber=[2, 3], EchoTime=10.0)


def op_query(stack):
    from dicomstack import DICOM
    stack((DICOM.SeriesNumber > 5) & DICOM.EchoTime.is_in([20.0]))


def op_unique(stack):
    stack.unique("SeriesNumber", "EchoTime")


def op_as_volume(stack):
    stack.as_volume(by=["SeriesNumber", "EchoTime"])


def op_describe(stack):
    stack.describe()


# scenario: (path relative to data folder, operation)
SCENARIOS = {
    "load": ("stack", None),
    "first_access": ("stack", op_first_access),
    "20_fields": ("stack", op_20_fields),
    "filter": ("stack", op_filter),
    "query": ("stack", op_query),
    "unique": ("stack", op_unique),
    "as_volume": ("stack", op_as_volume),
    "describe": ("stack", op_describe),
    "zip_first_access": ("stack.zip", op_first_access),
    "multi_first_access": ("multi", op_first_access),
}


def peak_memory_mb():
    """peak resident memory of the current process (MB)"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        # declare HANDLE types: the default (32-bit int) truncates the pseudo-handle
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 2**20

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # kB on Linux, bytes on macOS
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run_scenario(name, data):
    """run a single scenario in this process, print json results"""
    from dicomstack import DicomStack

    path, operation = SCENARIOS[name]
    t0 = time.perf_counter()
    stack = DicomStack(os.path.join(data, path))
    t1 = time.perf_counter()
    if operation:
        operation(stack)
    t2 = time.perf_counter()
    result = {
        "scenario": name,
        "frames": len(stack),
        "load_s": round(t1 - t0, 3),
        "op_s": round(t2 - t1, 3) if operation else None,
        "total_s": round(t2 - t0, 3),
        "peak_mb": round(peak_memory_mb(), 1),
    }
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("data", nargs="?", default=os.path.join(HERE, "data"))
    parser.add_argument("--scenario", nargs="*", choices=list(SCENARIOS))
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        run_scenario(args.run, args.data)
        return

    if not os.path.isdir(os.path.join(args.data, "stack")):
        sys.exit(f"No benchmark data in {args.data}: run benchmarks/make_data.py")

    print(f"{'scenario':<20} {'frames':>6} {'load (s)':>9} {'op (s)':>8} {'total (s)':>10} {'peak (MB)':>10}")
    for name in args.scenario or SCENARIOS:
        out = subprocess.run(
            [sys.executable, __file__, args.data, "--run", name],
            capture_output=True, text=True, check=True,
        ).stdout
        res = json.loads(out.strip().splitlines()[-1])
        op = "-" if res["op_s"] is None else f"{res['op_s']:.2f}"
        print(
            f"{name:<20} {res['frames']:>6} {res['load_s']:>9.2f} {op:>8} "
            f"{res['total_s']:>10.2f} {res['peak_mb']:>10.0f}"
        )


if __name__ == "__main__":
    main()
