"""GPU selection rule for every GPU job in this repo.

A GPU may be used only if
  1. it has at most MAX_USED_MB (20 GB) already in use, and
  2. its free memory is at least 2x the memory the job needs.

Call ``configure_gpu(...)`` before torch (or anything that imports torch, e.g.
``settings.py``) is imported: CUDA reads CUDA_VISIBLE_DEVICES once, at its first
device query, and ignores later changes. CUDA_DEVICE_ORDER is set to PCI_BUS_ID
so the indices here match nvidia-smi.

CLI (for shell scripts):  python gpu_select.py --need-mb 4000   -> prints an index

On a machine without NVIDIA GPUs (no ``nvidia-smi``), ``configure_gpu`` returns None
and the job runs on the CPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

MAX_USED_MB = 20 * 1024
HEADROOM_FACTOR = 2.0
DEFAULT_NEED_MB = 4000


@dataclass
class GPU:
    index: int
    total_mb: int
    used_mb: int
    free_mb: int

    def eligible(self, need_mb: int) -> bool:
        return self.used_mb <= MAX_USED_MB and self.free_mb >= HEADROOM_FACTOR * need_mb

    def reason(self, need_mb: int) -> str:
        if self.used_mb > MAX_USED_MB:
            return f"{self.used_mb} MiB used > {MAX_USED_MB} MiB limit"
        if self.free_mb < HEADROOM_FACTOR * need_mb:
            return f"{self.free_mb} MiB free < {HEADROOM_FACTOR:g} x {need_mb} MiB needed"
        return "ok"


def query_gpus() -> List[GPU]:
    """Return the NVIDIA GPUs of this machine, or an empty list if there are none."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, total, used, free = (int(x.strip()) for x in line.split(","))
        gpus.append(GPU(idx, total, used, free))
    return gpus


def select_gpu(need_mb: int, requested: Optional[List[int]] = None, gpus: Optional[List[GPU]] = None) -> int:
    """Return the physical index of a GPU that satisfies the rule, or raise."""
    gpus = gpus if gpus is not None else query_gpus()
    by_index = {g.index: g for g in gpus}
    candidates = [by_index[i] for i in requested if i in by_index] if requested else gpus
    if requested and len(candidates) != len(requested):
        missing = sorted(set(requested) - set(by_index))
        raise RuntimeError(f"Requested GPU(s) {missing} not present; available: {sorted(by_index)}")
    ok = [g for g in candidates if g.eligible(need_mb)]
    if not ok:
        detail = "; ".join(f"GPU {g.index}: {g.reason(need_mb)}" for g in candidates)
        raise RuntimeError(f"No GPU satisfies the rule for a {need_mb} MiB job ({detail})")
    return max(ok, key=lambda g: g.free_mb).index


def configure_gpu(need_mb: int = DEFAULT_NEED_MB, requested: Optional[List[int]] = None) -> Optional[int]:
    """Select a GPU by the rule above. Returns None (CPU) on machines without NVIDIA GPUs."""
    if "torch" in sys.modules:
        raise RuntimeError("configure_gpu() must run before torch is imported, or CUDA_VISIBLE_DEVICES is ignored.")
    gpus = query_gpus()
    if not gpus and not requested:
        print("[gpu_select] no NVIDIA GPU found, running on the CPU", flush=True)
        return None
    idx = select_gpu(need_mb, requested, gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"[gpu_select] using GPU {idx} (need {need_mb} MiB, rule: free >= {HEADROOM_FACTOR:g}x need, "
          f"used <= {MAX_USED_MB} MiB)", flush=True)
    return idx


def parse_device_list(value: Optional[str]) -> Optional[List[int]]:
    if value in (None, "", "auto"):
        return None
    return [int(v) for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need-mb", type=int, default=DEFAULT_NEED_MB)
    ap.add_argument("--devices", default=None, help="restrict to these indices, e.g. '1,2,3'")
    args = ap.parse_args()
    try:
        print(select_gpu(args.need_mb, parse_device_list(args.devices)))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
