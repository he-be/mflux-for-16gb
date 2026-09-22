import argparse
import csv
import ctypes
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Swap sentry for the low-memory experiments: runs a command, samples the machine's
# swap and compressor counters while it lives, and (by default) kills it if the run
# actually pushes pages out. Absolute "swap used" is not a usable signal on macOS -
# other processes leave swap allocated for hours - so the verdict is based on the
# swapout counter and the rise over the baseline taken just before the child starts.

PAGE_SIZE = 16384

# proc_pid_rusage(pid, RUSAGE_INFO_V0, buf). The struct is a 16 byte uuid followed by
# uint64s, and ri_phys_footprint is the eighth of them, so it sits at byte 72. Read this
# way rather than through the `footprint` tool, which rounds to whole GB once a process
# is that large: three different configurations all "peaked" at 13326 MB, which turned
# out to be 13 GB rounded plus the 14 MB uv launcher, not a ceiling. Validated against
# the tool on a 4 GB process, where it still prints MB: 4103570944 B vs 3914 MiB.
_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_RUSAGE_INFO_V0_SIZE = 96
_PHYS_FOOTPRINT_OFFSET = 72


@dataclass
class Sample:
    t: float
    swap_used_mb: float
    swap_delta_mb: float
    compressor_mb: float
    swapins: int
    swapouts: int
    swapouts_delta: int
    filebacked_mb: float
    anonymous_mb: float
    child_footprint_mb: float


class SwapWatch:
    def __init__(
        self,
        interval: float = 2.0,
        warn_delta_mb: float = 256.0,
        abort_delta_mb: float | None = 1024.0,
        csv_path: Path | None = None,
    ):
        self.interval = interval
        self.warn_delta_mb = warn_delta_mb
        self.abort_delta_mb = abort_delta_mb
        self.csv_path = csv_path
        self.samples: list[Sample] = []
        self.warned = False
        self.aborted = False

    def run(self, command: list[str]) -> int:
        base_swap, base_out = self._swap_used_mb(), self._vm_counters()["swapouts"]
        start = time.time()
        print(f"🛁 swapwatch: baseline swap used {base_swap:.0f} MB, swapouts {base_out} pages")
        print(f"   command: {' '.join(command)}")

        child = subprocess.Popen(command)
        writer, handle = self._open_csv()
        try:
            while child.poll() is None:
                self._sample(start, base_swap, base_out, child.pid, writer, handle)
                if self.aborted:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(self.interval)
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
        finally:
            if handle is not None:
                handle.close()

        self._report(base_swap, base_out)
        return 99 if self.aborted else (child.returncode or 0)

    def _sample(self, start, base_swap, base_out, pid, writer, handle) -> None:
        counters = self._vm_counters()
        used = self._swap_used_mb()
        footprint = self._footprint_mb(pid)
        sample = Sample(
            t=time.time() - start,
            swap_used_mb=used,
            swap_delta_mb=used - base_swap,
            compressor_mb=counters["compressor"] * PAGE_SIZE / 1e6,
            swapins=counters["swapins"],
            swapouts=counters["swapouts"],
            swapouts_delta=counters["swapouts"] - base_out,
            filebacked_mb=counters["filebacked"] * PAGE_SIZE / 1e6,
            anonymous_mb=counters["anonymous"] * PAGE_SIZE / 1e6,
            child_footprint_mb=footprint,
        )
        self.samples.append(sample)

        if writer is not None:
            writer.writerow(
                [
                    f"{sample.t:.1f}",
                    f"{sample.swap_used_mb:.1f}",
                    f"{sample.swap_delta_mb:.1f}",
                    f"{sample.compressor_mb:.1f}",
                    sample.swapins,
                    sample.swapouts,
                    sample.swapouts_delta,
                    f"{sample.filebacked_mb:.1f}",
                    f"{sample.anonymous_mb:.1f}",
                    f"{sample.child_footprint_mb:.1f}",
                ]
            )
            handle.flush()

        if not self.warned and sample.swap_delta_mb >= self.warn_delta_mb:
            self.warned = True
            print(f"⚠️  swapwatch: swap grew {sample.swap_delta_mb:.0f} MB at t={sample.t:.0f}s (footprint {sample.child_footprint_mb:.0f} MB)")  # fmt: skip

        if self.abort_delta_mb is not None and sample.swap_delta_mb >= self.abort_delta_mb:
            self.aborted = True
            print(f"🛑 swapwatch: swap grew {sample.swap_delta_mb:.0f} MB (limit {self.abort_delta_mb:.0f} MB) - killing the run")  # fmt: skip

    def _report(self, base_swap: float, base_out: int) -> None:
        if not self.samples:
            print("🛁 swapwatch: the command exited before the first sample")
            return
        peak_swap = max(s.swap_delta_mb for s in self.samples)
        peak_footprint = max(s.child_footprint_mb for s in self.samples)
        peak_comp = max(s.compressor_mb for s in self.samples)
        peak_file = max(s.filebacked_mb for s in self.samples)
        base_file = self.samples[0].filebacked_mb
        out_delta = self.samples[-1].swapouts - base_out
        in_delta = self.samples[-1].swapins - self.samples[0].swapins
        verdict = "SWAPPED" if (out_delta > 0 or peak_swap >= self.warn_delta_mb) else "clean"
        print("🛁 swapwatch summary")
        print(f"   duration          : {self.samples[-1].t:.0f} s ({len(self.samples)} samples)")
        print(f"   peak footprint    : {peak_footprint / 1000:.2f} GB (phys_footprint, sampled)")
        print(f"   swap used (base)  : {base_swap:.0f} MB")
        print(f"   swap rise (peak)  : {peak_swap:.0f} MB")
        print(f"   swapouts / swapins: {out_delta} / {in_delta} pages ({out_delta * PAGE_SIZE / 1e6:.0f} MB out)")
        print(f"   compressor (peak) : {peak_comp / 1000:.2f} GB")
        print(f"   page cache        : {base_file / 1000:.2f} -> {peak_file / 1000:.2f} GB (file-backed)")
        print(f"   verdict           : {verdict}")
        if self.csv_path is not None:
            print(f"   csv               : {self.csv_path}")

    def _open_csv(self):
        if self.csv_path is None:
            return None, None
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.csv_path.open("w", newline="")
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "swap_used_mb",
                "swap_delta_mb",
                "compressor_mb",
                "swapins",
                "swapouts",
                "swapouts_delta",
                "filebacked_mb",
                "anonymous_mb",
                "child_footprint_mb",
            ]  # fmt: skip
        )
        return writer, handle

    @staticmethod
    def _swap_used_mb() -> float:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        match = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
        if not match:
            return 0.0
        value = float(match.group(1))
        return value * 1024 if match.group(2) == "G" else value

    @staticmethod
    def _vm_counters() -> dict[str, int]:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        counters = {"swapins": 0, "swapouts": 0, "compressor": 0, "filebacked": 0, "anonymous": 0}
        for line in out.splitlines():
            if "Swapins" in line:
                counters["swapins"] = SwapWatch._parse_count(line)
            elif "Swapouts" in line:
                counters["swapouts"] = SwapWatch._parse_count(line)
            elif "occupied by compressor" in line:
                counters["compressor"] = SwapWatch._parse_count(line)
            elif "File-backed pages" in line:
                # The page cache. Reading a 13.6 GB checkpoint through mx.load fills this
                # alongside the Metal buffers holding the same bytes, which is the thing
                # to watch when the machine starts evicting.
                counters["filebacked"] = SwapWatch._parse_count(line)
            elif "Anonymous pages" in line:
                counters["anonymous"] = SwapWatch._parse_count(line)
        return counters

    @staticmethod
    def _parse_count(line: str) -> int:
        match = re.search(r"(\d+)\.?\s*$", line.strip())
        return int(match.group(1)) if match else 0

    @staticmethod
    def _footprint_mb(pid: int) -> float:
        # ps rss does not see MLX's buffers at all: a process holding a 4 GB mx array
        # reports 30 MB, because the allocation is an IOAccelerator region. phys_footprint
        # does count it, so it is the only honest per-process number here. Summed over the
        # process tree because every run goes through `uv run`, which execs the real
        # python as a grandchild.
        total = 0.0
        for target in SwapWatch._tree_pids(pid):
            value = SwapWatch._phys_footprint(target)
            if value is not None:
                total += value / 1e6
        return total

    @staticmethod
    def _phys_footprint(pid: int) -> int | None:
        buffer = (ctypes.c_uint8 * _RUSAGE_INFO_V0_SIZE)()
        if _LIBC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(0), ctypes.byref(buffer)) != 0:
            return None
        start = _PHYS_FOOTPRINT_OFFSET
        return int.from_bytes(bytes(buffer[start : start + 8]), "little")

    @staticmethod
    def _tree_pids(pid: int) -> list[int]:
        out = subprocess.run(["ps", "-Axo", "pid=,ppid="], capture_output=True, text=True).stdout
        children: dict[int, list[int]] = {}
        for line in out.splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            try:
                child, parent = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            children.setdefault(parent, []).append(child)

        pids: list[int] = []
        stack = [pid]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            pids.append(current)
            stack.extend(children.get(current, []))
        return pids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command under a swap sentry (macOS).",
        epilog="example: uv run python tools/swapwatch.py --csv docs/16gb/runs/q8.csv -- mflux-generate-krea2 ...",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between samples (default: 2)")
    parser.add_argument("--warn-delta-mb", type=float, default=256.0, help="warn once swap rises this far above the baseline")  # fmt: skip
    parser.add_argument("--abort-delta-mb", type=float, default=1024.0, help="kill the command at this rise (0 disables)")  # fmt: skip
    parser.add_argument("--csv", type=Path, default=None, help="write every sample to this CSV")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by the command to run")
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("no command given (put it after --)")

    watch = SwapWatch(
        interval=args.interval,
        warn_delta_mb=args.warn_delta_mb,
        abort_delta_mb=None if args.abort_delta_mb == 0 else args.abort_delta_mb,
        csv_path=args.csv,
    )
    return watch.run(command)


if __name__ == "__main__":
    sys.exit(main())
