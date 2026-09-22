import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Machine-state snapshot for the low-memory experiments (plan M1). Whether a 14 GB
# checkpoint fits is decided by how much memory is actually claimable at the moment a
# run starts, so every bench takes one of these before and after and files it next to
# the result. "Free" alone is not that number - macOS keeps most of RAM in the inactive
# and speculative queues and hands it back on demand - so the headline figure here is
# free + inactive + speculative + purgeable.

PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}


@dataclass
class Snapshot:
    label: str
    when: str
    page_size: int
    memsize_gb: float
    wired_limit_mb: int | None
    device: dict | None
    pages: dict[str, int]
    swap_mb: dict[str, float]
    pressure_level: int | None
    top_rss: list[list]

    @property
    def claimable_gb(self) -> float:
        keys = ("free", "inactive", "speculative", "purgeable")
        return sum(self.pages.get(k, 0) for k in keys) * self.page_size / 1e9

    @property
    def free_gb(self) -> float:
        return self.pages.get("free", 0) * self.page_size / 1e9

    @property
    def wired_gb(self) -> float:
        return self.pages.get("wired", 0) * self.page_size / 1e9

    @property
    def compressor_gb(self) -> float:
        return self.pages.get("compressor", 0) * self.page_size / 1e9

    @property
    def filebacked_gb(self) -> float:
        return self.pages.get("filebacked", 0) * self.page_size / 1e9

    def report(self) -> None:
        pressure = PRESSURE_LEVELS.get(self.pressure_level, str(self.pressure_level))
        print(f"📏 memstat {self.label or '(unlabeled)'} @ {self.when}")
        print(f"   hw.memsize        : {self.memsize_gb:.2f} GB")
        if self.device is not None:
            rec = self.device.get("max_recommended_working_set_size", 0) / 1e9
            buf = self.device.get("max_buffer_length", 0) / 1e9
            print(f"   GPU               : {self.device.get('device_name')} (working set {rec:.2f} GB, max buffer {buf:.2f} GB)")  # fmt: skip
        print(f"   iogpu.wired_limit : {self.wired_limit_mb} MB")
        print(f"   claimable         : {self.claimable_gb:.2f} GB  (free {self.free_gb:.2f})")
        print(f"   wired down        : {self.wired_gb:.2f} GB")
        print(f"   compressor        : {self.compressor_gb:.2f} GB")
        print(f"   page cache        : {self.filebacked_gb:.2f} GB (file-backed)")
        print(f"   swap used / total : {self.swap_mb.get('used', 0):.0f} / {self.swap_mb.get('total', 0):.0f} MB")  # fmt: skip
        print(f"   swapins/swapouts  : {self.pages.get('swapins', 0)} / {self.pages.get('swapouts', 0)} pages")
        print(f"   memory pressure   : {pressure}")
        if self.top_rss:
            print("   largest processes :")
            for name, rss_mb in self.top_rss:
                print(f"     {rss_mb / 1000:6.2f} GB  {name}")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {
            "claimable_gb": round(self.claimable_gb, 3),
            "free_gb": round(self.free_gb, 3),
            "wired_gb": round(self.wired_gb, 3),
            "compressor_gb": round(self.compressor_gb, 3),
            "filebacked_gb": round(self.filebacked_gb, 3),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")


class MemStat:
    @staticmethod
    def capture(label: str = "", top: int = 6, with_device: bool = True) -> Snapshot:
        page_size, pages = MemStat._vm_stat()
        return Snapshot(
            label=label,
            when=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            page_size=page_size,
            memsize_gb=MemStat._sysctl_int("hw.memsize") / 1e9,
            wired_limit_mb=MemStat._sysctl_int("iogpu.wired_limit_mb"),
            device=MemStat._device_info() if with_device else None,
            pages=pages,
            swap_mb=MemStat._swap_mb(),
            pressure_level=MemStat._sysctl_int("kern.memorystatus_vm_pressure_level"),
            top_rss=MemStat._top_rss(top),
        )

    @staticmethod
    def _sysctl_int(name: str) -> int | None:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
        try:
            return int(out.stdout.strip())
        except ValueError:
            return None

    @staticmethod
    def _vm_stat() -> tuple[int, dict[str, int]]:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        header = re.search(r"page size of (\d+) bytes", out)
        page_size = int(header.group(1)) if header else 16384
        wanted = {
            "Pages free": "free",
            "Pages active": "active",
            "Pages inactive": "inactive",
            "Pages speculative": "speculative",
            "Pages wired down": "wired",
            "Pages purgeable": "purgeable",
            "Pages occupied by compressor": "compressor",
            "File-backed pages": "filebacked",
            "Anonymous pages": "anonymous",
            "Swapins": "swapins",
            "Swapouts": "swapouts",
        }
        pages: dict[str, int] = {}
        for line in out.splitlines():
            key, _, value = line.partition(":")
            name = wanted.get(key.strip())
            if name is not None:
                digits = re.search(r"(\d+)", value)
                if digits:
                    pages[name] = int(digits.group(1))
        return page_size, pages

    @staticmethod
    def _swap_mb() -> dict[str, float]:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        result: dict[str, float] = {}
        for name in ("total", "used", "free"):
            match = re.search(rf"{name}\s*=\s*([\d.]+)([MG])", out)
            if match:
                value = float(match.group(1))
                result[name] = value * 1024 if match.group(2) == "G" else value
        return result

    @staticmethod
    def _device_info() -> dict | None:
        try:
            import mlx.core as mx
        except ImportError:
            return None
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return {k: v for k, v in info.items() if isinstance(v, (int, float, str))}

    @staticmethod
    def _top_rss(count: int) -> list[list]:
        out = subprocess.run(["ps", "-Axo", "rss=,comm="], capture_output=True, text=True).stdout
        rows: list[tuple[str, float]] = []
        for line in out.splitlines():
            rss, _, comm = line.strip().partition(" ")
            try:
                rows.append((Path(comm.strip()).name, int(rss) / 1024))
            except ValueError:
                continue
        rows.sort(key=lambda row: row[1], reverse=True)
        return [[name, round(rss, 1)] for name, rss in rows[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot the machine's memory state (macOS).",
        epilog="example: uv run python tools/bench/memstat.py --label clean-boot --json docs/16gb/runs/m1-baseline.json",  # fmt: skip
    )
    parser.add_argument("--label", default="", help="name for this snapshot (e.g. clean-boot, after-m3)")
    parser.add_argument("--json", type=Path, default=None, help="also write the snapshot to this JSON file")
    parser.add_argument("--top", type=int, default=6, help="how many of the largest processes to list")
    parser.add_argument("--no-device", action="store_true", help="skip the MLX device query (does not import mlx)")
    args = parser.parse_args()

    snapshot = MemStat.capture(label=args.label, top=args.top, with_device=not args.no_device)
    snapshot.report()
    if args.json is not None:
        snapshot.write(args.json)
        print(f"   json              : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
