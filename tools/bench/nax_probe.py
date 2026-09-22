import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

# Which Metal kernel does a q8 matmul of the shape the Krea2 DiT actually issues
# land on? On hardware with GPU neural accelerators MLX dispatches its "nax"
# kernels; the name shows up in a Metal capture. A pipeline name is only written
# to the trace the first time it is created, so each dtype needs a fresh
# process: with no argument this script re-execs itself once per dtype.
#
#   MTL_CAPTURE_ENABLED=1 uv run python tools/bench/nax_probe.py


class NaxProbe:
    M, K, N = 4126, 6144, 16384  # Krea2 DiT at 1024^2: 4096 image + 30 text tokens, FFN up
    TRACE = Path("/tmp/mlx_nax_probe.gputrace")

    @classmethod
    def kernels(cls):
        # The pipeline names live in the trace's small bookkeeping files; the
        # multi-hundred-MB ones are the captured buffers and the whole metallib,
        # which lists every kernel that exists and would tell us nothing.
        small = [str(f) for f in cls.TRACE.rglob("*") if f.is_file() and f.stat().st_size < 1 << 20]
        if not small:
            return ["(no bookkeeping files in trace - needs a macOS whose Metal capture writes them)"]
        out = subprocess.run(
            ["sh", "-c", f"strings -a {' '.join(small)} | grep -E '^[a-z][a-zA-Z0-9_]{{15,}}$' | sort -u"],
            capture_output=True,
            text=True,
        )
        return out.stdout.split() or ["(none)"]

    @classmethod
    def probe(cls, dtype_name):
        dt = getattr(mx, dtype_name)
        x = mx.random.normal((1, cls.M, cls.K)).astype(dt)
        w = mx.random.normal((cls.N, cls.K)).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
        del w
        mx.eval(x, wq, scales, biases)

        def run():
            return mx.quantized_matmul(x, wq, scales=scales, biases=biases, transpose=True, group_size=64, bits=8)

        # Capture the cold dispatch: a pipeline's name is written to the trace
        # only when it is created, so a warmed-up run captures nothing.
        shutil.rmtree(cls.TRACE, ignore_errors=True)
        mx.metal.start_capture(str(cls.TRACE))
        mx.eval(run())
        mx.metal.stop_capture()

        for _ in range(3):
            mx.eval(run())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(run())
        mx.synchronize()
        elapsed = (time.perf_counter() - t0) / 10

        flops = 2 * cls.M * cls.K * cls.N
        print(f"{dtype_name:>10} {elapsed * 1e3:8.2f} ms {flops / elapsed / 1e12:7.2f} TFLOPS  {cls.kernels()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        NaxProbe.probe(sys.argv[1])
    else:
        print(mx.device_info())
        print(f"q8 matmul {NaxProbe.M}x{NaxProbe.K} @ {NaxProbe.K}x{NaxProbe.N}, group_size 64")
        for name in ("float32", "bfloat16", "float16"):
            subprocess.run([sys.executable, __file__, name], check=True)
