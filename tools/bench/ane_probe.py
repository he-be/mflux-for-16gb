import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from multiprocessing import get_context, shared_memory
from pathlib import Path

import numpy as np

# Plan M11a: is the Neural Engine worth adding as a second compute unit beside the GPU for
# the Krea2 DiT? The candidate split (plan section 2) hands the ANE a share of the MLP's
# columns: gate/up (6144->16384) for its columns, silu(g)*u, and the down projection's
# partial sum over those columns, one 50 MB handoff in and one out per block. This measures
# the pieces that decide it, on the M6 mini with the real block 0 weights: the ANE's TOPS on
# the three matmul shapes and the fused MLP, whether a second process (the second ANE) adds
# throughput, what the GPU loses when both run, the cost of one handoff through shared
# memory, and how far fp16 activations move the numbers against the bf16 GPU path. The Core
# ML side lives in a worker process, as it would in production (predict holds the GIL).
#
# coremltools 9.0 has no native extensions for Python 3.14 (it imports, but every proxy is
# missing and the convert dies at "BlobWriter not loaded"), so pin 3.13 and keep the project's
# own venv out of the way:
#
#   UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 --with coremltools \
#       python tools/bench/ane_probe.py --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram
#   ... --only single --shapes gate,mlp --variants row
#   ... --only actstats     # a real 1024^2 run
#
# See docs/16gb/measurements/2026-09-22-m11a-ane-probe.md.

FEATURES, MLPDIM = 6144, 16384
TOKENS = {"1024": 4126, "1280": 6430}
# name -> (kind, K, N); the fused MLP kinds carry the share of columns the ANE would own
SHAPES = {
    "wq": ("linear", FEATURES, FEATURES),
    "gate": ("linear", FEATURES, MLPDIM),
    "down": ("linear", MLPDIM, FEATURES),
    "mlp": ("mlp", FEATURES, MLPDIM),
    "mlp57": ("mlp", FEATURES, 9344),  # a = 0.57 of 16384, rounded to a multiple of 128
}
VARIANTS = ("fp16", "row", "g64", "a8")

_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


class Footprint:
    # ri_phys_footprint out of proc_pid_rusage, the only per-process number that counts
    # IOAccelerator / ANE buffers (tools/swapwatch.py).
    @staticmethod
    def mb(pid: int | None = None) -> float:
        buffer = ctypes.create_string_buffer(96)
        if _LIBC.proc_pid_rusage(ctypes.c_int(pid or os.getpid()), ctypes.c_int(0), ctypes.byref(buffer)) != 0:
            return float("nan")
        return int.from_bytes(bytes(buffer[72:80]), "little") / 1e6


class AneLog:
    # Core ML never reports an ANE compile failure through its API: the model silently runs on
    # the CPU. The unified log does (Irodori-TTS, docs/experiments/17-m1-ane-factors.md).
    MARKS = ("Model load failed", "Compilation failed", "Register spiller failure")

    @staticmethod
    def show(start: datetime, end: datetime) -> dict:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            out = subprocess.run(
                [
                    "/usr/bin/log", "show", "--start", start.strftime(fmt), "--end", end.strftime(fmt),
                    "--predicate", 'process == "aned" OR process == "ANECompilerService"', "--style", "compact",
                ],
                capture_output=True, text=True, timeout=180.0,
            ).stdout  # fmt: skip
        except (OSError, subprocess.SubprocessError) as e:
            return {"error": str(e)}
        lines = [ln for ln in out.splitlines() if ln and not ln.startswith("Timestamp")]
        counts = {m: sum(m in ln for ln in lines) for m in AneLog.MARKS}
        return {"lines": len(lines), "failures": counts, "tail": lines[-5:]}


class Quant:
    # Symmetric int8, per row or per 64-wide block along K. The GPU's q8 is affine (uint8,
    # scale + float bias per group of 64); Core ML's integer zero point cannot express that
    # bias exactly, so the ANE weights are a re-quantization of the dequantized q8 weight.
    @staticmethod
    def int8(w: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        n, k = w.shape
        w32 = w.astype(np.float32)
        if mode == "row":
            amax = np.abs(w32).max(axis=1, keepdims=True)
            scale = np.maximum(amax, 1e-8) / 127.0
            q = np.clip(np.rint(w32 / scale), -127, 127).astype(np.int8)
            return q, scale.astype(np.float16)
        blocks = w32.reshape(n, k // 64, 64)
        amax = np.abs(blocks).max(axis=2, keepdims=True)
        scale = np.maximum(amax, 1e-8) / 127.0
        q = np.clip(np.rint(blocks / scale), -127, 127).astype(np.int8).reshape(n, k)
        return q, scale.reshape(n, k // 64).astype(np.float16)

    @staticmethod
    def dequant(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
        n, k = q.shape
        s = scale.astype(np.float32)
        if s.shape[1] == 1:
            return q.astype(np.float32) * s
        return (q.reshape(n, k // 64, 64).astype(np.float32) * s[:, :, None]).reshape(n, k)


class Converter:
    # Builds the Core ML program straight from MIL (no torch), with the weights as
    # constexpr int8 tensors so the file holds what the ANE will read.
    def __init__(self, cache: Path, layout: str):
        self.cache = cache
        self.layout = layout
        cache.mkdir(parents=True, exist_ok=True)

    def path(self, shape: str, m: int, variant: str) -> Path:
        return self.cache / f"{shape}-m{m}-{variant}-{self.layout}.mlmodelc"

    def build(self, shape: str, m: int, variant: str, weights: dict[str, np.ndarray], calib: np.ndarray | None) -> dict:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types

        kind, k, n = SHAPES[shape]
        mlmodelc = self.path(shape, m, variant)
        info = {"path": str(mlmodelc)}
        if mlmodelc.exists():
            info["cached"] = True
            return info
        wmode = "row" if variant == "a8" else variant
        conv = self.layout == "conv"

        def const(name: str, cols: int | None = None):
            w = weights[name]
            if cols is not None:
                w = w[:cols] if name != "mlp.down" else w[:, :cols]
            w = np.ascontiguousarray(w)
            if wmode == "fp16":
                data = w.astype(np.float16)
                return data.reshape(*data.shape, 1, 1) if conv else data
            q, scale = Quant.int8(w, wmode)
            if conv:
                q, scale = q.reshape(*q.shape, 1, 1), scale.reshape(*scale.shape, 1, 1)
            return mb.constexpr_blockwise_shift_scale(data=q, scale=scale)

        def linear(x, w, name=None):
            extra = {"name": name} if name else {}
            return mb.conv(x=x, weight=w, **extra) if conv else mb.linear(x=x, weight=w, **extra)

        in_shape = (1, k, 1, m) if conv else (m, k)

        @mb.program(input_specs=[mb.TensorSpec(shape=in_shape, dtype=types.fp16)], opset_version=ct.target.macOS15)
        def prog(x):
            if kind == "linear":
                return linear(x, const({"wq": "attn.wq", "gate": "mlp.gate", "down": "mlp.down"}[shape]), name="y")
            g = linear(x, const("mlp.gate", n))
            u = linear(x, const("mlp.up", n))
            h = mb.mul(x=mb.silu(x=g), y=u)
            return linear(h, const("mlp.down", n), name="y")

        t0 = time.perf_counter()
        kwargs = dict(
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=variant != "a8",
        )
        try:
            model = ct.convert(
                prog,
                inputs=[ct.TensorType(name="x", shape=in_shape, dtype=np.float16)],
                outputs=[ct.TensorType(name="y", dtype=np.float16)],
                **kwargs,
            )
            info["io"] = "fp16"
        except Exception as e:  # noqa: BLE001 - fall back to fp32 I/O, and say so
            info["io"] = f"fp32 (fp16 I/O refused: {str(e).splitlines()[0][:120]})"
            model = ct.convert(prog, **kwargs)
        if variant == "a8":
            from coremltools.optimize.coreml import (
                OpActivationLinearQuantizerConfig,
                OptimizationConfig,
                linear_quantize_activations,
            )

            config = OptimizationConfig(global_config=OpActivationLinearQuantizerConfig(mode="linear_symmetric"))
            sample = calib if calib is not None else np.random.normal(size=(m, k)).astype(np.float16)
            if conv:
                sample = sample.T.reshape(1, k, 1, m)
            model = linear_quantize_activations(model, config, [{"x": sample}])
        info["convert_s"] = time.perf_counter() - t0
        pkg = mlmodelc.with_suffix(".mlpackage")
        model.save(str(pkg))
        t0 = time.perf_counter()
        ct.models.utils.compile_model(str(pkg), destination_path=str(mlmodelc))
        info["compile_s"] = time.perf_counter() - t0
        info["package_mb"] = sum(f.stat().st_size for f in mlmodelc.rglob("*") if f.is_file()) / 1e6
        return info


# -- the Core ML worker ----------------------------------------------------------------------


class WorkerLoop:
    @staticmethod
    def serve(conn, x_name: str, y_name: str, units: str) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import coremltools as ct

        unit = {"all": ct.ComputeUnit.ALL, "ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[units]
        shm_x, shm_y = shared_memory.SharedMemory(name=x_name), shared_memory.SharedMemory(name=y_name)
        models: list = []
        shape = {}

        def view(block, shp):
            return np.ndarray(shp, dtype=np.float16, buffer=block.buf)

        def predict(model, x):
            return model.predict({"x": x})["y"]

        def plan(path: str) -> dict:
            try:
                from coremltools.models.compute_plan import MLComputePlan

                p = MLComputePlan.load_from_path(path=path, compute_units=unit)
                devices: dict[str, int] = {}
                cores = None
                for fn in p.model_structure.program.functions.values():
                    for op in fn.block.operations:
                        usage = p.get_compute_device_usage_for_mlprogram_operation(op)
                        if usage is None:
                            continue
                        dev = usage.preferred_compute_device
                        key = type(dev).__name__.replace("ML", "").replace("ComputeDevice", "")
                        devices[f"{op.operator_name}->{key}"] = devices.get(f"{op.operator_name}->{key}", 0) + 1
                        cores = getattr(dev, "total_core_count", cores)
                return {"ops": devices, "ane_cores": cores}
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)[:200]}

        try:
            while True:
                msg = conn.recv()
                kind = msg[0]
                try:
                    if kind == "quit":
                        conn.send(("ok", None))
                        break
                    if kind == "load":
                        _, path, in_shape, out_shape = msg
                        shape["in"], shape["out"] = tuple(in_shape), tuple(out_shape)
                        start = datetime.now()
                        t0 = time.perf_counter()
                        models.append(ct.models.CompiledMLModel(path, compute_units=unit))
                        sec = time.perf_counter() - t0
                        info = {"load_s": sec, "footprint_mb": Footprint.mb()}
                        if len(models) == 1:
                            info["plan"] = plan(path)
                            if sec > 2.0:
                                time.sleep(1.0)
                                info["log"] = AneLog.show(start, datetime.now())
                        conn.send(("ok", info))
                    elif kind == "bench":
                        # Predict from a fixed input already in the worker: the ANE's own speed.
                        _, warmup, iters, threads = msg
                        x = view(shm_x, shape["in"]).copy()
                        for _ in range(warmup):
                            predict(models[0], x)
                        if threads <= 1:
                            t0 = time.perf_counter()
                            for _ in range(iters):
                                predict(models[0], x)
                            wall = time.perf_counter() - t0
                        else:
                            # The same package twice in one process, one thread each: does one
                            # process alone reach the second Neural Engine?
                            def run(model):
                                for _ in range(iters):
                                    predict(model, x)

                            ts = [threading.Thread(target=run, args=(models[i % len(models)],)) for i in range(threads)]
                            t0 = time.perf_counter()
                            for t in ts:
                                t.start()
                            for t in ts:
                                t.join()
                            wall = time.perf_counter() - t0
                        conn.send(
                            ("ok", {"wall_s": wall, "calls": iters * max(threads, 1), "footprint_mb": Footprint.mb()})
                        )
                    elif kind == "step":
                        # One production-shaped call: read x out of shared memory, write y back.
                        t0 = time.perf_counter()
                        out = predict(models[0], view(shm_x, shape["in"]))
                        t1 = time.perf_counter()
                        np.copyto(view(shm_y, shape["out"]), out.reshape(shape["out"]), casting="same_kind")
                        conn.send(("ok", {"predict_s": t1 - t0, "copy_out_s": time.perf_counter() - t1}))
                    else:
                        conn.send(("err", f"unknown message {kind!r}"))
                except Exception as e:  # noqa: BLE001 - report, keep serving
                    import traceback

                    conn.send(("err", f"{e}\n{traceback.format_exc()[-800:]}"))
        finally:
            shm_x.close()
            shm_y.close()


class Worker:
    def __init__(self, in_shape: tuple, out_shape: tuple, units: str = "all"):
        self.in_shape, self.out_shape = in_shape, out_shape
        n_in, n_out = int(np.prod(in_shape)) * 2, int(np.prod(out_shape)) * 2
        self.shm_x = shared_memory.SharedMemory(create=True, size=n_in)
        self.shm_y = shared_memory.SharedMemory(create=True, size=n_out)
        ctx = get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=WorkerLoop.serve, args=(child, self.shm_x.name, self.shm_y.name, units), daemon=True
        )
        self.proc.start()

    @property
    def x(self) -> np.ndarray:
        return np.ndarray(self.in_shape, dtype=np.float16, buffer=self.shm_x.buf)

    @property
    def y(self) -> np.ndarray:
        return np.ndarray(self.out_shape, dtype=np.float16, buffer=self.shm_y.buf)

    def send(self, *msg) -> None:
        self.conn.send(msg)

    def recv(self) -> dict:
        status, info = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"worker: {info}")
        return info

    def call(self, *msg) -> dict:
        self.send(*msg)
        return self.recv()

    def load(self, path: Path) -> dict:
        return self.call("load", str(path), self.in_shape, self.out_shape)

    def step(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        np.copyto(self.x, x.reshape(self.in_shape))
        info = self.call("step")
        info["round_trip_s"] = time.perf_counter() - t0
        return self.y.copy(), info

    def footprint_mb(self) -> float:
        return Footprint.mb(self.proc.pid)

    def close(self) -> None:
        try:
            self.call("quit")
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(timeout=10)
        for block in (self.shm_x, self.shm_y):
            block.close()
            block.unlink()


# -- the GPU side ---------------------------------------------------------------------------


class GpuLoad:
    # The 6144->16384 q8 matmul in a loop, one epoch-stamped TFLOPS line per window, so a
    # parent can read off what the GPU did while the ANE was busy.
    @staticmethod
    def run(seconds: float, m: int) -> None:
        import mlx.core as mx

        x = mx.random.normal((1, m, FEATURES)).astype(mx.bfloat16)
        q = mx.quantize(mx.random.normal((MLPDIM, FEATURES)).astype(mx.bfloat16), group_size=64, bits=8)
        mx.eval(x, *q)
        flops = 2 * m * FEATURES * MLPDIM

        def step():
            return mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=8)

        for _ in range(3):
            mx.eval(step())
        mx.synchronize()
        end = time.time() + seconds
        while time.time() < end:
            t0 = time.time()
            for _ in range(8):
                mx.eval(step())
            mx.synchronize()
            t1 = time.time()
            print(f"window {t0:.3f} {t1:.3f} {8 * flops / (t1 - t0) / 1e12:.2f}", flush=True)

    @staticmethod
    def start(seconds: float, m: int) -> subprocess.Popen:
        cmd = [sys.executable, __file__, "--only", "gpu-load", "--seconds", str(seconds), "--m", str(m)]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})

    @staticmethod
    def windows(proc: subprocess.Popen) -> list[tuple[float, float, float]]:
        out, _ = proc.communicate()
        rows = []
        for line in out.splitlines():
            if line.startswith("window "):
                _, a, b, t = line.split()
                rows.append((float(a), float(b), float(t)))
        return rows

    @staticmethod
    def mean_within(rows, start: float, end: float) -> float | None:
        inside = [t for a, b, t in rows if a >= start and b <= end]
        return sum(inside) / len(inside) if inside else None


# -- weights and the reference path ----------------------------------------------------------


class AneLoad:
    # GpuLoad's mirror. Hammer one package and print epoch-stamped windows, so a parent that is
    # busy with something else can read off what the ANE managed while it was.
    @staticmethod
    def run(path: Path, seconds: float, in_shape: tuple, units: str, flops: float) -> None:
        import coremltools as ct

        unit = {"all": ct.ComputeUnit.ALL, "ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[units]
        model = ct.models.CompiledMLModel(str(path), compute_units=unit)
        x = (np.random.normal(size=in_shape) * 0.5).astype(np.float16)
        for _ in range(3):
            model.predict({"x": x})
        print(f"ready {time.time():.3f}", flush=True)
        end = time.time() + seconds
        while time.time() < end:
            t0 = time.time()
            for _ in range(4):
                model.predict({"x": x})
            t1 = time.time()
            print(f"window {t0:.3f} {t1:.3f} {4 * flops / (t1 - t0) / 1e12:.2f}", flush=True)

    @staticmethod
    def start(args, shape: str, variant: str, m: int, seconds: float) -> subprocess.Popen:
        cmd = [
            sys.executable, __file__, "--only", "ane-load", "--seconds", str(seconds), "--shapes", shape,
            "--variants", variant, "--m", str(m), "--layout", args.layout, "--units", args.units,
            "--cache", str(args.cache),
        ]  # fmt: skip
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


class RealCo:
    # M11a decided on a co-tenant that was a back-to-back quantized_matmul over resident random
    # weights at full duty - the most hostile neighbour there is, and not what the DiT does.
    # This runs the real generation instead, with and without the ANE hammering the shape it
    # would own, and reads s/step off the same tqdm line the other measurements quote.
    #
    # Two co-tenants at the same duty and different working sets (mlp57 172 MB, wq 38 MB)
    # separate a shared-cache story from a power/clock one without needing sudo.
    STEP = re.compile(r"([0-9.]+)s/it")
    PROMPT = (
        "a photograph of a weathered brass diving helmet on a workshop bench, "
        "morning light through a dusty window, shallow depth of field"
    )

    # Swapouts say whether the machine gave up; pageins say whether the streamed block weights
    # stopped coming out of the page cache, which is the thing a resident co-tenant would break.
    COUNTERS = ("Swapouts", "Pageins")

    @staticmethod
    def vm_counters() -> dict[str, int]:
        out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
        found = {}
        for name in RealCo.COUNTERS:
            hit = re.search(rf"{name}:\s+(\d+)", out)
            found[name.lower()] = int(hit.group(1)) if hit else -1
        return found

    @staticmethod
    def generate(args, tag: str) -> dict:
        out = args.cache / f"realco-{tag}.png"
        cmd = [
            sys.executable, "-c", "from mflux.models.krea2.cli import krea2_generate; krea2_generate.main()",
            "--model", str(args.model), "--base-model", "krea-2", "--block-streaming", "--prompt", RealCo.PROMPT,
            "--seed", "42", "--steps", str(args.steps), "--scheduler", "euler", "--guidance", "1.0",
            "--width", str(args.width), "--height", str(args.width), "--no-metadata", "--output", str(out),
        ]  # fmt: skip
        before = RealCo.vm_counters()
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        end = time.time()
        after = RealCo.vm_counters()
        hits = RealCo.STEP.findall(proc.stderr)
        return {
            "s_per_step": float(hits[-1]) if hits else float("nan"),
            "wall_s": end - start,
            "start": start,
            "end": end,
            "swapouts_delta": after["swapouts"] - before["swapouts"],
            "pagein_mb": (after["pageins"] - before["pageins"]) * 16384 / 1e6,
            "rc": proc.returncode,
            "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.returncode else [],
        }

    @staticmethod
    def run(args) -> None:
        print("\n== realco: a real generation, alone and with the ANE hammering beside it ==")
        results: dict = {}
        # baselines are interleaved so drift can be bounded: "base,wq,base,mlp57,base"
        seen: dict = {}
        cases = []
        for name in args.cases.split(","):
            tag = name if name not in seen else f"{name}{seen[name] + 1}"
            seen[name] = seen.get(name, 0) + 1
            cases.append((tag, None if name == "base" else (name, "row")))
        for tag, co in cases:
            proc = None
            if co is not None:
                shape, variant = co
                path = Converter(args.cache, args.layout).path(shape, 4126, variant)
                if not path.exists():
                    print(f"  {tag:12s} SKIPPED: {path.name} is not converted")
                    continue
                proc = AneLoad.start(args, shape, variant, 4126, 900.0)
                time.sleep(12.0)  # load, warm up, reach steady state before the generation starts
            entry = RealCo.generate(args, tag)
            if proc is not None:
                entry["ane_footprint_mb"] = Footprint.mb(proc.pid)
                proc.terminate()
                rows = GpuLoad.windows(proc)
                entry["ane_tops_during"] = GpuLoad.mean_within(rows, entry["start"], entry["end"])
                entry["ane_windows"] = len(rows)
                entry["ane_package_mb"] = sum(f.stat().st_size for f in path.rglob("*")) / 1e6
            results[tag] = entry
            bases = [v["s_per_step"] for k, v in results.items() if k.startswith("base")]
            base = min(bases) if bases else None
            delta = f"  ({entry['s_per_step'] / base * 100 - 100:+.1f}%)" if base and co is not None else ""
            ane = f"  ANE {entry['ane_tops_during']:.2f} TOPS" if entry.get("ane_tops_during") else ""
            swap = f"  swapouts +{entry['swapouts_delta']}" if entry["swapouts_delta"] else ""
            swap += f"  pagein {entry['pagein_mb']:.0f} MB"
            fail = f"  rc={entry['rc']} {entry['stderr_tail']}" if entry["rc"] else ""
            print(f"  {tag:12s} {entry['s_per_step']:6.2f} s/step{delta}{ane}{swap}{fail}", flush=True)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(results, indent=2, default=str))
            print(f"wrote {args.json}")


class Block0:
    HEADS, KVHEADS, MULT, HEAD_DIM = 48, 12, 4, 128

    def __init__(self, model: Path):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream

        self.mx = mx
        self.tree = Krea2BlockStream(Krea2BlockStream.locate(model)).read(0)
        self.flat = dict(tree_flatten(self.tree))
        mx.eval(*self.flat.values())

    def dequantized(self, name: str) -> np.ndarray:
        mx = self.mx
        w, s, b = (self.flat[f"{name}.{p}"] for p in ("weight", "scales", "biases"))
        return np.array(mx.dequantize(w, s, b, group_size=64, bits=8).astype(mx.float16))

    def weights(self) -> dict[str, np.ndarray]:
        return {name: self.dequantized(name) for name in ("attn.wq", "mlp.gate", "mlp.up", "mlp.down")}

    def block(self):
        from mlx import nn

        from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock

        block = SingleStreamBlock(FEATURES, self.HEADS, self.MULT, False, self.KVHEADS)
        block.set_dtype(self.mx.bfloat16)
        nn.quantize(block, group_size=64, bits=8)
        block.update(self.tree)
        self.mx.eval(block.parameters())
        return block

    def mlp_input(self, block, tokens: int, seed: int = 0):
        # The MLP's input as the block computes it, from a normal x and modulation vector
        # (block_budget.py's stand-in). Real activation ranges come from --only actstats.
        mx = self.mx
        mx.random.seed(seed)
        x = mx.random.normal((1, tokens, FEATURES)).astype(mx.bfloat16)
        vec = mx.random.normal((1, 1, 6 * FEATURES)).astype(mx.bfloat16)
        _, _, _, postscale, postshift, _ = block.mod(vec)
        xin = (1 + postscale) * block.postnorm(x) + postshift
        mx.eval(xin)
        return xin


class Numerics:
    @staticmethod
    def bf16_ulp(ref: np.ndarray) -> np.ndarray:
        mag = np.maximum(np.abs(ref), np.finfo(np.float32).tiny)
        return np.exp2(np.floor(np.log2(mag)) - 7)

    @staticmethod
    def compare(out: np.ndarray, ref: np.ndarray) -> dict:
        out, ref = out.astype(np.float32).ravel(), ref.astype(np.float32).ravel()
        d = np.abs(out - ref)
        rel = d / (np.abs(ref) + 1e-6)
        return {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_rel": float(rel.max()),
            "mean_rel": float(rel.mean()),
            "over_1ulp_bf16": float((d > Numerics.bf16_ulp(ref)).mean()),
            "changed": float((d > 0).mean()),
        }


# -- the probe ------------------------------------------------------------------------------


class AneProbe:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: dict = {"args": vars(args) | {"model": str(args.model), "cache": str(args.cache)}}
        self.converter = Converter(args.cache, args.layout)
        self.block0 = Block0(args.model)
        self.weights = self.block0.weights()
        self.results["host"] = self._host()

    @staticmethod
    def _host() -> dict:
        out = {}
        for key in ("machdep.cpu.brand_string", "hw.memsize", "kern.osproductversion"):
            out[key] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        return out

    @staticmethod
    def flops(shape: str, m: int) -> float:
        kind, k, n = SHAPES[shape]
        return 2.0 * m * k * n * (3 if kind == "mlp" else 1)

    @staticmethod
    def io_shapes(shape: str, m: int, layout: str) -> tuple[tuple, tuple]:
        kind, k, n = SHAPES[shape]
        n_out = k if kind == "mlp" else n
        if layout == "conv":
            return (1, k, 1, m), (1, n_out, 1, m)
        return (m, k), (m, n_out)

    def row(self, label: str, text: str) -> None:
        print(f"  {label:36s} {text}", flush=True)

    def convert_all(self) -> None:
        print("\n== convert: real block 0 weights -> Core ML mlprogram (fp16 I/O) ==")
        calib = None
        if "a8" in self.args.variants:
            block = self.block0.block()
            calib = np.array(self.block0.mlp_input(block, 4126).astype(self.block0.mx.float16))[0]
        conv = self.results.setdefault("convert", {})
        for shape in self.args.shapes:
            for m in self.args.m:
                for variant in self.args.variants:
                    key = f"{shape}-m{m}-{variant}"
                    kind, k, _ = SHAPES[shape]
                    sample = calib if (kind == "mlp" or k == FEATURES) and m == 4126 else None
                    try:
                        info = self.converter.build(shape, m, variant, self.weights, sample)
                    except Exception as e:  # noqa: BLE001 - one variant failing must not stop the rest
                        info = {"error": str(e).splitlines()[0][:200]}
                    conv[key] = info
                    if "error" in info:
                        self.row(key, f"FAILED: {info['error']}")
                    elif info.get("cached"):
                        self.row(key, "cached")
                    else:
                        self.row(
                            key,
                            f"convert {info['convert_s']:6.1f} s  compile {info['compile_s']:5.1f} s  "
                            f"{info['package_mb']:7.1f} MB  io {info['io']}",
                        )

    def single(self) -> None:
        print("\n== single: one worker, ANE alone (warmup 3, mean of --iters) ==")
        print("   first load compiles for the ANE; the second load (fresh process) reads the OS cache")
        res = self.results.setdefault("single", {})
        for shape in self.args.shapes:
            for m in self.args.m:
                for variant in self.args.variants:
                    key = f"{shape}-m{m}-{variant}"
                    path = self.converter.path(shape, m, variant)
                    if not path.exists():
                        continue
                    in_shape, out_shape = self.io_shapes(shape, m, self.args.layout)
                    entry = {}
                    w = Worker(in_shape, out_shape, self.args.units)
                    try:
                        w.x[...] = np.random.normal(size=in_shape).astype(np.float16) * 0.5
                        first = w.load(path)
                        entry["first_load_s"] = first["load_s"]
                        entry["plan"] = first.get("plan")
                        entry["log"] = first.get("log")
                        bench = w.call("bench", 3, self.args.iters, 1)
                        per = bench["wall_s"] / bench["calls"]
                        entry["ms"] = per * 1e3
                        entry["tops"] = self.flops(shape, m) / per / 1e12
                        entry["worker_footprint_mb"] = w.footprint_mb()
                    except Exception as e:  # noqa: BLE001
                        entry["error"] = str(e)[:300]
                    finally:
                        w.close()
                    if "error" not in entry:
                        w2 = Worker(in_shape, out_shape, self.args.units)
                        try:
                            entry["second_load_s"] = w2.load(path)["load_s"]
                        finally:
                            w2.close()
                    res[key] = entry
                    if "error" in entry:
                        self.row(key, f"FAILED: {entry['error']}")
                        continue
                    plan = entry.get("plan") or {}
                    devices = ", ".join(f"{k}x{v}" for k, v in (plan.get("ops") or {}).items()) or plan.get(
                        "error", "?"
                    )
                    fails = sum((entry.get("log") or {}).get("failures", {}).values()) if entry.get("log") else 0
                    self.row(
                        key,
                        f"{entry['ms']:7.1f} ms {entry['tops']:6.2f} TOPS  load {entry['first_load_s']:5.1f} s"
                        f" -> {entry['second_load_s']:4.2f} s  worker {entry['worker_footprint_mb']:6.0f} MB  "
                        f"[{devices}]" + (f"  ANE LOG FAILURES {fails}" if fails else ""),
                    )
                    if plan.get("ane_cores"):
                        self.results["ane_cores"] = plan["ane_cores"]

    def dual(self) -> None:
        print("\n== dual: the same package in two processes at once, and in two threads of one ==")
        res = self.results.setdefault("dual", {})
        for shape, m, variant in self._picks():
            key = f"{shape}-m{m}-{variant}"
            path = self.converter.path(shape, m, variant)
            in_shape, out_shape = self.io_shapes(shape, m, self.args.layout)
            flops = self.flops(shape, m)
            entry = {}
            workers = [Worker(in_shape, out_shape, self.args.units) for _ in range(2)]
            try:
                for w in workers:
                    w.x[...] = np.random.normal(size=in_shape).astype(np.float16) * 0.5
                    w.load(path)
                    w.call("bench", 3, 2, 1)
                t0 = time.perf_counter()
                for w in workers:
                    w.send("bench", 0, self.args.iters, 1)
                infos = [w.recv() for w in workers]
                wall = time.perf_counter() - t0
                entry["two_procs_tops"] = sum(i["calls"] for i in infos) * flops / wall / 1e12
                entry["two_procs_each_ms"] = [i["wall_s"] / i["calls"] * 1e3 for i in infos]
                entry["two_procs_footprint_mb"] = [w.footprint_mb() for w in workers]
                # one process, second instance, two threads
                w = workers[0]
                w.load(path)
                info = w.call("bench", 0, self.args.iters, 2)
                entry["two_threads_tops"] = info["calls"] * flops / info["wall_s"] / 1e12
                info = w.call("bench", 0, self.args.iters, 1)
                entry["one_thread_tops"] = info["calls"] * flops / info["wall_s"] / 1e12
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)[:300]
            finally:
                for w in workers:
                    w.close()
            res[key] = entry
            if "error" in entry:
                self.row(key, f"FAILED: {entry['error']}")
            else:
                self.row(
                    key,
                    f"1 proc {entry['one_thread_tops']:6.2f}  2 procs {entry['two_procs_tops']:6.2f}  "
                    f"2 threads {entry['two_threads_tops']:6.2f} TOPS  "
                    f"(each {entry['two_procs_each_ms'][0]:.1f} / {entry['two_procs_each_ms'][1]:.1f} ms)",
                )

    def concurrent(self) -> None:
        print("\n== concurrent: the ANE bench while the GPU runs the 6144->16384 q8 matmul ==")
        res = self.results.setdefault("concurrent", {})
        alone = GpuLoad.start(8.0, 4126)
        rows = GpuLoad.windows(alone)
        gpu_alone = GpuLoad.mean_within(rows, 0, float("inf"))
        res["gpu_alone_tflops"] = gpu_alone
        self.row("gpu alone (q8 6144->16384)", f"{gpu_alone:6.2f} TFLOPS" if gpu_alone else "no windows")
        for shape, m, variant in self._picks():
            key = f"{shape}-m{m}-{variant}"
            path = self.converter.path(shape, m, variant)
            in_shape, out_shape = self.io_shapes(shape, m, self.args.layout)
            flops = self.flops(shape, m)
            entry = {}
            w = Worker(in_shape, out_shape, self.args.units)
            try:
                w.x[...] = np.random.normal(size=in_shape).astype(np.float16) * 0.5
                w.load(path)
                warm = w.call("bench", 3, 3, 1)
                per = warm["wall_s"] / warm["calls"]
                iters = max(self.args.iters, int(12.0 / per))  # at least ~12 s of overlap
                gpu = GpuLoad.start(per * iters + 12.0, 4126)
                time.sleep(4.0)  # let the GPU loop reach steady state
                start = time.time()
                info = w.call("bench", 0, iters, 1)
                end = time.time()
                rows = GpuLoad.windows(gpu)
                entry["ane_alone_tops"] = flops / per / 1e12
                entry["ane_with_gpu_tops"] = info["calls"] * flops / info["wall_s"] / 1e12
                entry["gpu_with_ane_tflops"] = GpuLoad.mean_within(rows, start, end)
                entry["overlap_s"] = end - start
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)[:300]
            finally:
                w.close()
            res[key] = entry
            if "error" in entry:
                self.row(key, f"FAILED: {entry['error']}")
            else:
                gpu_b = entry["gpu_with_ane_tflops"]
                gpu_txt = f"{gpu_b:6.2f} ({gpu_b / gpu_alone * 100 - 100:+.0f}%)" if gpu_b and gpu_alone else "n/a"
                self.row(
                    key,
                    f"ANE {entry['ane_alone_tops']:6.2f} -> {entry['ane_with_gpu_tops']:6.2f} TOPS   "
                    f"GPU {gpu_alone:6.2f} -> {gpu_txt} TFLOPS   overlap {entry['overlap_s']:.0f} s",
                )

    def handoff(self) -> None:
        print("\n== handoff: one production call through shared memory (x in, y out), mean of --iters ==")
        res = self.results.setdefault("handoff", {})
        for shape, m, variant in self._picks():
            key = f"{shape}-m{m}-{variant}"
            path = self.converter.path(shape, m, variant)
            in_shape, out_shape = self.io_shapes(shape, m, self.args.layout)
            entry = {}
            w = Worker(in_shape, out_shape, self.args.units)
            try:
                w.load(path)
                x = (np.random.normal(size=in_shape) * 0.5).astype(np.float16)
                for _ in range(3):
                    w.step(x)
                infos = [w.step(x)[1] for _ in range(self.args.iters)]
                rt = np.mean([i["round_trip_s"] for i in infos])
                pr = np.mean([i["predict_s"] for i in infos])
                co = np.mean([i["copy_out_s"] for i in infos])
                entry = {
                    "round_trip_ms": rt * 1e3,
                    "predict_ms": pr * 1e3,
                    "copy_out_ms": co * 1e3,
                    "overhead_ms": (rt - pr) * 1e3,
                    "bytes_in_mb": x.nbytes / 1e6,
                    "bytes_out_mb": int(np.prod(out_shape)) * 2 / 1e6,
                }
            except Exception as e:  # noqa: BLE001
                entry["error"] = str(e)[:300]
            finally:
                w.close()
            res[key] = entry
            if "error" in entry:
                self.row(key, f"FAILED: {entry['error']}")
            else:
                self.row(
                    key,
                    f"round trip {entry['round_trip_ms']:7.1f} ms = predict {entry['predict_ms']:7.1f}"
                    f" + overhead {entry['overhead_ms']:5.1f} ms  ({entry['bytes_in_mb']:.0f} MB in, "
                    f"{entry['bytes_out_mb']:.0f} MB out)",
                )

    def numerics(self) -> None:
        print("\n== numerics: the fused MLP on the ANE (fp16) against the bf16 GPU path and an fp32 truth ==")
        mx = self.block0.mx
        res = self.results.setdefault("numerics", {})
        block = self.block0.block()
        xin = self.block0.mlp_input(block, 4126)
        x16 = np.array(xin.astype(mx.float16))[0]
        truth_w = {k: v.astype(np.float32) for k, v in self.weights.items() if k.startswith("mlp")}
        x32 = x16.astype(np.float32)
        h32 = x32 @ truth_w["mlp.gate"].T
        h32 = h32 / (1 + np.exp(-h32)) * (x32 @ truth_w["mlp.up"].T)
        truth = h32 @ truth_w["mlp.down"].T
        res["activation_max_abs"] = {"mlp_input": float(np.abs(x32).max()), "silu_gate_x_up": float(np.abs(h32).max())}
        self.row("max |x| / max |silu(g)*u| (random x)", f"{np.abs(x32).max():.2f} / {np.abs(h32).max():.2f}")
        block.mlp.down_splits = 4
        gpu = np.array(block.mlp(xin).astype(mx.float32))[0]
        res["gpu_bf16_q8_vs_truth"] = Numerics.compare(gpu, truth)
        self._numerics_row("GPU bf16 q8 (down K4) vs truth", res["gpu_bf16_q8_vs_truth"])
        for variant in self.args.variants:
            path = self.converter.path("mlp", 4126, variant)
            if not path.exists():
                continue
            in_shape, out_shape = self.io_shapes("mlp", 4126, self.args.layout)
            w = Worker(in_shape, out_shape, self.args.units)
            try:
                w.load(path)
                xin_l = x16.T.reshape(in_shape) if self.args.layout == "conv" else x16
                y, _ = w.step(xin_l)
                y = y.reshape(-1, FEATURES) if self.args.layout != "conv" else y[0, :, 0, :].T
                res[f"ane_{variant}_vs_truth"] = Numerics.compare(y, truth)
                res[f"ane_{variant}_vs_gpu"] = Numerics.compare(y, gpu)
                self._numerics_row(f"ANE {variant} fp16 vs truth", res[f"ane_{variant}_vs_truth"])
                self._numerics_row(f"ANE {variant} fp16 vs GPU", res[f"ane_{variant}_vs_gpu"])
            except Exception as e:  # noqa: BLE001
                res[f"ane_{variant}"] = {"error": str(e)[:300]}
                self.row(f"ANE {variant}", f"FAILED: {str(e)[:200]}")
            finally:
                w.close()

    def _numerics_row(self, label: str, c: dict) -> None:
        self.row(
            label,
            f"max abs {c['max_abs']:.4f}  mean abs {c['mean_abs']:.5f}  max rel {c['max_rel'] * 100:6.2f}%  "
            f"mean rel {c['mean_rel'] * 100:.3f}%  >1 ULP(bf16) {c['over_1ulp_bf16'] * 100:5.1f}%  "
            f"changed {c['changed'] * 100:5.1f}%",
        )

    def _picks(self) -> list[tuple[str, int, str]]:
        # The shapes that matter for the split, at 1024^2, in the variants that converted.
        return [
            (shape, 4126, variant)
            for shape in self.args.shapes
            if shape in ("gate", "mlp", "mlp57")
            for variant in self.args.variants
            if self.converter.path(shape, 4126, variant).exists()
        ]

    def save(self) -> None:
        if self.args.json:
            self.args.json.parent.mkdir(parents=True, exist_ok=True)
            self.args.json.write_text(json.dumps(self.results, indent=2, default=str))
            print(f"\nwrote {self.args.json}")


class ActStats:
    # A real 1024^2 generation (the M7 command) with the SwiGLU patched to record, per block
    # and step, the max |.| of its input, of silu(gate)*up and of its output: fp16 tops out at
    # 65504, and the ANE takes nothing else.
    @staticmethod
    def run(args: argparse.Namespace) -> None:
        import mlx.core as mx
        from mlx import nn

        from mflux.models.krea2.cli import krea2_generate
        from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU

        stats: list[dict] = []

        def patched(self, x):
            # feed_forward.py's body, with h kept so it can be measured without a second pass
            h = nn.silu(self.gate(x)) * self.up(x)
            base, delta = self._down_adapter if self._down_adapter is not None else (self.down, None)
            if self.down_splits > 1 and isinstance(base, nn.QuantizedLinear):
                y = self._down_in_slices(h, base)
                y = y if delta is None else y + delta(h)
            else:
                y = self.down(h)
            record = {k: mx.abs(v).max() for k, v in (("x", x), ("h", h), ("y", y))}
            mx.async_eval(*record.values())
            stats.append(record)
            return y

        Krea2SwiGLU.__call__ = patched
        out = args.cache / "actstats-1024.png"
        sys.argv = [
            "mflux-generate-krea2", "--model", str(args.model), "--base-model", "krea-2", "--block-streaming",
            "--prompt", "a photograph of a weathered brass diving helmet on a workshop bench, "
            "morning light through a dusty window, shallow depth of field",
            "--seed", "42", "--steps", "4", "--scheduler", "euler", "--guidance", "1.0",
            "--width", "1024", "--height", "1024", "--no-metadata", "--output", str(out),
        ]  # fmt: skip
        t0 = time.perf_counter()
        krea2_generate.main()
        elapsed = time.perf_counter() - t0
        blocks = 28
        rows = []
        for i in range(blocks):
            calls = stats[i::blocks]
            rows.append({k: max(float(c[k]) for c in calls) for k in ("x", "h", "y")} | {"block": i})
        print(f"\n== actstats: max |.| over {len(stats) // blocks} steps, per block (fp16 max 65504) ==")
        print("  block   max|x_in|   max|silu(g)*u|   max|y|")
        for r in rows:
            print(f"  {r['block']:5d}  {r['x']:10.2f}  {r['h']:14.2f}  {r['y']:9.2f}")
        worst = max(rows, key=lambda r: r["h"])
        print(
            f"  worst silu(g)*u: block {worst['block']} at {worst['h']:.1f} = {worst['h'] / 65504 * 100:.2f}% of fp16 max"
        )
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({"elapsed_s": elapsed, "calls": len(stats), "blocks": rows}, indent=2))
            print(f"wrote {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("~/Library/Caches/mflux/16gb-bench/krea2-lowram").expanduser()
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("~/Library/Caches/mflux/16gb-bench/krea2-lowram/ane/probe").expanduser()
    )
    parser.add_argument("--only", default="convert,single,dual,concurrent,handoff,numerics")
    parser.add_argument("--shapes", default="wq,gate,down,mlp,mlp57")
    parser.add_argument("--variants", default="row,g64,fp16")
    parser.add_argument("--m", default="4126,6430", help="token counts (1024^2 = 4126, 1280^2 = 6430)")
    parser.add_argument("--layout", default="linear", choices=("linear", "conv"))
    parser.add_argument("--units", default="all", choices=("all", "ne", "gpu"))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=10.0, help="gpu-load only")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cases", default="base,mlp57,wq,base", help="realco only")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    args.shapes = [s for s in args.shapes.split(",") if s]
    args.variants = [v for v in args.variants.split(",") if v]
    if args.only == "gpu-load":
        GpuLoad.run(args.seconds, int(args.m))
        return
    if args.only == "ane-load":
        shape, variant, m = args.shapes[0], args.variants[0], int(args.m)
        in_shape, _ = AneProbe.io_shapes(shape, m, args.layout)
        path = Converter(args.cache, args.layout).path(shape, m, variant)
        AneLoad.run(path, args.seconds, in_shape, args.units, AneProbe.flops(shape, m))
        return
    args.m = [int(v) for v in args.m.split(",") if v]
    if args.only == "actstats":
        ActStats.run(args)
        return
    if args.only == "realco":
        RealCo.run(args)
        return
    probe = AneProbe(args)
    print(json.dumps(probe.results["host"]))
    steps = {
        "convert": probe.convert_all,
        "single": probe.single,
        "dual": probe.dual,
        "concurrent": probe.concurrent,
        "handoff": probe.handoff,
        "numerics": probe.numerics,
    }
    try:
        for name in args.only.split(","):
            steps[name]()
    finally:
        probe.save()


if __name__ == "__main__":
    main()
