import argparse
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# A browser front end for the low-memory Krea 2 pipeline: type a prompt, press generate,
# watch the steps tick, see the image. It runs on whichever machine holds the weights -
# normally the M6 mac mini, reached from the MacBook over the Thunderbolt bridge - so the
# picture is fetched over HTTP and nothing has to be copied by hand.
#
# Every generation is a fresh `mflux-generate-krea2` process. That is deliberate: the memory
# discipline this fork is built on (docs/16gb/) was measured on exactly that command, and a
# long-lived process holding MLX buffers between runs would invalidate it. The cost is the
# ~8 s of interpreter and text-encoder startup on a ~42 s image.


@dataclass
class StudioConfig:
    repo: Path
    model: Path
    out_dir: Path
    host: str
    port: int
    steps: int = 4
    scheduler: str = "euler"
    guidance: float = 1.0
    width: int = 1024
    height: int = 1024

    CACHE = Path.home() / "Library/Caches/mflux/16gb-bench"

    @staticmethod
    def resolve(args: argparse.Namespace) -> "StudioConfig":
        model = Path(args.model).expanduser() if args.model else StudioConfig.CACHE / "krea2-lowram"
        if not model.is_dir():
            raise SystemExit(
                f"No low-memory snapshot at {model}.\n"
                "Build one first (see docs/16gb/README.md, 一回だけの前処理), or name one with --model."
            )
        return StudioConfig(
            repo=Path(__file__).resolve().parents[2],
            model=model,
            out_dir=Path(args.out_dir).expanduser(),
            host=args.host,
            port=args.port,
        )


@dataclass
class Job:
    id: str
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float
    scheduler: str
    loras: list = field(default_factory=list)
    state: str = "queued"
    stage: str = "queued"
    step: int = 0
    rate: float | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    image: str | None = None
    error: str | None = None
    log: list = field(default_factory=list)

    def public(self) -> dict:
        row = asdict(self)
        row["log"] = self.log[-12:]
        row["elapsed"] = round((self.finished or time.time()) - self.started, 1) if self.started else None
        return row


class JobStore:
    # One queue, one worker: the GPU is a single resource and two generations at once would
    # only make both swap.
    HISTORY = 200

    def __init__(self):
        self.jobs: list[Job] = []
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.running: Job | None = None
        self.process: subprocess.Popen | None = None
        self.revision = 0

    def submit(self, job: Job) -> None:
        with self.wake:
            self.jobs.append(job)
            del self.jobs[: max(0, len(self.jobs) - JobStore.HISTORY)]
            self.revision += 1
            self.wake.notify()

    def claim(self) -> Job | None:
        with self.wake:
            while True:
                pending = next((j for j in self.jobs if j.state == "queued"), None)
                if pending is not None:
                    pending.state = "running"
                    pending.stage = "loading"
                    pending.started = time.time()
                    self.running = pending
                    self.revision += 1
                    return pending
                self.wake.wait()

    def touch(self) -> None:
        with self.lock:
            self.revision += 1

    def finish(self) -> None:
        with self.lock:
            self.running = None
            self.process = None
            self.revision += 1

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            job = next((j for j in self.jobs if j.id == job_id), None)
            if job is None:
                return False
            if job.state == "queued":
                job.state, job.stage = "cancelled", "cancelled"
                job.finished = time.time()
                self.revision += 1
                return True
            if job.state == "running" and self.process is not None:
                # mflux turns SIGINT into StopImageGenerationException and exits cleanly.
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
                except (ProcessLookupError, PermissionError):
                    return False
                job.stage = "cancelling"
                self.revision += 1
                return True
            return False

    def snapshot(self) -> dict:
        with self.lock:
            return {"revision": self.revision, "jobs": [j.public() for j in reversed(self.jobs[-40:])]}


class Gallery:
    # The studio writes its own sidecar next to each image rather than reading mflux's, so
    # "load these settings" restores exactly what was asked for, LoRAs included.
    SUFFIX = ".studio.json"

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def record(self, job: Job, name: str) -> None:
        settings = {
            "prompt": job.prompt,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "seed": job.seed,
            "guidance": job.guidance,
            "scheduler": job.scheduler,
            "loras": job.loras,
            "elapsed_s": round((job.finished or time.time()) - (job.started or time.time()), 1),
            "rate_s_per_step": job.rate,
            "host": socket.gethostname(),
            "created": job.created,
        }
        (self.out_dir / (Path(name).stem + Gallery.SUFFIX)).write_text(json.dumps(settings, indent=2))

    def listing(self, limit: int = 60) -> list[dict]:
        rows = []
        for image in sorted(self.out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            sidecar = self.out_dir / (image.stem + Gallery.SUFFIX)
            meta = {}
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text())
                except json.JSONDecodeError:
                    meta = {}
            rows.append({"name": image.name, "mtime": image.stat().st_mtime, "settings": meta})
        return rows

    def read(self, name: str) -> bytes | None:
        path = (self.out_dir / name).resolve()
        if path.parent != self.out_dir.resolve() or not path.is_file() or path.suffix != ".png":
            return None
        return path.read_bytes()

    def delete(self, name: str) -> bool:
        path = (self.out_dir / name).resolve()
        if path.parent != self.out_dir.resolve() or not path.is_file() or path.suffix != ".png":
            return False
        path.unlink()
        (self.out_dir / (path.stem + Gallery.SUFFIX)).unlink(missing_ok=True)
        return True


class Runner(threading.Thread):
    PROGRESS = re.compile(r"(\d+)/(\d+)\s*\[")
    RATE = re.compile(r"([\d.]+)s/it")

    def __init__(self, config: StudioConfig, store: JobStore, gallery: Gallery):
        super().__init__(daemon=True)
        self.config = config
        self.store = store
        self.gallery = gallery

    def run(self) -> None:
        while True:
            job = self.store.claim()
            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001 - the worker must outlive any single bad job
                job.state, job.stage, job.error = "error", "error", f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                self.store.finish()

    def _execute(self, job: Job) -> None:
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{job.seed}.png"
        target = self.config.out_dir / name
        process = subprocess.Popen(
            self._argv(job, target),
            cwd=self.config.repo,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self.store.lock:
            self.store.process = process
        self._pump(job, process)
        code = process.wait()

        if job.stage == "cancelling":
            job.state, job.stage = "cancelled", "cancelled"
        elif code != 0:
            job.state, job.stage = "error", "error"
            job.error = "\n".join(job.log[-8:]) or f"exit code {code}"
        elif not target.is_file():
            job.state, job.stage = "error", "error"
            job.error = "The run finished but wrote no image.\n" + "\n".join(job.log[-6:])
        else:
            job.state, job.stage, job.image = "done", "done", name
            job.finished = time.time()
            self.gallery.record(job, name)
        self.store.touch()

    def _argv(self, job: Job, target: Path) -> list[str]:
        argv = [
            "uv", "run", "mflux-generate-krea2",
            "--model", str(self.config.model),
            "--base-model", "krea-2",
            "--block-streaming",
            "--prompt", job.prompt,
            "--seed", str(job.seed),
            "--steps", str(job.steps),
            "--scheduler", job.scheduler,
            "--guidance", str(job.guidance),
            "--width", str(job.width),
            "--height", str(job.height),
            "--output", str(target),
        ]  # fmt: skip
        for lora in job.loras:
            argv += ["--lora", str(lora["path"]), str(lora.get("scale", 1.0))]
        return argv

    def _pump(self, job: Job, process: subprocess.Popen) -> None:
        # tqdm redraws with \r, so the stream is split on both terminators. read1 hands back
        # whatever has arrived instead of waiting for a full buffer, or a 4-step run would
        # show no progress at all until its last redraw pushed the count over the threshold.
        buffer = b""
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            buffer += chunk
            parts = re.split(rb"[\r\n]", buffer)
            buffer = parts.pop()
            for part in parts:
                self._consume(job, part.decode("utf-8", "replace").strip())
        if buffer:
            self._consume(job, buffer.decode("utf-8", "replace").strip())

    def _consume(self, job: Job, line: str) -> None:
        if not line:
            return
        progress = Runner.PROGRESS.search(line)
        if progress:
            done, total = int(progress.group(1)), int(progress.group(2))
            job.step, job.steps = done, total
            if job.stage != "cancelling":
                job.stage = "decoding" if done >= total else "sampling"
            rate = Runner.RATE.search(line)
            if rate:
                job.rate = float(rate.group(1))
        else:
            # Everything before the first progress line is one opaque stage: mflux prints no
            # marker between loading the encoder and finishing the prompt. The last line is
            # shown as-is instead of being guessed at.
            job.log.append(line)
            del job.log[: max(0, len(job.log) - 60)]
        self.store.touch()


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "mflux-studio"
    config: StudioConfig
    store: JobStore
    gallery: Gallery

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            page = (Path(__file__).resolve().parent / "index.html").read_bytes()
            return self._send(200, "text/html; charset=utf-8", page)
        if route == "/api/state":
            state = self.store.snapshot()
            state["gallery"] = self.gallery.listing()
            state["config"] = {
                "host": socket.gethostname(),
                "model": str(self.config.model),
                "out_dir": str(self.config.out_dir),
                "steps": self.config.steps,
                "scheduler": self.config.scheduler,
                "guidance": self.config.guidance,
                "width": self.config.width,
                "height": self.config.height,
            }
            return self._json(200, state)
        if route.startswith("/images/"):
            blob = self.gallery.read(unquote(route[len("/images/") :]))
            if blob is None:
                return self._json(404, {"error": "no such image"})
            return self._send(200, "image/png", blob)
        return self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "body is not JSON"})
        if route == "/api/generate":
            return self._generate(body)
        if route == "/api/cancel":
            return self._json(200, {"cancelled": self.store.cancel(body.get("id", ""))})
        if route == "/api/delete":
            return self._json(200, {"deleted": self.gallery.delete(body.get("name", ""))})
        return self._json(404, {"error": "no such route"})

    def log_message(self, fmt: str, *args) -> None:
        pass  # the terminal belongs to the generation logs

    def _generate(self, body: dict) -> None:
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return self._json(400, {"error": "プロンプトが空です"})
        count = max(1, min(int(body.get("count", 1)), 16))
        seed = body.get("seed")
        loras = [
            {"path": lora["path"], "scale": float(lora.get("scale", 1.0))}
            for lora in body.get("loras", [])
            if str(lora.get("path", "")).strip()
        ]
        ids = []
        for index in range(count):
            job = Job(
                id=uuid.uuid4().hex[:12],
                prompt=prompt,
                width=int(body.get("width", self.config.width)),
                height=int(body.get("height", self.config.height)),
                steps=int(body.get("steps", self.config.steps)),
                seed=int(seed) + index if seed not in (None, "") else random.randint(0, 2**31 - 1),
                guidance=float(body.get("guidance", self.config.guidance)),
                scheduler=str(body.get("scheduler", self.config.scheduler)),
                loras=loras,
            )
            self.store.submit(job)
            ids.append(job.id)
        return self._json(200, {"ids": ids})

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send(self, code: int, content_type: str, blob: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(blob)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-poll


def main() -> int:
    parser = argparse.ArgumentParser(description="A browser front end for low-memory Krea 2 generation.")
    parser.add_argument("--model", default=None, help="low-memory snapshot (default: the 16gb-bench one)")
    parser.add_argument("--out-dir", default="~/Pictures/mflux-studio", help="where generated images are kept")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind; name the machine's IP to reach it from another Mac")  # fmt: skip
    parser.add_argument("--port", type=int, default=8765)
    config = StudioConfig.resolve(parser.parse_args())

    store = JobStore()
    gallery = Gallery(config.out_dir)
    Runner(config, store, gallery).start()

    StudioHandler.config, StudioHandler.store, StudioHandler.gallery = config, store, gallery
    server = ThreadingHTTPServer((config.host, config.port), StudioHandler)
    shown = config.host if config.host not in ("0.0.0.0", "") else socket.gethostbyname(socket.gethostname())
    print(f"🎨 mflux studio on {socket.gethostname()}  →  http://{shown}:{config.port}")
    print(f"   model  : {config.model}")
    print(f"   images : {config.out_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
