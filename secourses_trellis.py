"""
SECourses TRELLIS Studio - Image -> 3D asset generation.

Forked from trellis-stable-projectorz.  https://www.patreon.com/posts/117470976

Design notes
------------
*  Only light-weight modules are imported at start-up.  Everything expensive
   (torch, flash-attn, spconv, kaolin, nvdiffrast, the TRELLIS pipeline itself)
   is pulled in the first time the user actually starts processing, so the web
   UI is reachable within a couple of seconds.
*  Every long running operation reports live progress to BOTH the console
   (with speed / ETA) and the browser, because the work runs on a background
   thread while the Gradio handler streams status snapshots.
*  The whole UI is built on stock Gradio 6 components - no custom components
   are required any more (``gradio_litmodel3d`` has been dropped in favour of
   the native ``gr.Model3D``).
"""

from __future__ import annotations

import argparse
import glob as globmod
import html as html_mod
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Windows consoles (and any redirected stdout) default to a legacy code page, which makes
# printing a single non-ASCII character blow up a whole generation. Force UTF-8 and never
# raise on an unencodable glyph.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                          # pragma: no cover
        pass

# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SECourses TRELLIS Studio")
parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="Model precision: fp32 (full, default) or fp16 (half, less VRAM).")
parser.add_argument("--attention", choices=["flash_attn", "xformers", "sdpa", "naive"], default=None,
                    help="Attention backend. Default: flash_attn when available, else sdpa.")
parser.add_argument("--xformers", action="store_true",
                    help="Shortcut for --attention xformers.")
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
parser.add_argument("--highvram", action="store_true",
                    help="Keep every model resident in VRAM (fastest, needs ~10 GB).")
parser.add_argument("--preload", action="store_true",
                    help="Start loading the engine in the background right after the UI comes up.")
parser.add_argument("--no-tf32", action="store_true",
                    help="Disable TF32 matmuls (slower on Ampere+, marginally more precise).")
parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab on start.")
parser.add_argument("--port", type=int, default=None, help="Server port (default: 7860 upwards).")
parser.add_argument("--listen", action="store_true", help="Bind to 0.0.0.0 instead of 127.0.0.1.")
cmd_args = parser.parse_args()

# --------------------------------------------------------------------------------------
# Environment - must be set before anything imports `trellis`
# --------------------------------------------------------------------------------------
_ATTENTION_NOTE = ""


def _pick_attention_backend() -> str:
    global _ATTENTION_NOTE
    requested = cmd_args.attention
    if requested is None and cmd_args.xformers:
        requested = "xformers"

    def _have(mod: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            return False

    have_flash = _have("flash_attn")
    have_xformers = _have("xformers")

    if requested == "xformers":
        if have_xformers:
            _ATTENTION_NOTE = "xformers (requested)"
            return "xformers"
        _ATTENTION_NOTE = "xformers requested but not installed -> falling back"
        requested = None
    elif requested in ("flash_attn", "sdpa", "naive"):
        if requested == "flash_attn" and not have_flash:
            _ATTENTION_NOTE = "flash-attn requested but not installed -> falling back"
            requested = None
        else:
            _ATTENTION_NOTE = f"{requested} (requested)"
            return requested

    if have_flash:
        _ATTENTION_NOTE = "flash-attn (auto)"
        return "flash_attn"
    if have_xformers:
        _ATTENTION_NOTE = "xformers (auto, flash-attn unavailable)"
        return "xformers"
    _ATTENTION_NOTE = "sdpa (auto, no flash-attn/xformers)"
    return "sdpa"


ATTENTION_BACKEND = _pick_attention_backend()
# NOTE: trellis validates this against ['xformers', 'flash_attn', 'sdpa', 'naive'] - the
# old value 'flash-attn' (with a dash) silently fell through to the default.
os.environ["ATTN_BACKEND"] = ATTENTION_BACKEND
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# DINOv2 (the image conditioner) reaches for xformers' `memory_efficient_attention`
# whenever xformers can be imported.  That kernel has no fp32 path on Blackwell
# (sm120) and simply raises, so we tell DINOv2 to stay on its own implementation -
# which we then upgrade to torch SDPA in Engine._patch_dinov2_attention().
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np                       # noqa: E402  (already loaded by gradio)
import gradio as gr                      # noqa: E402
from filelock import FileLock            # noqa: E402

from version import code_version          # noqa: E402

APP_VERSION = "V10"

# --------------------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------------------
try:
    import colorama
    colorama.just_fix_windows_console()
    _C = {
        "reset": colorama.Style.RESET_ALL,
        "dim": colorama.Style.DIM,
        "bold": colorama.Style.BRIGHT,
        "cyan": colorama.Fore.CYAN,
        "green": colorama.Fore.GREEN,
        "yellow": colorama.Fore.YELLOW,
        "red": colorama.Fore.RED,
        "magenta": colorama.Fore.MAGENTA,
        "blue": colorama.Fore.BLUE,
    }
except Exception:                                                   # pragma: no cover
    _C = {k: "" for k in ("reset", "dim", "bold", "cyan", "green", "yellow", "red", "magenta", "blue")}


def cprint(msg: str = "", color: str = "", bold: bool = False) -> None:
    prefix = (_C["bold"] if bold else "") + _C.get(color, "")
    sys.stdout.write(f"{prefix}{msg}{_C['reset']}\n")
    sys.stdout.flush()


def banner(text: str, color: str = "cyan") -> None:
    line = "=" * 78
    cprint(line, color)
    cprint(f"  {text}", color, bold=True)
    cprint(line, color)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_rate(rate: Optional[float]) -> str:
    if not rate or rate != rate:
        return "--"
    if rate >= 1:
        return f"{rate:.2f} it/s"
    return f"{1.0 / rate:.2f} s/it"


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
MAX_SEED = 2 ** 31 - 1

OUTPUT_DIR_BASE = os.path.join(APP_DIR, "outputs_trellis")
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR_BASE, "video")
OUTPUT_GLB_DIR = os.path.join(OUTPUT_DIR_BASE, "glb")
OUTPUT_GAUSSIAN_DIR = os.path.join(OUTPUT_DIR_BASE, "gaussian")
OUTPUT_METADATA_DIR = os.path.join(OUTPUT_DIR_BASE, "metadata")
BATCH_OUTPUT_DIR_BASE_DEFAULT = "batch_outputs_trellis"
DEFAULT_BATCH_INPUT = os.path.join(APP_DIR, "batch_input_images")

FAVICON_PATH = os.path.join(APP_DIR, "favicon.svg")

CONFIG_DIR = os.path.join(APP_DIR, "configs_trellis")
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config_trellis.json")
DEFAULT_CONFIG_NAME = "Default"

for _d in (OUTPUT_DIR_BASE, OUTPUT_VIDEO_DIR, OUTPUT_GLB_DIR, OUTPUT_GAUSSIAN_DIR,
           OUTPUT_METADATA_DIR, CONFIG_DIR, DEFAULT_BATCH_INPUT):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def alphanum_key(s: str):
    def try_int(chunk):
        try:
            return int(chunk)
        except ValueError:
            return chunk
    return [try_int(c) for c in re.split("([0-9]+)", s)]


def sorted_glob(pattern: str) -> List[str]:
    files = globmod.glob(pattern)
    files.sort(key=alphanum_key)
    return files


def get_next_output_path_numeric(output_dir: str, extension: str, prefix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", prefix)
    lock = FileLock(os.path.join(output_dir, f".trellis_lock_{safe_prefix}.lock"), timeout=20)
    with lock:
        counter = 1
        while counter <= 99999:
            filename_base = f"{prefix}{counter:04d}"
            final_path = os.path.join(output_dir, f"{filename_base}.{extension}")
            reservation = final_path + ".tmp"
            # A multi-generation run reserves NNNN and writes NNNN_0001, NNNN_0002, ...
            # so a plain NNNN must not be handed out again for that number.
            series_taken = bool(globmod.glob(os.path.join(output_dir, f"{filename_base}_[0-9][0-9][0-9][0-9].*")))
            if not series_taken and not os.path.exists(final_path) and not os.path.exists(reservation):
                try:
                    with open(reservation, "w") as fh:
                        fh.write(f"reserved pid={os.getpid()} t={time.time()}")
                    return final_path, reservation, filename_base
                except IOError as exc:
                    cprint(f"Warning: could not create reservation {reservation}: {exc}", "yellow")
            counter += 1
    raise RuntimeError(f"Could not find a free numeric filename in {output_dir}.")


def remove_temp_reservation_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def open_folder(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            return f"Folder not found and could not be created: {exc}"
    try:
        if platform.system() == "Windows":
            os.startfile(path)                                   # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        cprint(f"[{stamp()}] Opened folder: {path}", "cyan")
        return f"Opened: {path}"
    except Exception as exc:                                     # pragma: no cover
        return f"Error opening folder: {exc}"


def human_bytes(num: Optional[float]) -> str:
    if not num:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


# --------------------------------------------------------------------------------------
# Progress plumbing
# --------------------------------------------------------------------------------------
_job_local = threading.local()
_CURRENT_JOB_LOCK = threading.Lock()
_CURRENT_JOBS: Dict[str, "JobState"] = {}


def _set_thread_job(job: Optional["JobState"]) -> None:
    _job_local.job = job


def _get_thread_job() -> Optional["JobState"]:
    return getattr(_job_local, "job", None)


def register_job(channel: str, job: Optional["JobState"]) -> None:
    with _CURRENT_JOB_LOCK:
        if job is None:
            _CURRENT_JOBS.pop(channel, None)
        else:
            _CURRENT_JOBS[channel] = job


def cancel_channel(channel: str) -> str:
    with _CURRENT_JOB_LOCK:
        job = _CURRENT_JOBS.get(channel)
    if job is None:
        msg = "Nothing is running right now."
        cprint(f"[{stamp()}] {msg}", "yellow")
        gr.Info(msg)
        return msg
    job.request_cancel()
    msg = "Cancel requested - stopping at the next safe point."
    gr.Info(msg)
    return msg


class JobState:
    """Thread-safe progress record shared between a worker thread and the UI poller."""

    MAX_LOG_LINES = 400

    def __init__(self, title: str = "Working"):
        self._lock = threading.RLock()
        self.title = title
        self.log_lines: List[str] = []
        self.stage_name = "Starting"
        self.stage_idx = 0
        self.stage_count = 1
        self.stage_started = time.time()
        self.stage_bars = 1
        self.bar_index = 0
        self.bar_desc = ""
        self.bar_n = 0
        self.bar_total = 0
        self.bar_rate: Optional[float] = None
        self.t_start = time.time()
        self.status = "running"            # running | done | error | cancelled
        self.headline = ""
        self.detail = ""
        self.cancel_event = threading.Event()
        self.eta_override: Optional[float] = None

    # ---- cancellation -------------------------------------------------------------
    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.log("Cancel requested by user - finishing the current step first.", level="warn")

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Cancelled by user.")

    # ---- logging ------------------------------------------------------------------
    def log(self, msg: str, level: str = "info", console: bool = True) -> None:
        line = f"[{stamp()}] {msg}"
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > self.MAX_LOG_LINES:
                del self.log_lines[: -self.MAX_LOG_LINES]
        if console:
            color = {"info": "", "ok": "green", "warn": "yellow", "err": "red", "step": "cyan"}.get(level, "")
            cprint(line, color)

    # ---- stages -------------------------------------------------------------------
    def set_plan(self, stage_count: int) -> None:
        with self._lock:
            self.stage_count = max(1, int(stage_count))

    def stage(self, name: str, index: Optional[int] = None, bars: int = 1) -> None:
        """Start a stage. `bars` is how many tqdm progress bars it is expected to run,
        so the overall percentage keeps climbing instead of restarting at every bar."""
        with self._lock:
            self.stage_idx = self.stage_idx + 1 if index is None else index
            self.stage_name = name
            self.stage_started = time.time()
            self.stage_bars = max(1, int(bars))
            self.bar_index = 0
            self.bar_desc = ""
            self.bar_n = 0
            self.bar_total = 0
            self.bar_rate = None
            idx, total = self.stage_idx, self.stage_count
        self.log(f"> Step {idx}/{total}: {name}", level="step")

    def stage_done(self, note: str = "") -> None:
        with self._lock:
            took = time.time() - self.stage_started
            name = self.stage_name
        suffix = f" ({note})" if note else ""
        self.log(f"  finished '{name}' in {took:.1f}s{suffix}", level="ok")

    def set_headline(self, text: str) -> None:
        with self._lock:
            self.headline = text

    def set_detail(self, text: str) -> None:
        with self._lock:
            self.detail = text

    # ---- tqdm bridge --------------------------------------------------------------
    def bar_start(self, desc: str, total: Optional[int]) -> None:
        with self._lock:
            self.bar_index = min(self.stage_bars, self.bar_index + 1)
            self.bar_desc = desc or ""
            self.bar_n = 0
            self.bar_total = int(total or 0)
            self.bar_rate = None

    def bar_update(self, n: int, total: Optional[int], rate: Optional[float]) -> None:
        with self._lock:
            self.bar_n = int(n or 0)
            if total:
                self.bar_total = int(total)
            if rate:
                self.bar_rate = float(rate)

    def bar_end(self) -> None:
        with self._lock:
            if self.bar_total:
                self.bar_n = self.bar_total

    # ---- snapshot -----------------------------------------------------------------
    def _stage_inner(self) -> float:
        """Progress inside the current stage, 0..1. Assumes the lock is held."""
        bar_frac = (self.bar_n / self.bar_total) if self.bar_total else 0.0
        bar_frac = max(0.0, min(1.0, bar_frac))
        done_bars = max(0, self.bar_index - 1)
        return max(0.0, min(1.0, (done_bars + bar_frac) / float(self.stage_bars)))

    def fraction(self) -> float:
        with self._lock:
            if self.status == "done":
                return 1.0
            base = max(0, self.stage_idx - 1)
            frac = (base + self._stage_inner()) / float(self.stage_count)
        # Never claim 100% while work is still going on.
        return max(0.0, min(0.995, frac))

    def finish(self, headline: str = "") -> None:
        """Mark the job complete so the UI lands on a clean 100%."""
        with self._lock:
            self.status = "done"
            self.stage_idx = self.stage_count
            self.bar_index = self.stage_bars
            self.bar_desc = "complete"
            self.bar_n = self.bar_total
            self.eta_override = 0.0
            self.detail = ""
            if headline:
                self.headline = headline

    def snapshot(self) -> Tuple[str, str]:
        with self._lock:
            status = self.status
            title = self.title
            headline = self.headline
            detail = self.detail
            stage_name = self.stage_name
            stage_idx = self.stage_idx
            stage_count = self.stage_count
            bar_desc = self.bar_desc
            bar_n = self.bar_n
            bar_total = self.bar_total
            bar_rate = self.bar_rate
            elapsed = time.time() - self.t_start
            log_text = "\n".join(self.log_lines[-200:])
            eta_override = self.eta_override

        frac = self.fraction()
        pct = int(round(frac * 100))

        if eta_override is not None:
            eta = eta_override
        elif bar_total and bar_rate:
            eta = (bar_total - bar_n) / bar_rate if bar_rate else None
        elif frac > 0.02:
            eta = elapsed * (1.0 - frac) / frac
        else:
            eta = None

        state_class = {"running": "run", "done": "ok", "error": "err", "cancelled": "warn"}.get(status, "run")
        state_word = {"running": "Running", "done": "Finished", "error": "Failed",
                      "cancelled": "Cancelled"}.get(status, status)

        bits = []
        if bar_total:
            bits.append(f"{bar_n}/{bar_total}")
        if bar_rate:
            bits.append(fmt_rate(bar_rate))
        if status == "running":
            bits.append(f"ETA {fmt_hms(eta)}")
        bits.append(f"elapsed {fmt_hms(elapsed)}")

        esc = html_mod.escape
        sub = esc(bar_desc or stage_name or "")
        head = esc(headline or title)
        # Always emitted (empty when unused) so the card keeps a constant height.
        det = f'<div class="tp-detail">{esc(detail) or "&nbsp;"}</div>' 

        html = f"""
<div class="tp-status tp-{state_class}">
  <div class="tp-row">
    <span class="tp-badge">{esc(state_word)}</span>
    <span class="tp-title">{head}</span>
    <span class="tp-pct">{pct}%</span>
  </div>
  <div class="tp-track"><div class="tp-fill" style="width:{pct}%"></div></div>
  <div class="tp-row tp-sub">
    <span class="tp-stage">Step {stage_idx}/{stage_count} &middot; {sub}</span>
    <span class="tp-meta">{esc(' · '.join(bits))}</span>
  </div>
  {det}
</div>""".strip()
        return html, log_text


class JobCancelled(Exception):
    pass


def idle_status_html(message: str = "Idle - ready when you are.") -> str:
    return f"""
<div class="tp-status tp-idle">
  <div class="tp-row">
    <span class="tp-badge">Idle</span>
    <span class="tp-title">{html_mod.escape(message)}</span>
    <span class="tp-pct">0%</span>
  </div>
  <div class="tp-track"><div class="tp-fill" style="width:0%"></div></div>
  <div class="tp-row tp-sub"><span class="tp-stage">No task running</span><span class="tp-meta">&nbsp;</span></div>
  <div class="tp-detail">&nbsp;</div>
</div>""".strip()


def _run_in_thread(fn, name: str = "trellis-job"):
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn()
        except BaseException as exc:                              # noqa: BLE001
            box["error"] = exc
            box["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=_target, name=name, daemon=True)
    thread.start()
    return thread, box


# --------------------------------------------------------------------------------------
# Engine (lazy heavy imports + pipeline)
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready = False
        self.loading = False
        self.pipeline = None
        self.torch = None
        self.imageio = None
        self.Image = None
        self.render_utils = None
        self.postprocessing_utils = None
        self.Gaussian = None
        self.MeshExtractResult = None
        self.edict = None
        self.load_seconds = 0.0
        self.device_name = "unknown"
        self.total_vram_gb = 0.0
        self.error: Optional[str] = None

    # -- public ---------------------------------------------------------------------
    def ensure(self, job: Optional[JobState] = None):
        if self.ready:
            return self.pipeline
        with self.lock:
            if self.ready:
                return self.pipeline
            self._load(job)
        return self.pipeline

    def status_line(self) -> str:
        if self.ready:
            return f"Engine ready ({self.device_name}, loaded in {self.load_seconds:.1f}s)"
        if self.loading:
            return "Engine is loading..."
        if self.error:
            return f"Engine failed to load: {self.error}"
        return "Engine not loaded yet - it starts on your first action"

    # -- internals ------------------------------------------------------------------
    def _load(self, job: Optional[JobState]):
        def note(msg: str, level: str = "info"):
            if job is not None:
                job.log(msg, level=level)
            else:
                cprint(f"[{stamp()}] {msg}", {"ok": "green", "warn": "yellow", "err": "red"}.get(level, "cyan"))

        self.loading = True
        self.error = None
        t0 = time.time()
        try:
            banner("Loading the TRELLIS engine (first use only)")
            note(f"Attention backend: {ATTENTION_BACKEND} - {_ATTENTION_NOTE}")

            note("Importing PyTorch ...")
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.device_name = torch.cuda.get_device_name(0)
                self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                note(f"CUDA device: {self.device_name} ({self.total_vram_gb:.1f} GB), torch {torch.__version__}")
            else:
                self.device_name = "CPU only"
                note("CUDA is NOT available - TRELLIS needs an NVIDIA GPU.", level="err")

            self._apply_torch_tuning(note)

            note("Importing imaging helpers ...")
            import imageio.v2 as imageio_v2
            from PIL import Image
            self.imageio = imageio_v2
            self.Image = Image

            note("Importing TRELLIS (flash-attn / spconv / kaolin - this is the slow part) ...")
            from trellis.pipelines import TrellisImageTo3DPipeline
            from trellis.representations import Gaussian, MeshExtractResult
            from trellis.utils import render_utils, postprocessing_utils
            from easydict import EasyDict as edict
            self.render_utils = render_utils
            self.postprocessing_utils = postprocessing_utils
            self.Gaussian = Gaussian
            self.MeshExtractResult = MeshExtractResult
            self.edict = edict

            install_tqdm_hook()

            note("Loading model weights from ./models ...")
            pipeline = TrellisImageTo3DPipeline.from_pretrained(os.path.join(APP_DIR, "models"))
            self._patch_dinov2_attention(note)

            if cmd_args.precision == "fp16":
                note("Converting the pipeline to half precision (fp16).")
                pipeline.to(torch.float16)
                cond = pipeline.models.get("image_cond_model")
                if cond is not None and hasattr(cond, "half"):
                    cond.half()

            self._patch_rembg(pipeline, note)

            if cmd_args.highvram:
                self._apply_highvram(pipeline, note)

            self.pipeline = pipeline
            self.ready = True
            self.load_seconds = time.time() - t0
            note(f"Engine ready in {self.load_seconds:.1f}s.", level="ok")
            banner("Engine ready", "green")
        except BaseException as exc:                              # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            note(f"Engine failed to load: {self.error}", level="err")
            traceback.print_exc()
            raise
        finally:
            self.loading = False

    def _apply_torch_tuning(self, note):
        torch = self.torch
        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if not cmd_args.no_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                note("TF32 matmuls enabled (Ampere+ fast path).")
            except Exception as exc:                              # pragma: no cover
                note(f"Could not enable TF32: {exc}", level="warn")
        else:
            note("TF32 disabled by --no-tf32.")

    def _patch_dinov2_attention(self, note):
        """Route DINOv2 through torch SDPA.

        With XFORMERS_DISABLED set, DINOv2 falls back to a hand written
        ``q @ k.T -> softmax -> @ v``.  That is correct but materialises the whole
        (heads x tokens x tokens) matrix.  SDPA computes exactly the same thing with a
        fused kernel, so this is a pure speed/VRAM win with identical output.
        """
        module = sys.modules.get("dinov2.layers.attention")
        if module is None or not hasattr(module, "MemEffAttention"):
            note("DINOv2 attention module not found - leaving it untouched.", level="warn")
            return
        if getattr(module.MemEffAttention, "_secourses_sdpa", False):
            return

        import torch.nn.functional as F

        def sdpa_forward(self, x, attn_bias=None):
            if attn_bias is not None:                             # pragma: no cover
                raise AssertionError("Nested tensors are not supported in this build.")
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(out))

        module.MemEffAttention.forward = sdpa_forward
        module.MemEffAttention._secourses_sdpa = True
        note("DINOv2 attention switched to torch SDPA (fp32-safe on Blackwell).")

    def _patch_rembg(self, pipeline, note):
        """Give rembg the fastest execution provider that is actually installed."""
        try:
            import onnxruntime as ort
            available = list(ort.get_available_providers())
        except Exception:                                         # pragma: no cover
            available = []
        preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                     if p in available] or ["CPUExecutionProvider"]

        import rembg
        original = pipeline.preprocess_image

        def preprocess_with_session(image):
            if getattr(pipeline, "rembg_session", None) is None:
                # Build it lazily and log to whatever job is active *now*, not the one
                # that happened to be running while the engine was being loaded.
                active = _get_thread_job()
                emit = active.log if active is not None else (lambda m, level="info": cprint(f"[{stamp()}] {m}", "cyan"))
                try:
                    pipeline.rembg_session = rembg.new_session("u2net", providers=preferred)
                    emit(f"Background remover session created ({preferred[0]}).")
                except Exception as exc:                          # pragma: no cover
                    emit(f"rembg fell back to CPU ({exc}).", level="warn")
                    pipeline.rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return original(image)

        pipeline.preprocess_image = preprocess_with_session
        note(f"Background remover will use: {preferred[0]}")

    def _apply_highvram(self, pipeline, note):
        torch = self.torch
        note("High-VRAM mode: keeping every model resident on the GPU.")
        original_move = pipeline._move_models

        def keep_on_gpu(names, device, empty_cache):
            if device == "cpu":
                # Skip the offload entirely, and skip the (expensive) cache flush too:
                # in high-VRAM mode there is nothing to reclaim.
                return
            original_move(names, device, empty_cache)

        pipeline._move_models = keep_on_gpu
        pipeline._move_all_models_to_cpu = lambda: None

        for name in pipeline.models:
            try:
                pipeline.models[name].to("cuda")
            except Exception as exc:                              # pragma: no cover
                note(f"Could not move '{name}' to the GPU: {exc}", level="warn")
        torch.cuda.empty_cache()

    # -- misc -----------------------------------------------------------------------
    def vram_report(self) -> str:
        torch = self.torch
        if torch is None or not torch.cuda.is_available():
            return "CUDA not initialised."
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        return (f"VRAM allocated {alloc:.2f} GB | reserved {reserved:.2f} GB | "
                f"peak {peak:.2f} GB / {self.total_vram_gb:.1f} GB")


ENGINE = Engine()


# --------------------------------------------------------------------------------------
# tqdm -> JobState bridge
# --------------------------------------------------------------------------------------
_TQDM_HOOKED = False


def install_tqdm_hook() -> None:
    """Route the tqdm bars inside TRELLIS into the active JobState.

    The console bars keep working exactly as before (tqdm still writes to stderr with
    its own rate/ETA); we only mirror the numbers so the browser can show them too.
    """
    global _TQDM_HOOKED
    if _TQDM_HOOKED:
        return

    from tqdm import tqdm as _BaseTqdm

    class HookedTqdm(_BaseTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            job = _get_thread_job()
            if job is not None:
                job.bar_start(getattr(self, "desc", "") or "", getattr(self, "total", None))

        def _mirror(self):
            job = _get_thread_job()
            if job is None:
                return
            rate = None
            try:
                rate = self.format_dict.get("rate")
            except Exception:                                     # pragma: no cover
                pass
            job.bar_update(getattr(self, "n", 0), getattr(self, "total", None), rate)

        def update(self, n=1):
            result = super().update(n)
            self._mirror()
            return result

        def __iter__(self):
            done = 0
            for obj in super().__iter__():
                yield obj
                done += 1
                job = _get_thread_job()
                if job is not None:
                    rate = None
                    try:
                        elapsed = self.format_dict.get("elapsed") or 0
                        rate = self.format_dict.get("rate") or (done / elapsed if elapsed else None)
                    except Exception:                             # pragma: no cover
                        pass
                    job.bar_update(done, getattr(self, "total", None), rate)

        def close(self):
            job = _get_thread_job()
            if job is not None:
                job.bar_end()
            return super().close()

    import trellis.pipelines.samplers.flow_euler as _flow_euler
    import trellis.pipelines.trellis_image_to_3d as _pipe_mod
    import trellis.utils.render_utils as _render_mod
    import trellis.utils.postprocessing_utils as _post_mod

    for module in (_flow_euler, _pipe_mod, _render_mod, _post_mod):
        module.tqdm = HookedTqdm

    _TQDM_HOOKED = True


# --------------------------------------------------------------------------------------
# Generation core
# --------------------------------------------------------------------------------------
def pack_state(gs, mesh) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict):
    torch = ENGINE.torch
    gs = ENGINE.Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = ENGINE.edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )
    return gs, mesh


def get_seed(randomize_seed: bool, seed) -> int:
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0
    return int(np.random.randint(0, MAX_SEED)) if randomize_seed else seed


def _ensure_pil(image):
    """Accept PIL, numpy or a path and return an RGBA/RGB PIL image."""
    if image is None:
        return None
    Image = ENGINE.Image
    if isinstance(image, str):
        return Image.open(image).convert("RGBA")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def preprocess_for_pipeline(image, already_clean: bool):
    """Run TRELLIS' preprocessing unless the image is already a clean cut-out."""
    pipeline = ENGINE.pipeline
    image = _ensure_pil(image)
    if image is None:
        raise ValueError("No input image was provided.")
    if already_clean:
        return image
    return pipeline.preprocess_image(image)


def _write_video(path: str, frames_color, frames_geo, fps: int, quality: int, job: JobState) -> None:
    """Stream frames straight into the encoder instead of building one giant array."""
    imageio = ENGINE.imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = len(frames_color)
    writer = imageio.get_writer(
        path, fps=int(fps), codec="libx264", quality=int(quality),
        macro_block_size=1, ffmpeg_log_level="error",
        # yuv420p + faststart = plays everywhere, and the browser can start the
        # turntable before the whole file has arrived.
        pixelformat="yuv420p", output_params=["-movflags", "+faststart"],
    )
    try:
        for i in range(total):
            frame = frames_color[i]
            if frames_geo is not None:
                frame = np.concatenate([frame, frames_geo[i]], axis=1)
            writer.append_data(frame)
            # Release as we go: at 1024px / 240 frames the two lists are ~1.5 GB.
            frames_color[i] = None
            if frames_geo is not None:
                frames_geo[i] = None
            if (i + 1) % 30 == 0 or i + 1 == total:
                job.bar_update(i + 1, total, None)
    finally:
        writer.close()


def generate_one(
    job: JobState,
    *,
    image,
    multiimages,
    is_multiimage: bool,
    image_is_clean: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: str,
    video_resolution: int,
    video_num_frames: int,
    video_fps: int,
    video_quality: int,
    include_geometry: bool,
    save_metadata: bool,
    make_video: bool = True,
    output_filename_prefix: Optional[str] = None,
    custom_output_dirs: Optional[dict] = None,
) -> Tuple[dict, Optional[str]]:
    """One image -> 3D run. Returns (state, video_path or None)."""
    torch = ENGINE.torch
    render_utils = ENGINE.render_utils
    pipeline = ENGINE.pipeline
    start_time = time.time()

    video_dir = custom_output_dirs["video"] if custom_output_dirs else OUTPUT_VIDEO_DIR
    metadata_dir = custom_output_dirs["metadata"] if custom_output_dirs else OUTPUT_METADATA_DIR

    job.stage("Preparing the input image")
    if is_multiimage:
        images = [_ensure_pil(item[0] if isinstance(item, (tuple, list)) else item) for item in (multiimages or [])]
        images = [img for img in images if img is not None]
        if not images:
            raise ValueError("Multi-image mode is selected but the gallery is empty.")
        if not image_is_clean:
            images = [pipeline.preprocess_image(img) for img in images]
        job.log(f"  {len(images)} views prepared.")
    else:
        image = preprocess_for_pipeline(image, image_is_clean)
        images = None
    job.stage_done()
    job.raise_if_cancelled()

    sampler_ss = {"steps": int(ss_sampling_steps), "cfg_strength": float(ss_guidance_strength)}
    sampler_slat = {"steps": int(slat_sampling_steps), "cfg_strength": float(slat_guidance_strength)}

    job.stage(f"Sparse structure + latent sampling (seed {seed})", bars=2)
    job.set_detail(f"SS {ss_sampling_steps} steps @ cfg {ss_guidance_strength} · "
                   f"SLat {slat_sampling_steps} steps @ cfg {slat_guidance_strength}")
    if is_multiimage:
        outputs = pipeline.run_multi_image(
            images, seed=int(seed), formats=["gaussian", "mesh"], preprocess_image=False,
            sparse_structure_sampler_params=sampler_ss, slat_sampler_params=sampler_slat,
            mode=multiimage_algo, cancel_event=job.cancel_event,
        )
    else:
        outputs = pipeline.run(
            image, seed=int(seed), formats=["gaussian", "mesh"], preprocess_image=False,
            sparse_structure_sampler_params=sampler_ss, slat_sampler_params=sampler_slat,
            cancel_event=job.cancel_event,
        )
    job.stage_done()
    job.raise_if_cancelled()

    gaussian = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    job.log(f"  Mesh: {mesh.vertices.shape[0]:,} vertices / {mesh.faces.shape[0]:,} triangles | "
            f"Gaussians: {gaussian._xyz.shape[0]:,}")

    if make_video:
        job.stage(f"Rendering the preview video ({video_num_frames} frames @ {video_resolution}px)",
                  bars=2 if include_geometry else 1)
        frames_color = render_utils.render_video(
            gaussian, resolution=int(video_resolution), bg_color=(0, 0, 0),
            num_frames=int(video_num_frames), cancel_event=job.cancel_event,
        )["color"]
        frames_geo = None
        if include_geometry:
            job.raise_if_cancelled()
            frames_geo = render_utils.render_video(
                mesh, resolution=int(video_resolution), bg_color=(0, 0, 0),
                num_frames=int(video_num_frames), cancel_event=job.cancel_event,
            )["normal"]
        job.stage_done()
        job.raise_if_cancelled()
    else:
        job.stage("Skipping the preview video (not requested)")
        job.stage_done()

    job.stage("Encoding the video file" if make_video else "Reserving the output name")
    temp_reservation_path = None
    if output_filename_prefix:
        filename_base = output_filename_prefix
        video_path = os.path.join(video_dir, f"{filename_base}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
    else:
        video_path, temp_reservation_path, filename_base = get_next_output_path_numeric(video_dir, "mp4")

    if make_video:
        job.bar_start("frames", int(video_num_frames))
        _write_video(video_path, frames_color, frames_geo, video_fps, video_quality, job)
        job.stage_done(os.path.basename(video_path))
    else:
        video_path = None
        job.stage_done()
    remove_temp_reservation_file(temp_reservation_path)

    state = pack_state(gaussian, mesh)
    state["filename_base"] = filename_base
    if custom_output_dirs:
        state["custom_output_dirs"] = custom_output_dirs

    generation_duration = time.time() - start_time

    if save_metadata:
        metadata = {
            "app": f"SECourses TRELLIS Studio {APP_VERSION}",
            "code_version": code_version,
            "source_image_provided": image is not None or bool(images),
            "multi_image_mode": bool(is_multiimage),
            "num_multi_images": len(images) if images else 0,
            "seed": int(seed),
            "ss_guidance_strength": float(ss_guidance_strength),
            "ss_sampling_steps": int(ss_sampling_steps),
            "slat_guidance_strength": float(slat_guidance_strength),
            "slat_sampling_steps": int(slat_sampling_steps),
            "multiimage_algo": multiimage_algo if is_multiimage else "N/A",
            "video_written": bool(make_video),
            "video_resolution": int(video_resolution) if make_video else None,
            "video_num_frames": int(video_num_frames) if make_video else None,
            "video_fps": int(video_fps) if make_video else None,
            "video_quality": int(video_quality) if make_video else None,
            "video_includes_geometry_pass": bool(include_geometry) if make_video else False,
            "attention_backend": ATTENTION_BACKEND,
            "precision": cmd_args.precision,
            "mesh_vertices": int(mesh.vertices.shape[0]),
            "mesh_triangles": int(mesh.faces.shape[0]),
            "num_gaussians": int(gaussian._xyz.shape[0]),
            "generation_duration_seconds": round(generation_duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "output_filename_base": filename_base,
        }
        metadata_path = os.path.join(metadata_dir, f"{filename_base}.txt")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4)

    if not cmd_args.highvram:
        torch.cuda.empty_cache()
    job.log(f"  {ENGINE.vram_report()}")
    return state, video_path


def extract_glb_file(job: JobState, state: dict, mesh_simplify: float, texture_size: int,
                     save_metadata: bool) -> str:
    torch = ENGINE.torch
    gs, mesh = unpack_state(state)
    vertex_count = int(mesh.vertices.shape[0])
    face_count = int(mesh.faces.shape[0])

    custom_output_dirs = state.get("custom_output_dirs")
    glb_dir = custom_output_dirs["glb"] if custom_output_dirs else OUTPUT_GLB_DIR
    metadata_dir = custom_output_dirs["metadata"] if custom_output_dirs else OUTPUT_METADATA_DIR

    glb_filename_base = state.get("filename_base") or ""
    if not glb_filename_base:
        _, _, glb_filename_base = get_next_output_path_numeric(glb_dir, "glb")
    glb_path = os.path.join(glb_dir, f"{glb_filename_base}.glb")
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)

    # to_glb runs four tracked tqdm bars: hole-fill rasterise, multiview render,
    # texture-bake UV pass, texture-bake optimisation.
    job.stage("Building the textured GLB (decimate -> hole fill -> UV -> bake)", bars=4)
    job.set_detail(f"input mesh {vertex_count:,} verts / {face_count:,} tris · "
                   f"simplify {mesh_simplify} · texture {texture_size}px")
    glb_data = ENGINE.postprocessing_utils.to_glb(
        gs, mesh, simplify=float(mesh_simplify), texture_size=int(texture_size),
        verbose=True, cancel_event=job.cancel_event,
    )
    job.raise_if_cancelled()
    glb_data.export(glb_path)
    job.stage_done(os.path.basename(glb_path))
    job.log(f"  GLB written: {glb_path} ({human_bytes(os.path.getsize(glb_path))})", level="ok")

    if save_metadata and glb_filename_base:
        metadata_path = os.path.join(metadata_dir, f"{glb_filename_base}.txt")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                metadata.update({
                    "vertex_count": vertex_count,
                    "triangle_count": face_count,
                    "mesh_simplify_factor": float(mesh_simplify),
                    "texture_size": int(texture_size),
                    "glb_file_bytes": os.path.getsize(glb_path),
                })
                with open(metadata_path, "w", encoding="utf-8") as fh:
                    json.dump(metadata, fh, indent=4)
            except (json.JSONDecodeError, IOError) as exc:
                job.log(f"  Could not update metadata: {exc}", level="warn")

    if not cmd_args.highvram:
        torch.cuda.empty_cache()
    return glb_path


def extract_gaussian_file(job: JobState, state: dict) -> str:
    torch = ENGINE.torch
    gs, _ = unpack_state(state)
    custom_output_dirs = state.get("custom_output_dirs")
    gs_dir = custom_output_dirs["gaussian"] if custom_output_dirs else OUTPUT_GAUSSIAN_DIR

    gs_filename_base = state.get("filename_base") or ""
    if not gs_filename_base:
        _, _, gs_filename_base = get_next_output_path_numeric(gs_dir, "ply")
    gs_path = os.path.join(gs_dir, f"{gs_filename_base}.ply")
    os.makedirs(os.path.dirname(gs_path), exist_ok=True)

    job.stage("Saving the Gaussian splat (.ply)")
    gs.save_ply(gs_path)
    job.stage_done(os.path.basename(gs_path))
    job.log(f"  PLY written: {gs_path} ({human_bytes(os.path.getsize(gs_path))})", level="ok")
    if not cmd_args.highvram:
        torch.cuda.empty_cache()
    return gs_path


# --------------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------------
PRESET_KEYS = [
    "seed_val", "randomize_seed_val", "num_generations_val",
    "ss_guidance_strength_val", "ss_sampling_steps_val",
    "slat_guidance_strength_val", "slat_sampling_steps_val", "multiimage_algo_val",
    "mesh_simplify_val", "texture_size_val",
    "video_resolution_val", "video_num_frames_val", "video_fps_val",
    "video_quality_val", "include_geometry_val",
    "save_metadata_val",
    "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
    "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val",
]


def get_default_config_values() -> Dict[str, Any]:
    return {
        "seed_val": 0,
        "randomize_seed_val": True,
        "num_generations_val": 1,
        "ss_guidance_strength_val": 7.5,
        "ss_sampling_steps_val": 12,
        "slat_guidance_strength_val": 3.0,
        "slat_sampling_steps_val": 12,
        "multiimage_algo_val": "stochastic",
        "mesh_simplify_val": 0.9,
        "texture_size_val": 1024,
        "video_resolution_val": 1024,
        "video_num_frames_val": 240,
        "video_fps_val": 60,
        "video_quality_val": 8,
        "include_geometry_val": True,
        "save_metadata_val": True,
        "batch_input_folder_val": DEFAULT_BATCH_INPUT,
        "batch_output_folder_val": BATCH_OUTPUT_DIR_BASE_DEFAULT,
        "batch_skip_existing_val": True,
        "batch_gen_video_cb_val": True,
        "batch_extract_glb_cb_val": True,
        "batch_extract_gaussian_cb_val": True,
    }


def get_config_list() -> List[str]:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    configs = [os.path.splitext(f)[0] for f in os.listdir(CONFIG_DIR)
               if f.endswith(".json") and f != os.path.basename(LAST_CONFIG_FILE)]
    return sorted(configs) if configs else [DEFAULT_CONFIG_NAME]


def _ordered_values(config_data: Dict[str, Any]) -> List[Any]:
    defaults = get_default_config_values()
    return [config_data.get(key, defaults[key]) for key in PRESET_KEYS]


def _remember_last(config_name: str, config_data: Dict[str, Any]) -> None:
    try:
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump({"last_config_name": config_name, "data": config_data}, fh, indent=4)
    except OSError:
        pass


def save_config(config_name: str, *values):
    config_name = (config_name or "").strip()
    if not config_name:
        return "Please type a preset name first.", gr.update(choices=get_config_list())
    if re.search(r"[\\/:*?\"<>|]", config_name):
        return "Preset name contains invalid characters.", gr.update(choices=get_config_list())

    config_data = {key: val for key, val in zip(PRESET_KEYS, values)}
    try:
        with open(os.path.join(CONFIG_DIR, f"{config_name}.json"), "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=4)
        _remember_last(config_name, config_data)
        cprint(f"[{stamp()}] Preset '{config_name}' saved.", "green")
        return (f"Preset '{config_name}' saved.",
                gr.update(choices=get_config_list(), value=config_name))
    except Exception as exc:                                       # noqa: BLE001
        return f"Error saving preset: {exc}", gr.update(choices=get_config_list())


def load_config(config_name_to_load: Optional[str]):
    defaults = get_default_config_values()
    if not config_name_to_load:
        _remember_last(DEFAULT_CONFIG_NAME, defaults)
        return tuple(["Loaded built-in defaults."] + _ordered_values(defaults))

    path = os.path.join(CONFIG_DIR, f"{config_name_to_load}.json")
    if not os.path.exists(path):
        return tuple([f"Preset '{config_name_to_load}' not found - loaded defaults."] + _ordered_values(defaults))

    try:
        with open(path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return tuple([f"Preset '{config_name_to_load}' is unreadable ({exc}) - loaded defaults."]
                     + _ordered_values(defaults))

    _remember_last(config_name_to_load, config_data)
    cprint(f"[{stamp()}] Preset '{config_name_to_load}' loaded.", "cyan")
    return tuple([f"Preset '{config_name_to_load}' loaded."] + _ordered_values(config_data))


def delete_config(config_name: Optional[str]):
    if not config_name:
        return "Select a preset to delete.", gr.update(choices=get_config_list())
    path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    if not os.path.exists(path):
        return f"Preset '{config_name}' does not exist.", gr.update(choices=get_config_list())
    try:
        os.remove(path)
    except OSError as exc:
        return f"Could not delete preset: {exc}", gr.update(choices=get_config_list())
    remaining = get_config_list()
    cprint(f"[{stamp()}] Preset '{config_name}' deleted.", "yellow")
    return (f"Preset '{config_name}' deleted.",
            gr.update(choices=remaining, value=remaining[0] if remaining else None))


def reset_to_defaults():
    defaults = get_default_config_values()
    return tuple(["Controls reset to the built-in defaults."] + _ordered_values(defaults))


def initial_load_config():
    """Restore whichever preset was used last. The dropdown value and the slider values
    always come from the *same* file, so the dropdown's own change handler is a no-op."""
    config_data = get_default_config_values()
    last_config_name = DEFAULT_CONFIG_NAME

    default_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    if not os.path.exists(default_path):
        try:
            with open(default_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh, indent=4)
        except OSError:
            pass

    if os.path.exists(LAST_CONFIG_FILE):
        try:
            with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as fh:
                saved_state = json.load(fh)
            candidate = saved_state.get("last_config_name", DEFAULT_CONFIG_NAME)
            if os.path.exists(os.path.join(CONFIG_DIR, f"{candidate}.json")):
                last_config_name = candidate
        except (OSError, json.JSONDecodeError):
            pass

    choices = get_config_list()
    if last_config_name not in choices:
        last_config_name = choices[0]

    path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                config_data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    cprint(f"[{stamp()}] UI ready - restored preset '{last_config_name}'.", "cyan")
    return tuple([gr.update(choices=choices, value=last_config_name)] + _ordered_values(config_data))


# --------------------------------------------------------------------------------------
# UI handlers
# --------------------------------------------------------------------------------------
# How often the streaming handlers push a status snapshot to the browser. Each push
# re-renders the status HTML, so this trades smoothness against DOM churn.
GEN_POLL = 0.4


def _skips(n: int):
    return tuple(gr.skip() for _ in range(n))


def _stream_status(job: "JobState", thread: threading.Thread, emit, poll: float = GEN_POLL):
    """Yield UI updates while `thread` runs, but only when something actually changed.

    Re-sending an identical snapshot makes Gradio replace the status markup for no
    reason, which reads as flicker in the browser.
    """
    last_html = last_log = None
    while thread.is_alive():
        html, log_text = job.snapshot()
        if html != last_html or log_text != last_log:
            last_html, last_log = html, log_text
            yield emit(html, log_text)
        time.sleep(poll)
    thread.join()


def do_preprocess_single(image):
    """Remove the background so the user can preview exactly what the model will see."""
    if image is None:
        gr.Warning("Upload an image first.")
        return gr.skip(), False, idle_status_html("No image to preprocess.")
    job = JobState("Preprocess")
    job.set_plan(2)
    register_job("generate", job)
    try:
        job.stage("Loading the engine")
        ENGINE.ensure(job)
        job.stage_done()
        job.stage("Removing the background")
        _set_thread_job(job)
        result = ENGINE.pipeline.preprocess_image(_ensure_pil(image))
        job.stage_done()
        job.finish("Image ready")
        html, _ = job.snapshot()
        gr.Info("Background removed - the preview now matches the model input.")
        return result, True, html
    except Exception as exc:                                       # noqa: BLE001
        traceback.print_exc()
        job.status = "error"
        job.log(f"Preprocess failed: {exc}", level="err")
        html, _ = job.snapshot()
        gr.Warning(f"Preprocess failed: {exc}")
        return gr.skip(), False, html
    finally:
        _set_thread_job(None)
        register_job("generate", None)


def do_preprocess_multi(images):
    if not images:
        gr.Warning("Add some images to the gallery first.")
        return gr.skip(), False, idle_status_html("Gallery is empty.")
    job = JobState("Preprocess views")
    job.set_plan(2)
    register_job("generate", job)
    try:
        job.stage("Loading the engine")
        ENGINE.ensure(job)
        job.stage_done()
        job.stage(f"Removing backgrounds ({len(images)} views)")
        job.bar_start("views", len(images))
        _set_thread_job(job)
        out = []
        for idx, item in enumerate(images):
            raw = item[0] if isinstance(item, (tuple, list)) else item
            out.append(ENGINE.pipeline.preprocess_image(_ensure_pil(raw)))
            job.bar_update(idx + 1, len(images), None)
        job.stage_done()
        job.finish("Views ready")
        html, _ = job.snapshot()
        return out, True, html
    except Exception as exc:                                       # noqa: BLE001
        traceback.print_exc()
        job.status = "error"
        job.log(f"Preprocess failed: {exc}", level="err")
        html, _ = job.snapshot()
        gr.Warning(f"Preprocess failed: {exc}")
        return gr.skip(), False, html
    finally:
        _set_thread_job(None)
        register_job("generate", None)


def multi_example_sets() -> Dict[str, List[str]]:
    """Group assets/example_multi_image/<case>_<n>.png into {case: [paths]}."""
    folder = os.path.join(APP_DIR, "assets", "example_multi_image")
    sets: Dict[str, List[str]] = {}
    if not os.path.isdir(folder):
        return sets
    for name in sorted(os.listdir(folder), key=alphanum_key):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        case = name.rsplit("_", 1)[0]
        sets.setdefault(case, []).append(os.path.join(folder, name))
    return {k: v for k, v in sets.items() if len(v) > 1}


MULTI_EXAMPLE_SETS = multi_example_sets()


def load_multi_example(case: Optional[str]):
    """Drop a ready-made multi-view set into the gallery (raw - preprocessing happens
    at generation time, or on demand with the 'Remove backgrounds' button)."""
    paths = MULTI_EXAMPLE_SETS.get(case or "")
    if not paths:
        gr.Warning("Pick an example set first.")
        return gr.skip(), gr.skip()
    cprint(f"[{stamp()}] Loaded multi-view example '{case}' ({len(paths)} views).", "cyan")
    return paths, False


def run_generation(
    do_extract: bool,
    num_generations, image, multiimages, is_multiimage, image_is_clean,
    seed, randomize_seed,
    ss_guidance, ss_steps, slat_guidance, slat_steps, multiimage_algo,
    video_resolution, video_frames, video_fps, video_quality, include_geometry,
    save_metadata, mesh_simplify, texture_size,
):
    """Streaming handler used by both generate buttons."""
    n_out = 9   # state, video, model, dl_glb, dl_gs, btn_glb, btn_gs, status, log

    try:
        num_gens = max(1, int(float(num_generations)))
    except (TypeError, ValueError):
        num_gens = 1

    if not is_multiimage and image is None:
        gr.Warning("Please provide an input image.")
        yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
               idle_status_html("Waiting for an input image."), "No input image.")
        return
    if is_multiimage and not multiimages:
        gr.Warning("Please add at least one view to the gallery.")
        yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
               idle_status_html("Waiting for gallery images."), "Gallery is empty.")
        return

    per_gen_stages = 4 + (2 if do_extract else 0)
    job = JobState("Generation")
    job.set_plan(1 + per_gen_stages * num_gens)
    job.set_headline(f"{num_gens} generation(s)" + (" + extraction" if do_extract else ""))
    register_job("generate", job)

    results: Dict[str, Any] = {"state": None, "video": None, "glb": None, "gs": None, "count": 0}

    def worker():
        _set_thread_job(job)
        job.stage("Preparing the engine")
        ENGINE.ensure(job)
        job.stage_done()

        base_prefix = None
        reservation = None
        if num_gens > 1:
            _, reservation, base_prefix = get_next_output_path_numeric(OUTPUT_VIDEO_DIR, "mp4", prefix="")

        # The click chain already resolved (and displayed) the seed via get_seed(), so use
        # it verbatim for the first run - otherwise the slider would show a different seed
        # than the one that actually produced the asset.
        try:
            current_seed = int(seed)
        except (TypeError, ValueError):
            current_seed = 0
        gen_times: List[float] = []
        try:
            for i in range(num_gens):
                job.raise_if_cancelled()
                banner(f"GENERATION {i + 1}/{num_gens}", "magenta")
                t_gen = time.time()

                if i == 0:
                    seed_for_iter = current_seed
                elif randomize_seed:
                    seed_for_iter = get_seed(True, 0)
                else:
                    seed_for_iter = (current_seed + i) % MAX_SEED

                prefix = f"{base_prefix}_{i + 1:04d}" if base_prefix else None
                job.set_headline(f"Generation {i + 1}/{num_gens} · seed {seed_for_iter}")

                state, video_path = generate_one(
                    job,
                    image=image, multiimages=multiimages, is_multiimage=bool(is_multiimage),
                    image_is_clean=bool(image_is_clean), seed=seed_for_iter,
                    ss_guidance_strength=ss_guidance, ss_sampling_steps=ss_steps,
                    slat_guidance_strength=slat_guidance, slat_sampling_steps=slat_steps,
                    multiimage_algo=multiimage_algo,
                    video_resolution=video_resolution, video_num_frames=video_frames,
                    video_fps=video_fps, video_quality=video_quality,
                    include_geometry=bool(include_geometry),
                    save_metadata=bool(save_metadata),
                    output_filename_prefix=prefix,
                )
                results["state"] = state
                results["video"] = video_path
                results["count"] = i + 1

                if do_extract:
                    job.raise_if_cancelled()
                    results["glb"] = extract_glb_file(job, state, mesh_simplify, texture_size, save_metadata)
                    job.raise_if_cancelled()
                    results["gs"] = extract_gaussian_file(job, state)

                took = time.time() - t_gen
                gen_times.append(took)
                avg = sum(gen_times) / len(gen_times)
                remaining = (num_gens - i - 1) * avg
                job.eta_override = remaining if i + 1 < num_gens else 0.0
                cprint(f"[{stamp()}] Generation {i + 1}/{num_gens} done in {took:.1f}s "
                       f"(avg {avg:.1f}s, ETA for the rest {fmt_hms(remaining)})", "green", bold=True)
                job.log(f"Generation {i + 1}/{num_gens} finished in {took:.1f}s "
                        f"| average {avg:.1f}s | ETA {fmt_hms(remaining)}", level="ok")
        finally:
            remove_temp_reservation_file(reservation)
            _set_thread_job(None)
        return results

    thread, box = _run_in_thread(worker, "trellis-generate")
    yield from _stream_status(job, thread, lambda html, log_text: (*_skips(7), html, log_text))
    register_job("generate", None)

    error = box.get("error")
    if error is not None and not job.cancelled and not isinstance(error, JobCancelled):
        job.status = "error"
        job.log(f"FAILED: {type(error).__name__}: {error}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"Generation failed: {error}")
    elif error is not None or job.cancelled:
        job.status = "cancelled"
        job.log("Task cancelled.", level="warn")
    else:
        total = time.time() - job.t_start
        job.finish(f"Done - {results['count']} generation(s) in {fmt_hms(total)}")
        job.log(f"All done in {fmt_hms(total)}.", level="ok")
        banner(f"ALL DONE - {results['count']} generation(s) in {fmt_hms(total)}", "green")

    html, log_text = job.snapshot()
    state = results["state"]
    model_path = results["glb"] or results["gs"]
    yield (
        state,
        results["video"],
        model_path,
        gr.update(value=results["glb"], interactive=bool(results["glb"])),
        gr.update(value=results["gs"], interactive=bool(results["gs"])),
        gr.update(interactive=bool(state)),
        gr.update(interactive=bool(state)),
        html,
        log_text,
    )


def run_extract_glb(state, mesh_simplify, texture_size, save_metadata):
    if not state:
        gr.Warning("Generate something first.")
        yield gr.skip(), gr.skip(), idle_status_html("Nothing to extract yet."), "No generation in memory."
        return

    job = JobState("GLB extraction")
    job.set_plan(2)
    register_job("generate", job)
    out: Dict[str, Any] = {}

    def worker():
        _set_thread_job(job)
        try:
            job.stage("Preparing the engine")
            ENGINE.ensure(job)
            job.stage_done()
            out["glb"] = extract_glb_file(job, state, mesh_simplify, texture_size, save_metadata)
        finally:
            _set_thread_job(None)

    thread, box = _run_in_thread(worker, "trellis-glb")
    yield from _stream_status(job, thread, lambda html, log_text: (gr.skip(), gr.skip(), html, log_text))
    register_job("generate", None)

    error = box.get("error")
    if error is not None and not job.cancelled and not isinstance(error, JobCancelled):
        job.status = "error"
        job.log(f"FAILED: {error}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"GLB extraction failed: {error}")
    elif error is not None or job.cancelled:
        job.status = "cancelled"
    else:
        job.finish("GLB ready")

    html, log_text = job.snapshot()
    glb = out.get("glb")
    yield (glb if glb else gr.skip(),
           gr.update(value=glb, interactive=bool(glb)),
           html, log_text)


def run_extract_gaussian(state):
    if not state:
        gr.Warning("Generate something first.")
        yield gr.skip(), gr.skip(), idle_status_html("Nothing to extract yet."), "No generation in memory."
        return

    job = JobState("Gaussian extraction")
    job.set_plan(2)
    register_job("generate", job)
    out: Dict[str, Any] = {}

    def worker():
        _set_thread_job(job)
        try:
            job.stage("Preparing the engine")
            ENGINE.ensure(job)
            job.stage_done()
            out["gs"] = extract_gaussian_file(job, state)
        finally:
            _set_thread_job(None)

    thread, box = _run_in_thread(worker, "trellis-gs")
    yield from _stream_status(job, thread, lambda html, log_text: (gr.skip(), gr.skip(), html, log_text))
    register_job("generate", None)

    error = box.get("error")
    if error is not None and not job.cancelled:
        job.status = "error"
        job.log(f"FAILED: {error}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"Gaussian extraction failed: {error}")
    elif error is not None:
        job.status = "cancelled"
    else:
        job.finish("Gaussian splat ready")

    html, log_text = job.snapshot()
    gs = out.get("gs")
    yield (gs if gs else gr.skip(),
           gr.update(value=gs, interactive=bool(gs)),
           html, log_text)


# --------------------------------------------------------------------------------------
# Batch processing
# --------------------------------------------------------------------------------------
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff")


def collect_batch_images(folder: str) -> List[str]:
    files: List[str] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted_glob(os.path.join(folder, pattern)))
    return sorted(set(files), key=alphanum_key)


def run_batch_processing(
    num_generations, batch_input_dir, batch_output_base_name, skip_existing,
    gen_video_cb, extract_glb_cb, extract_gs_cb,
    seed, randomize_seed, ss_guidance, ss_steps, slat_guidance, slat_steps, multiimage_algo,
    mesh_simplify, texture_size, video_resolution, video_frames, video_fps, video_quality,
    include_geometry, save_metadata,
):
    batch_input_dir = (batch_input_dir or "").strip()
    if not batch_input_dir or not os.path.isdir(batch_input_dir):
        gr.Warning("Batch input folder not found.")
        yield idle_status_html("Batch input folder not found."), f"Folder not found: {batch_input_dir}"
        return

    all_files = collect_batch_images(batch_input_dir)
    if not all_files:
        gr.Warning("No images found in the batch input folder.")
        yield idle_status_html("No images found."), f"No images in {batch_input_dir}"
        return

    try:
        num_gens_per_image = max(1, int(float(num_generations)))
    except (TypeError, ValueError):
        num_gens_per_image = 1

    base_name = (batch_output_base_name or BATCH_OUTPUT_DIR_BASE_DEFAULT).strip()
    batch_root = base_name if os.path.isabs(base_name) else os.path.join(APP_DIR, base_name)
    custom_dirs = {
        "video": os.path.join(batch_root, "video"),
        "glb": os.path.join(batch_root, "glb"),
        "gaussian": os.path.join(batch_root, "gaussian"),
        "metadata": os.path.join(batch_root, "metadata"),
    }
    for path in [batch_root, *custom_dirs.values()]:
        os.makedirs(path, exist_ok=True)

    total_iterations = len(all_files) * num_gens_per_image
    # prepare image, sample, render video, write video (+ optional extractions)
    per_item_stages = 4 + (1 if extract_glb_cb else 0) + (1 if extract_gs_cb else 0)

    job = JobState("Batch")
    job.set_plan(1 + per_item_stages * total_iterations)
    job.set_headline(f"{len(all_files)} image(s) × {num_gens_per_image} generation(s)")
    register_job("batch", job)

    # Kept outside worker() so a cancel or a crash still reports what actually ran.
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    durations: List[float] = []

    def worker():
        _set_thread_job(job)
        try:
            job.stage("Preparing the engine")
            ENGINE.ensure(job)
            job.stage_done()
            banner(f"BATCH START - {len(all_files)} images x {num_gens_per_image} gen(s) "
                   f"= {total_iterations} iterations", "magenta")
            job.log(f"Input : {batch_input_dir}")
            job.log(f"Output: {batch_root}")

            for file_index, image_path in enumerate(all_files):
                job.raise_if_cancelled()
                input_basename = os.path.splitext(os.path.basename(image_path))[0]
                seed_for_file = get_seed(bool(randomize_seed), seed)

                for iteration in range(num_gens_per_image):
                    job.raise_if_cancelled()
                    suffix = f"_{iteration + 1:04d}" if num_gens_per_image > 1 else ""
                    output_prefix = input_basename + suffix
                    done_so_far = file_index * num_gens_per_image + iteration
                    job.set_headline(f"Image {file_index + 1}/{len(all_files)} · "
                                     f"gen {iteration + 1}/{num_gens_per_image} · {input_basename}")

                    if skip_existing:
                        targets = []
                        if gen_video_cb:
                            targets.append(os.path.join(custom_dirs["video"], f"{output_prefix}.mp4"))
                        if extract_glb_cb:
                            targets.append(os.path.join(custom_dirs["glb"], f"{output_prefix}.glb"))
                        if extract_gs_cb:
                            targets.append(os.path.join(custom_dirs["gaussian"], f"{output_prefix}.ply"))
                        if targets and all(os.path.exists(t) for t in targets):
                            stats["skipped"] += 1
                            job.stage_idx += per_item_stages
                            job.log(f"Skipping {output_prefix} - all requested outputs already exist.",
                                    level="warn")
                            continue

                    t_item = time.time()
                    try:
                        seed_for_iter = (get_seed(True, 0) if randomize_seed
                                         else (seed_for_file + iteration) % MAX_SEED)
                        raw = ENGINE.Image.open(image_path).convert("RGBA")
                        state, video_path = generate_one(
                            job,
                            image=raw, multiimages=None, is_multiimage=False, image_is_clean=False,
                            seed=seed_for_iter,
                            ss_guidance_strength=ss_guidance, ss_sampling_steps=ss_steps,
                            slat_guidance_strength=slat_guidance, slat_sampling_steps=slat_steps,
                            multiimage_algo=multiimage_algo,
                            video_resolution=video_resolution, video_num_frames=video_frames,
                            video_fps=video_fps, video_quality=video_quality,
                            include_geometry=bool(include_geometry),
                            save_metadata=bool(save_metadata),
                            make_video=bool(gen_video_cb),
                            output_filename_prefix=output_prefix,
                            custom_output_dirs=custom_dirs,
                        )

                        if extract_glb_cb:
                            job.raise_if_cancelled()
                            extract_glb_file(job, state, mesh_simplify, texture_size, save_metadata)
                        if extract_gs_cb:
                            job.raise_if_cancelled()
                            extract_gaussian_file(job, state)

                        stats["processed"] += 1
                        durations.append(time.time() - t_item)
                        avg = sum(durations) / len(durations)
                        remaining_items = total_iterations - (done_so_far + 1)
                        eta = remaining_items * avg
                        job.eta_override = eta
                        cprint(f"[{stamp()}] [{done_so_far + 1}/{total_iterations}] {output_prefix} "
                               f"done in {durations[-1]:.1f}s | avg {avg:.1f}s | ETA {fmt_hms(eta)}",
                               "green", bold=True)
                        job.log(f"[{done_so_far + 1}/{total_iterations}] {output_prefix} finished in "
                                f"{durations[-1]:.1f}s | avg {avg:.1f}s | ETA {fmt_hms(eta)}", level="ok")
                    except JobCancelled:
                        raise
                    except Exception as exc:                       # noqa: BLE001
                        # TRELLIS raises its own CancelledException from deep inside the
                        # samplers/renderers; that is a cancel, not a failure.
                        if job.cancelled:
                            raise JobCancelled(str(exc)) from exc
                        stats["failed"] += 1
                        job.log(f"ERROR on {output_prefix}: {exc}", level="err")
                        traceback.print_exc()
                        job.stage_idx = min(job.stage_count, 1 + (done_so_far + 1) * per_item_stages)
        finally:
            _set_thread_job(None)
        return dict(stats)

    thread, box = _run_in_thread(worker, "trellis-batch")
    yield from _stream_status(job, thread, lambda html, log_text: (html, log_text))
    register_job("batch", None)

    error = box.get("error")
    summary = box.get("result") or stats
    if error is not None and not job.cancelled and not isinstance(error, JobCancelled):
        job.status = "error"
        job.log(f"Batch failed: {error}", level="err")
        cprint(box.get("traceback", ""), "red")
    elif error is not None or job.cancelled:
        job.status = "cancelled"
        job.log("Batch cancelled by user.", level="warn")
    else:
        job.finish()

    took = fmt_hms(time.time() - job.t_start)
    line = (f"Batch finished in {took} | processed {summary.get('processed', 0)} | "
            f"skipped {summary.get('skipped', 0)} | failed {summary.get('failed', 0)}")
    job.set_headline(line)
    job.log(line, level="ok")
    banner(line.upper(), "green")
    html, log_text = job.snapshot()
    yield html, log_text


# --------------------------------------------------------------------------------------
# System tab helpers
# --------------------------------------------------------------------------------------
def system_info_markdown() -> str:
    rows = [
        ("App", f"SECourses TRELLIS Studio {APP_VERSION} (core {code_version})"),
        ("Python", sys.version.split()[0]),
        ("Platform", f"{platform.system()} {platform.release()}"),
        ("Attention backend", f"{ATTENTION_BACKEND} — {_ATTENTION_NOTE}"),
        ("Sparse backend", os.environ.get("SPARSE_BACKEND", "spconv")),
        ("Precision", cmd_args.precision),
        ("High-VRAM mode", "on" if cmd_args.highvram else "off"),
        ("TF32", "off (--no-tf32)" if cmd_args.no_tf32 else "on"),
        ("Engine", ENGINE.status_line()),
    ]
    if ENGINE.torch is not None:
        torch = ENGINE.torch
        rows.append(("Torch", torch.__version__))
        if torch.cuda.is_available():
            rows.append(("GPU", f"{ENGINE.device_name} — {ENGINE.total_vram_gb:.1f} GB"))
            rows.append(("VRAM", ENGINE.vram_report()))
    rows.append(("Outputs folder", OUTPUT_DIR_BASE))
    rows.append(("Presets folder", CONFIG_DIR))

    body = "\n".join(f"| **{k}** | {v} |" for k, v in rows)
    return f"| | |\n|---|---|\n{body}"


def do_load_engine():
    if ENGINE.ready:
        yield "Engine is already loaded.", system_info_markdown()
        return
    job = JobState("Engine")
    job.set_plan(1)

    def worker():
        _set_thread_job(job)
        try:
            job.stage("Loading the engine")
            ENGINE.ensure(job)
            job.stage_done()
        finally:
            _set_thread_job(None)

    thread, box = _run_in_thread(worker, "trellis-engine")
    last = None
    while thread.is_alive():
        _, log_text = job.snapshot()
        if log_text != last:
            last = log_text
            yield log_text, gr.skip()
        time.sleep(GEN_POLL)
    thread.join()
    if box.get("error"):
        yield f"Engine failed to load:\n{box['traceback']}", system_info_markdown()
    else:
        _, log_text = job.snapshot()
        yield log_text, system_info_markdown()


def do_free_vram():
    if ENGINE.torch is None:
        return "Engine has not been loaded yet - nothing to free.", system_info_markdown()
    torch = ENGINE.torch
    before = torch.cuda.memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0
    if ENGINE.pipeline is not None and not cmd_args.highvram:
        try:
            ENGINE.pipeline._move_all_models_to_cpu()
        except Exception:                                          # pragma: no cover
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        after = torch.cuda.memory_reserved() / 1024 ** 3
        msg = f"Released {max(0.0, before - after):.2f} GB — {ENGINE.vram_report()}"
    else:
        msg = "CUDA is not available."
    cprint(f"[{stamp()}] {msg}", "cyan")
    return msg, system_info_markdown()


# --------------------------------------------------------------------------------------
# CSS - modern look + one colour per button, per tab
# --------------------------------------------------------------------------------------
BUTTON_COLORS = {
    "indigo": ("#4f46e5", "#6366f1"),
    "emerald": ("#059669", "#10b981"),
    "amber": ("#d97706", "#f59e0b"),
    "rose": ("#e11d48", "#f43f5e"),
    "cyan": ("#0891b2", "#06b6d4"),
    "violet": ("#7c3aed", "#8b5cf6"),
    "teal": ("#0d9488", "#14b8a6"),
    "orange": ("#ea580c", "#f97316"),
    "pink": ("#db2777", "#ec4899"),
    "slate": ("#475569", "#64748b"),
    "lime": ("#4d7c0f", "#65a30d"),
    "sky": ("#0284c7", "#0ea5e9"),
}

_BTN_CSS = "\n".join(
    f""".gradio-container button.tb-{name}, .gradio-container a.tb-{name} {{
  background: linear-gradient(135deg, {dark} 0%, {light} 100%) !important;
  border: 1px solid {dark} !important;
  color: #ffffff !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.16) !important;
}}
.gradio-container button.tb-{name}:hover, .gradio-container a.tb-{name}:hover {{
  background: linear-gradient(135deg, {light} 0%, {dark} 100%) !important;
  filter: brightness(1.08);
}}
.gradio-container button.tb-{name}:disabled {{ opacity: .45 !important; filter: grayscale(.5); }}"""
    for name, (dark, light) in BUTTON_COLORS.items()
)

CUSTOM_CSS = """
.gradio-container { max-width: 1680px !important; }

/* ---------- header ---------- */
#tp-header {
  border-radius: 16px;
  padding: 18px 24px;
  margin-bottom: 10px;
  background: linear-gradient(120deg, #1e1b4b 0%, #4c1d95 40%, #0e7490 100%);
  color: #f8fafc;
  box-shadow: 0 8px 24px rgba(30, 27, 75, .25);
}
#tp-header h1 { margin: 0 0 4px 0; font-size: 1.6rem; font-weight: 750; letter-spacing: -.02em; color:#fff; }
#tp-header p  { margin: 0; opacity: .88; font-size: .93rem; color:#e2e8f0; }
#tp-header a  { color: #a5f3fc; text-decoration: none; font-weight: 600; }
#tp-header a:hover { text-decoration: underline; }
#tp-header .tp-chips { margin-top: 10px; display:flex; gap:8px; flex-wrap:wrap; }
#tp-header .tp-chip {
  background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  border-radius: 999px; padding: 3px 11px; font-size: .78rem; letter-spacing:.01em;
}

/* ---------- status panel ----------
   The markup is replaced wholesale on every update, so every row here has a fixed
   height and reserved width. Without that the card resizes as the text changes and
   the whole column below it jumps around. */
#tp-gen-status, #tp-batch-status { min-height: 118px; }
#tp-gen-status > *, #tp-batch-status > * { margin: 0 !important; }
.tp-status {
  border-radius: 12px; padding: 12px 14px;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  font-size: .88rem;
  contain: layout style;
}
.tp-status .tp-row { display: flex; align-items: center; gap: 10px; height: 22px; }
.tp-status .tp-badge {
  font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  padding: 2px 9px; border-radius: 999px; color: #fff; background: #64748b;
  white-space: nowrap; min-width: 74px; text-align: center;
}
.tp-run   .tp-badge { background: #4f46e5; animation: tp-pulse 1.4s ease-in-out infinite; }
.tp-ok    .tp-badge { background: #059669; }
.tp-err   .tp-badge { background: #e11d48; }
.tp-warn  .tp-badge { background: #d97706; }
.tp-idle  .tp-badge { background: #64748b; }
@keyframes tp-pulse { 0%,100% { opacity:1 } 50% { opacity:.55 } }
.tp-status .tp-title {
  font-weight: 650; flex: 1 1 auto; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tp-status .tp-pct {
  font-variant-numeric: tabular-nums; font-weight: 700; opacity: .85;
  min-width: 52px; text-align: right; flex: 0 0 auto;
}
.tp-status .tp-track {
  height: 9px; border-radius: 999px; margin: 9px 0 7px 0; overflow: hidden;
  background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
}
.tp-status .tp-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #6366f1 0%, #06b6d4 55%, #10b981 100%);
}
.tp-ok  .tp-fill { background: linear-gradient(90deg,#10b981 0%,#34d399 100%); }
.tp-err .tp-fill { background: linear-gradient(90deg,#e11d48 0%,#fb7185 100%); }
.tp-warn .tp-fill { background: linear-gradient(90deg,#d97706 0%,#fbbf24 100%); }
.tp-status .tp-sub {
  font-size: .78rem; opacity: .78; justify-content: space-between; gap: 14px;
}
.tp-status .tp-sub .tp-stage {
  flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tp-status .tp-sub .tp-meta {
  flex: 0 0 auto; font-variant-numeric: tabular-nums; white-space: nowrap;
}
.tp-status .tp-detail {
  margin-top: 4px; font-size: .76rem; opacity: .68; height: 18px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ---------- log box ---------- */
#tp-gen-log textarea, #tp-batch-log textarea, #tp-engine-log textarea {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
  font-size: 11.5px !important; line-height: 1.45 !important;
}

/* ---------- misc polish ---------- */
.tp-card {
  border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 12px; background: var(--background-fill-secondary);
}
.tp-actions { gap: 8px !important; }
.gradio-container .tabitem { padding-top: 12px; }
footer { display: none !important; }
"""  + _BTN_CSS


# Gradio serves the favicon from /favicon.ico but never emits a <link> for it, so we
# declare it here - that is what puts the SVG in the browser tab.
# The <script> makes the turntable actually autoplay: the clips carry no audio track,
# but Chrome still blocks autoplay unless the element is explicitly muted. Gradio reuses
# one <video> element and only swaps its `src`, so a childList observer never sees the
# new clip - capture-phase media events (which do not bubble) are the reliable hook.
HEAD_HTML = """
<link rel="icon" type="image/svg+xml" href="/favicon.ico">
<script>
(function () {
  // Start every new turntable clip exactly once.
  //   * muted, because Chrome blocks autoplay otherwise even with no audio track;
  //   * keep retrying, because Gradio re-renders the player just after the clip lands
  //     and leaves it paused, so a one-shot play() on a media event loses the race;
  //   * remember the src once playback actually starts, so a deliberate pause by the
  //     user is never overridden.
  function start(video) {
    var src = video.currentSrc || video.src;
    if (!src || video.readyState < 2) return;
    if (video.dataset.tpStartedFor === src) return;
    video.muted = true;
    video.playsInline = true;
    var p = video.play();
    if (p && p.then) {
      p.then(function () { video.dataset.tpStartedFor = src; }).catch(function () {});
    } else {
      video.dataset.tpStartedFor = src;
    }
  }
  function sweep() { document.querySelectorAll("video").forEach(start); }
  ["loadeddata", "canplay", "canplaythrough"].forEach(function (evt) {
    document.addEventListener(evt, function (e) {
      if (e.target && e.target.tagName === "VIDEO") start(e.target);
    }, true);
  });
  setInterval(sweep, 600);
  document.addEventListener("DOMContentLoaded", sweep);
})();
</script>
"""


HEADER_HTML = f"""
<div id="tp-header">
  <h1>SECourses TRELLIS Studio {APP_VERSION}</h1>
  <p>Turn a single image (or a few views) into a textured 3D mesh, a Gaussian splat and a turntable video —
     forked from trellis-stable-projectorz ·
     <a href="https://www.patreon.com/posts/117470976" target="_blank" rel="noopener">Patreon post</a></p>
  <div class="tp-chips">
    <span class="tp-chip">core {code_version}</span>
    <span class="tp-chip">attention: {ATTENTION_BACKEND}</span>
    <span class="tp-chip">precision: {cmd_args.precision}</span>
    <span class="tp-chip">{'high-VRAM' if cmd_args.highvram else 'model offloading'}</span>
  </div>
</div>
"""


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    example_images = sorted(
        [os.path.join(APP_DIR, "assets", "example_image", f)
         for f in os.listdir(os.path.join(APP_DIR, "assets", "example_image"))
         if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))],
        key=alphanum_key,
    ) if os.path.isdir(os.path.join(APP_DIR, "assets", "example_image")) else []

    with gr.Blocks(title=f"TRELLIS Studio {APP_VERSION}", delete_cache=(3600, 3600),
                   fill_width=True) as demo:
        gr.HTML(HEADER_HTML)

        is_multiimage = gr.State(False)
        image_is_clean = gr.State(False)
        output_buf = gr.State()

        with gr.Tabs():
            # ============================== GENERATE =================================
            with gr.Tab("🎨  Generate", id="tab-generate"):
                with gr.Row():
                    # ---------------- left: inputs -------------------------------
                    with gr.Column(scale=5):
                        with gr.Tabs() as input_tabs:
                            with gr.Tab("Single image", id=0) as single_image_tab:
                                image_prompt = gr.Image(label="Input image", image_mode="RGBA",
                                                        type="pil", height=460, sources=["upload", "clipboard"])
                                with gr.Row(elem_classes=["tp-actions"]):
                                    preprocess_btn = gr.Button("✂️  Remove background & preview",
                                                               elem_classes=["tb-sky"], size="sm")
                                    clear_image_btn = gr.Button("🧹  Clear image",
                                                                elem_classes=["tb-slate"], size="sm")
                            with gr.Tab("Multiple images", id=1) as multiimage_tab:
                                multiimage_prompt = gr.Gallery(label="Views of the same object",
                                                               format="png", type="pil", height=460,
                                                               columns=3, interactive=True)
                                with gr.Row(elem_classes=["tp-actions"]):
                                    preprocess_multi_btn = gr.Button("✂️  Remove backgrounds",
                                                                     elem_classes=["tb-sky"], size="sm")
                                    clear_multi_btn = gr.Button("🧹  Clear views",
                                                                elem_classes=["tb-slate"], size="sm")
                                with gr.Row(elem_classes=["tp-actions"]):
                                    multi_example_dropdown = gr.Dropdown(
                                        label="Example view sets", scale=3,
                                        choices=sorted(MULTI_EXAMPLE_SETS.keys()) or None,
                                        value=None)
                                    multi_example_btn = gr.Button("📥  Load example views", scale=2,
                                                                  elem_classes=["tb-pink"], size="sm")
                                gr.Markdown(
                                    "Give 2-4 clean views of the **same** object. "
                                    "*This algorithm is experimental — single-image mode is usually sharper.*"
                                )

                        with gr.Row(elem_classes=["tp-actions"]):
                            generate_btn = gr.Button("🎬  Generate preview video", scale=3,
                                                     elem_classes=["tb-indigo"])
                            generate_and_extract_btn = gr.Button("🚀  Generate + extract everything", scale=3,
                                                                 elem_classes=["tb-emerald"])
                            cancel_btn = gr.Button("⛔  Cancel", scale=1, elem_classes=["tb-rose"])

                        gen_status = gr.HTML(idle_status_html(), elem_id="tp-gen-status")
                        with gr.Accordion("Live log", open=True):
                            gen_log = gr.Textbox(label=None, show_label=False, lines=12, max_lines=20,
                                                 interactive=False, autoscroll=True, elem_id="tp-gen-log")

                        with gr.Accordion("Generation settings", open=True):
                            with gr.Row():
                                seed_slider = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
                            num_generations_slider = gr.Number(
                                label="Number of generations", value=1, minimum=1, step=1, precision=0,
                                info="Repeat the run N times. With a fixed seed the seed increments each run.")
                            gr.Markdown("**Stage 1 — sparse structure**")
                            with gr.Row():
                                ss_guidance_strength_slider = gr.Slider(0.0, 10.0, label="Guidance strength",
                                                                        value=7.5, step=0.1)
                                ss_sampling_steps_slider = gr.Slider(1, 50, label="Sampling steps",
                                                                     value=12, step=1)
                            gr.Markdown("**Stage 2 — structured latent**")
                            with gr.Row():
                                slat_guidance_strength_slider = gr.Slider(0.0, 10.0, label="Guidance strength",
                                                                          value=3.0, step=0.1)
                                slat_sampling_steps_slider = gr.Slider(1, 50, label="Sampling steps",
                                                                       value=12, step=1)
                            multiimage_algo_radio = gr.Radio(["stochastic", "multidiffusion"],
                                                             label="Multi-image algorithm", value="stochastic")

                    # ---------------- right: outputs ------------------------------
                    with gr.Column(scale=5):
                        video_output = gr.Video(label="Turntable preview (colour | geometry)",
                                                autoplay=True, loop=True, height=430)
                        with gr.Row(elem_classes=["tp-actions"]):
                            extract_glb_btn = gr.Button("📦  Extract GLB", interactive=False,
                                                        elem_classes=["tb-amber"])
                            extract_gs_btn = gr.Button("✨  Extract Gaussian splat", interactive=False,
                                                       elem_classes=["tb-cyan"])
                        model_output = gr.Model3D(label="3D preview (GLB / PLY)", height=430,
                                                  clear_color=(0.09, 0.10, 0.13, 1.0),
                                                  camera_position=(90, 75, 3.2),
                                                  zoom_speed=1.0, pan_speed=1.0, interactive=False)
                        with gr.Row(elem_classes=["tp-actions"]):
                            download_glb = gr.DownloadButton(label="⬇️  Download GLB", interactive=False,
                                                             elem_classes=["tb-violet"])
                            download_gs = gr.DownloadButton(label="⬇️  Download PLY", interactive=False,
                                                            elem_classes=["tb-teal"])
                        with gr.Row(elem_classes=["tp-actions"]):
                            open_outputs_btn = gr.Button("📂  Open outputs folder",
                                                         elem_classes=["tb-lime"])
                        gr.Markdown(
                            "💡 **Tips** — Gaussian `.ply` files are large (~50 MB) and take a moment to "
                            "appear in the viewer. GLB extraction is the slowest step: it decimates the mesh, "
                            "removes hidden faces, unwraps UVs and bakes a texture."
                        )

                        with gr.Accordion("Preview video settings", open=True):
                            with gr.Row():
                                video_resolution_slider = gr.Slider(256, 2048, label="Resolution (px)",
                                                                    value=1024, step=64,
                                                                    info="Height of each half of the video.")
                                video_num_frames_slider = gr.Slider(30, 480, label="Frames", value=240, step=10)
                            with gr.Row():
                                video_fps_slider = gr.Slider(10, 120, label="FPS", value=60, step=1)
                                video_quality_slider = gr.Slider(1, 10, label="Encoder quality", value=8, step=1,
                                                                 info="Higher = better looking, bigger file.")
                            include_geometry_checkbox = gr.Checkbox(
                                label="Include the geometry (normals) pass", value=True,
                                info="Off ≈ 2× faster video, colour only.")

                        with gr.Accordion("Extraction & metadata", open=True):
                            mesh_simplify_slider = gr.Slider(
                                0.2, 0.99, label="Mesh simplification factor", value=0.9, step=0.01,
                                info="Lower keeps more triangles (bigger, more detailed GLB).")
                            texture_size_slider = gr.Slider(
                                512, 2048, label="Texture size (px)", value=1024, step=512,
                                info="Resolution of the baked texture.")
                            save_metadata_checkbox = gr.Checkbox(
                                label="Save generation metadata (.txt next to the outputs)", value=True)

                        if example_images:
                            with gr.Accordion("Example images", open=True) as single_example_box:
                                gr.Examples(examples=example_images, inputs=[image_prompt],
                                            label=None, examples_per_page=24)
                        else:
                            single_example_box = gr.Accordion("Example images", open=True, visible=False)

            # ================================ BATCH ==================================
            with gr.Tab("📦  Batch", id="tab-batch"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group():
                            batch_input_folder_textbox = gr.Textbox(
                                label="Input folder", value=DEFAULT_BATCH_INPUT,
                                info="Every png/jpg/jpeg/webp/bmp/tiff in this folder is processed.")
                            batch_output_folder_textbox = gr.Textbox(
                                label="Output folder", value=BATCH_OUTPUT_DIR_BASE_DEFAULT,
                                info="Relative names are created next to the app; absolute paths work too.")
                            batch_skip_existing_checkbox = gr.Checkbox(
                                label="Skip an image when all of its requested outputs already exist",
                                value=True)
                        gr.Markdown("**What should the batch produce?**")
                        with gr.Row():
                            batch_gen_video_checkbox = gr.Checkbox(label="Preview video", value=True)
                            batch_extract_glb_checkbox = gr.Checkbox(label="GLB mesh", value=True)
                            batch_extract_gs_checkbox = gr.Checkbox(label="Gaussian .ply", value=True)
                        gr.Markdown(
                            "Batch runs reuse **all** the sliders from the Generate tab "
                            "(seed, steps, guidance, video and extraction settings)."
                        )
                        with gr.Row(elem_classes=["tp-actions"]):
                            batch_process_button = gr.Button("▶️  Start batch", scale=3,
                                                             elem_classes=["tb-indigo"])
                            batch_cancel_button = gr.Button("⛔  Cancel batch", scale=1,
                                                            elem_classes=["tb-rose"])
                        with gr.Row(elem_classes=["tp-actions"]):
                            open_batch_input_btn = gr.Button("📂  Open input folder",
                                                             elem_classes=["tb-emerald"])
                            open_batch_outputs_btn = gr.Button("📁  Open output folder",
                                                               elem_classes=["tb-amber"])
                            batch_scan_button = gr.Button("🔍  Count images",
                                                          elem_classes=["tb-cyan"])
                    with gr.Column(scale=6):
                        batch_status = gr.HTML(idle_status_html("Batch idle."), elem_id="tp-batch-status")
                        batch_log = gr.Textbox(label="Batch log", lines=26, max_lines=40,
                                               interactive=False, autoscroll=True, elem_id="tp-batch-log")

            # =============================== PRESETS =================================
            with gr.Tab("💾  Presets", id="tab-presets"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### Presets\n"
                            "A preset stores every slider, checkbox and batch path in this app. "
                            "The preset you used last is restored automatically on the next start."
                        )
                        config_status_textbox = gr.Textbox(label="Status", interactive=False, lines=2)
                        config_load_dropdown = gr.Dropdown(label="Preset", choices=get_config_list(),
                                                           value=None, allow_custom_value=False)
                        with gr.Row(elem_classes=["tp-actions"]):
                            config_load_button = gr.Button("📥  Load selected", elem_classes=["tb-emerald"])
                            config_delete_button = gr.Button("🗑️  Delete selected", elem_classes=["tb-rose"])
                        config_save_name_textbox = gr.Textbox(label="Save as", placeholder="my-favourite-settings")
                        with gr.Row(elem_classes=["tp-actions"]):
                            config_save_button = gr.Button("💾  Save preset", elem_classes=["tb-indigo"])
                            config_reset_button = gr.Button("♻️  Reset to defaults", elem_classes=["tb-amber"])
                            config_folder_button = gr.Button("📂  Open presets folder", elem_classes=["tb-cyan"])
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### Quality cheat-sheet\n"
                            "| Goal | What to change |\n|---|---|\n"
                            "| Sharper geometry | Sparse-structure steps 20-25, guidance 7.5 |\n"
                            "| Cleaner texture | Texture size 2048, simplification 0.7-0.8 |\n"
                            "| Faster iteration | 10-12 steps, video 512 px / 120 frames, geometry pass off |\n"
                            "| Lower VRAM | Start without `--highvram`, texture size 1024 |\n"
                            "| Reproducible runs | Uncheck *Randomize seed* and note the seed |\n"
                        )

            # ================================ SYSTEM =================================
            with gr.Tab("🛠️  System", id="tab-system") as system_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Engine")
                        gr.Markdown(
                            "Models are **not** loaded at start-up — the UI comes up immediately and the "
                            "engine is built the first time you press a processing button. "
                            "Use the button below to warm it up ahead of time."
                        )
                        with gr.Row(elem_classes=["tp-actions"]):
                            load_engine_btn = gr.Button("⚡  Load engine now", elem_classes=["tb-indigo"])
                            free_vram_btn = gr.Button("🧹  Free VRAM", elem_classes=["tb-amber"])
                            refresh_info_btn = gr.Button("🔄  Refresh info", elem_classes=["tb-emerald"])
                            open_app_folder_btn = gr.Button("📂  Open app folder", elem_classes=["tb-cyan"])
                        engine_log = gr.Textbox(label="Engine / system log", lines=16, max_lines=30,
                                                interactive=False, autoscroll=True, elem_id="tp-engine-log")
                    with gr.Column(scale=1):
                        gr.Markdown("### Environment")
                        system_info_md = gr.Markdown(system_info_markdown())
                        gr.Markdown(
                            "### Command line flags\n"
                            "```\n"
                            "--highvram      keep every model in VRAM (fastest)\n"
                            "--precision     fp32 (default) | fp16\n"
                            "--attention     flash_attn | xformers | sdpa | naive\n"
                            "--preload       warm the engine up in the background at start\n"
                            "--no-tf32       disable TF32 matmuls\n"
                            "--share         create a public Gradio link\n"
                            "--port N        pick the server port\n"
                            "--listen        bind 0.0.0.0 for LAN access\n"
                            "```"
                        )

        # ------------------------------------------------------------------ wiring
        preset_ui_components = [
            seed_slider, randomize_seed_checkbox, num_generations_slider,
            ss_guidance_strength_slider, ss_sampling_steps_slider,
            slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
            mesh_simplify_slider, texture_size_slider,
            video_resolution_slider, video_num_frames_slider, video_fps_slider,
            video_quality_slider, include_geometry_checkbox,
            save_metadata_checkbox,
            batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
            batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox,
        ]
        assert len(preset_ui_components) == len(PRESET_KEYS), "preset component/key mismatch"

        # Every event in this app opts out of Gradio's built-in spinner:
        #   * on the heavy events it covers the video and the 3D viewer and repaints on
        #     every streamed update (the status card already reports stage/percent/ETA);
        #   * on the light events it is what produced the "infinite loading" sliders.
        #     Gradio paints a spinner + elapsed timer on an event's *output* components
        #     for as long as that event is pending, and a preset (or demo.load) event
        #     that lands while a generation is running has to wait its turn.
        # concurrency_limit=None keeps the light events out of the queue altogether, so
        # loading a preset - or pressing Cancel - is never stuck behind a running job.
        LIGHT = {"show_progress": "hidden", "concurrency_limit": None}

        single_image_tab.select(lambda: (False, gr.update(visible=True)),
                                outputs=[is_multiimage, single_example_box], **LIGHT)
        multiimage_tab.select(lambda: (True, gr.update(visible=False)),
                              outputs=[is_multiimage, single_example_box], **LIGHT)

        image_prompt.upload(lambda: False, outputs=[image_is_clean], **LIGHT)
        image_prompt.clear(lambda: False, outputs=[image_is_clean], **LIGHT)
        multiimage_prompt.upload(lambda: False, outputs=[image_is_clean], **LIGHT)
        preprocess_btn.click(do_preprocess_single, inputs=[image_prompt],
                             outputs=[image_prompt, image_is_clean, gen_status],
                             show_progress="hidden")
        preprocess_multi_btn.click(do_preprocess_multi, inputs=[multiimage_prompt],
                                   outputs=[multiimage_prompt, image_is_clean, gen_status],
                                   show_progress="hidden")
        clear_image_btn.click(lambda: (None, False, idle_status_html()),
                              outputs=[image_prompt, image_is_clean, gen_status], **LIGHT)
        clear_multi_btn.click(lambda: (None, False, idle_status_html()),
                              outputs=[multiimage_prompt, image_is_clean, gen_status], **LIGHT)
        multi_example_btn.click(load_multi_example, inputs=[multi_example_dropdown],
                                outputs=[multiimage_prompt, image_is_clean], **LIGHT)

        gen_inputs = [
            num_generations_slider, image_prompt, multiimage_prompt, is_multiimage, image_is_clean,
            seed_slider, randomize_seed_checkbox,
            ss_guidance_strength_slider, ss_sampling_steps_slider,
            slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
            video_resolution_slider, video_num_frames_slider, video_fps_slider,
            video_quality_slider, include_geometry_checkbox,
            save_metadata_checkbox, mesh_simplify_slider, texture_size_slider,
        ]
        gen_outputs = [output_buf, video_output, model_output, download_glb, download_gs,
                       extract_glb_btn, extract_gs_btn, gen_status, gen_log]

        # Constant State components let both buttons share one generator without
        # wrapping it in a lambda (a lambda cannot contain `yield from`).
        extract_off = gr.State(False)
        extract_on = gr.State(True)

        generate_btn.click(
            get_seed, inputs=[randomize_seed_checkbox, seed_slider], outputs=[seed_slider],
            **LIGHT,
        ).then(
            run_generation, inputs=[extract_off] + gen_inputs, outputs=gen_outputs,
            show_progress="hidden",
        )

        generate_and_extract_btn.click(
            get_seed, inputs=[randomize_seed_checkbox, seed_slider], outputs=[seed_slider],
            **LIGHT,
        ).then(
            run_generation, inputs=[extract_on] + gen_inputs, outputs=gen_outputs,
            show_progress="hidden",
        )

        cancel_btn.click(lambda: cancel_channel("generate"), outputs=None, **LIGHT)

        extract_glb_btn.click(
            run_extract_glb,
            inputs=[output_buf, mesh_simplify_slider, texture_size_slider, save_metadata_checkbox],
            outputs=[model_output, download_glb, gen_status, gen_log],
            show_progress="hidden",
        )
        extract_gs_btn.click(
            run_extract_gaussian, inputs=[output_buf],
            outputs=[model_output, download_gs, gen_status, gen_log],
            show_progress="hidden",
        )

        open_outputs_btn.click(lambda: open_folder(OUTPUT_DIR_BASE), outputs=None, **LIGHT)

        # -- batch
        batch_process_button.click(
            run_batch_processing,
            inputs=[
                num_generations_slider, batch_input_folder_textbox, batch_output_folder_textbox,
                batch_skip_existing_checkbox, batch_gen_video_checkbox, batch_extract_glb_checkbox,
                batch_extract_gs_checkbox,
                seed_slider, randomize_seed_checkbox,
                ss_guidance_strength_slider, ss_sampling_steps_slider,
                slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
                mesh_simplify_slider, texture_size_slider,
                video_resolution_slider, video_num_frames_slider, video_fps_slider,
                video_quality_slider, include_geometry_checkbox, save_metadata_checkbox,
            ],
            outputs=[batch_status, batch_log],
            show_progress="hidden",
        )
        batch_cancel_button.click(lambda: cancel_channel("batch"), outputs=None, **LIGHT)
        open_batch_input_btn.click(lambda p: open_folder(p or DEFAULT_BATCH_INPUT),
                                   inputs=[batch_input_folder_textbox], outputs=None, **LIGHT)
        open_batch_outputs_btn.click(
            lambda name: open_folder(name if os.path.isabs(name or "")
                                     else os.path.join(APP_DIR, name or BATCH_OUTPUT_DIR_BASE_DEFAULT)),
            inputs=[batch_output_folder_textbox], outputs=None, **LIGHT)

        def scan_batch_folder(folder):
            folder = (folder or "").strip()
            if not folder or not os.path.isdir(folder):
                return idle_status_html("Folder not found."), f"Folder not found: {folder}"
            files = collect_batch_images(folder)
            listing = "\n".join(f"  {i + 1:>4}. {os.path.basename(f)}" for i, f in enumerate(files[:200]))
            more = f"\n  ... and {len(files) - 200} more" if len(files) > 200 else ""
            cprint(f"[{stamp()}] Batch scan: {len(files)} image(s) in {folder}", "cyan")
            return (idle_status_html(f"{len(files)} image(s) ready in {os.path.basename(folder) or folder}"),
                    f"Found {len(files)} image(s) in {folder}:\n{listing}{more}")

        batch_scan_button.click(scan_batch_folder, inputs=[batch_input_folder_textbox],
                                outputs=[batch_status, batch_log], **LIGHT)

        # -- presets
        config_save_button.click(save_config,
                                 inputs=[config_save_name_textbox] + preset_ui_components,
                                 outputs=[config_status_textbox, config_load_dropdown], **LIGHT)
        config_load_button.click(load_config, inputs=[config_load_dropdown],
                                 outputs=[config_status_textbox] + preset_ui_components, **LIGHT)
        config_load_dropdown.change(load_config, inputs=[config_load_dropdown],
                                    outputs=[config_status_textbox] + preset_ui_components, **LIGHT)
        config_delete_button.click(delete_config, inputs=[config_load_dropdown],
                                   outputs=[config_status_textbox, config_load_dropdown], **LIGHT)
        config_reset_button.click(reset_to_defaults, inputs=None,
                                  outputs=[config_status_textbox] + preset_ui_components, **LIGHT)
        config_folder_button.click(lambda: open_folder(CONFIG_DIR), outputs=None, **LIGHT)

        # -- system
        load_engine_btn.click(do_load_engine, outputs=[engine_log, system_info_md],
                              show_progress="hidden")
        free_vram_btn.click(do_free_vram, outputs=[engine_log, system_info_md], **LIGHT)
        refresh_info_btn.click(lambda: (ENGINE.status_line(), system_info_markdown()),
                               outputs=[engine_log, system_info_md], **LIGHT)
        # The table is rendered once while the UI is being built - i.e. before the engine
        # exists - so refresh it whenever the tab is opened, otherwise it keeps claiming
        # the engine is not loaded long after it is.
        system_tab.select(system_info_markdown, outputs=[system_info_md], **LIGHT)
        open_app_folder_btn.click(lambda: open_folder(APP_DIR), outputs=None, **LIGHT)

        demo.load(initial_load_config, inputs=None,
                  outputs=[config_load_dropdown] + preset_ui_components, **LIGHT)

    return demo


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def _background_preload():
    try:
        ENGINE.ensure(None)
    except Exception:                                              # pragma: no cover
        cprint("Background preload failed - the engine will retry on your first action.", "yellow")


def main() -> None:
    banner(f"SECourses TRELLIS Studio {APP_VERSION}")
    cprint(f"  Attention backend : {ATTENTION_BACKEND} ({_ATTENTION_NOTE})", "cyan")
    cprint(f"  Precision         : {cmd_args.precision}", "cyan")
    cprint(f"  High-VRAM mode    : {'on' if cmd_args.highvram else 'off (models offload to RAM)'}", "cyan")
    cprint(f"  Outputs           : {OUTPUT_DIR_BASE}", "cyan")
    cprint("  Heavy libraries load on your first action, so the UI is up in a moment.", "dim")
    cprint("")

    demo = build_ui()

    if cmd_args.preload:
        cprint("--preload given: warming the engine up in the background.", "yellow")
        threading.Thread(target=_background_preload, name="trellis-preload", daemon=True).start()

    demo.launch(
        inbrowser=not cmd_args.no_browser,
        share=cmd_args.share,
        server_name="0.0.0.0" if cmd_args.listen else None,
        server_port=cmd_args.port,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate"),
        css=CUSTOM_CSS,
        favicon_path=FAVICON_PATH if os.path.isfile(FAVICON_PATH) else None,
        # Gradio serves the favicon from /favicon.ico but does not emit a <link>, so
        # declare it explicitly - that is what makes the SVG show up in the tab.
        head=HEAD_HTML,
        show_error=True,
        footer_links=["settings"],
    )


if __name__ == "__main__":
    main()
