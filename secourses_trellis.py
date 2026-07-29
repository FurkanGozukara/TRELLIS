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
parser.add_argument("--worker", metavar="JOBFILE", default=None,
                    help=argparse.SUPPRESS)   # internal: run one isolated job, then exit
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

# Linux only, and a no-op everywhere else.  torch's Linux wheels do not bundle the CUDA
# runtime the way the Windows ones bundle it in torch\lib, so cumm/spconv can die with
# "libnvrtc-builtins.so.<major>.<minor>: cannot open shared object file".  Pre-loading
# the libraries with RTLD_GLOBAL here fixes it for every extension imported later.
# LD_LIBRARY_PATH is not an option: the loader reads it once, before Python starts.
try:
    import cuda_linux_deps                # noqa: E402  (APP_DIR is on sys.path above)
except ImportError:                       # helper not shipped - carry on regardless
    cuda_linux_deps = None
else:
    try:
        cuda_linux_deps.preload()
    except OSError:                       # never let library probing stop the app
        pass

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
# Two preset folders. The app owns everything in `presets_builtin` and rewrites it on
# every start, so a user can never end up with a corrupted or stale VRAM preset; their
# own presets live next door in `presets_user` and are never touched by the app.
PRESET_DIR_BUILTIN = os.path.join(CONFIG_DIR, "presets_builtin")
PRESET_DIR_USER = os.path.join(CONFIG_DIR, "presets_user")
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config_trellis.json")
DEFAULT_CONFIG_NAME = "Default"

JOB_TMP_DIR = os.path.join(APP_DIR, "tmp", "jobs")

for _d in (OUTPUT_DIR_BASE, OUTPUT_VIDEO_DIR, OUTPUT_GLB_DIR, OUTPUT_GAUSSIAN_DIR,
           OUTPUT_METADATA_DIR, CONFIG_DIR, PRESET_DIR_BUILTIN, PRESET_DIR_USER,
           DEFAULT_BATCH_INPUT, JOB_TMP_DIR):
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


# --------------------------------------------------------------------------------------
# GPU probe (no torch - this runs before the heavy imports so the UI still starts fast)
# --------------------------------------------------------------------------------------
def _visible_device_index() -> int:
    """Which physical GPU torch will call device 0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 0
    first = visible.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:              # a UUID - we cannot map it, assume the first card
        return 0


def probe_gpu() -> Tuple[Optional[str], Optional[float]]:
    """Return (name, total VRAM in GiB) for the GPU the app will use, or (None, None).

    nvidia-smi reports MiB as the card advertises it, which is what a user compares
    against when they say "I have a 12 GB card" - so this is the right number to bucket
    on, not torch's slightly smaller `total_memory`.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    idx = _visible_device_index()
    if idx >= len(rows):
        idx = 0
    if not rows:
        return None, None
    try:
        name, mib = rows[idx].rsplit(",", 1)
        return name.strip(), int(mib.strip()) / 1024.0
    except (ValueError, IndexError):
        return None, None


GPU_NAME, GPU_VRAM_GB = probe_gpu()


# --------------------------------------------------------------------------------------
# VRAM presets
# --------------------------------------------------------------------------------------
VRAM_TIERS = (6, 8, 10, 12, 16, 24, 32)
# A card that advertises 12 GB reports 12288 MiB but hands torch a little less, and a
# 32 GB card can report 31.8 GB. Round up by this much before bucketing so those land
# on the tier the user would name themselves.
VRAM_TIER_MARGIN_GB = 0.5


def vram_preset_name(tier_gb: int) -> str:
    return f"VRAM {tier_gb} GB"


def pick_vram_tier(total_gb: Optional[float]) -> int:
    """Highest tier the card can claim once the margin is added."""
    if not total_gb:
        return 12                                     # no GPU detected - middle of the road
    budget = total_gb + VRAM_TIER_MARGIN_GB
    eligible = [t for t in VRAM_TIERS if t <= budget]
    return eligible[-1] if eligible else VRAM_TIERS[0]


# `peak_gb` is measured, not estimated. Every preset below was run end to end through the
# app's own isolated worker (generate -> GLB -> Gaussian) while a background thread sampled
# `torch.cuda.mem_get_info()` every 50 ms, so these are true device occupancy - allocator
# blocks, CUDA context and driver overhead included. torch's own `max_memory_allocated`
# reports roughly half of this because it only counts live tensors; do not trust it here.
#
# The shape of the numbers is worth knowing before touching any of these values:
#   * The peak is almost always the SLat mesh decoder. It expands ~17k voxels to a 256³
#     flexicubes grid and costs ~7.6 GB in fp16 / ~12.0 GB in fp32 - and it does not care
#     about video resolution, texture size, step counts or process isolation.
#   * Which is why `generate_mesh_val` exists. Turning the mesh off is the only thing that
#     moves that number, and it is what makes the 6 GB tier real rather than aspirational.
#   * Everything else is comparatively cheap: video rendering peaks ~3.7-5.5 GB, GLB
#     texture baking ~3.8-9.0 GB depending on bake_resolution x bake_nviews.
#
# `peak_gb` is a typical subject and `peak_max_gb` is the worst case measured, because the
# mesh decoder scales with how much of the volume the subject occupies: the reference
# dragon activates ~17k voxels and peaks at 7.65 GB in fp16, a detailed cottage activates
# ~20k and peaks at 8.08 GB. Quote the range, never just the low end.
#
# Measured on an RTX 5090 (170 SMs, 1.69 GB CUDA context). Cards with fewer SMs carry a
# smaller context - an RTX 3090 measured 1.25 GB - so a small GPU lands 0.4-0.8 GB below
# these figures. They are therefore a safe upper bound, not an optimistic one.
VRAM_PRESET_SPECS: Dict[int, Dict[str, Any]] = {
    6: {
        "peak_gb": 5.1,
        "peak_max_gb": 5.5,
        "mesh_peak_gb": 7.6,
        "note": "Gaussian splat + turntable video only. The mesh decoder needs ~7.6 GB "
                "even in fp16, which no 6 GB card can give it, so this tier skips it.",
        "values": {
            "precision_val": "fp16", "isolate_jobs_val": True, "generate_mesh_val": False,
            "ss_sampling_steps_val": 12, "slat_sampling_steps_val": 12,
            "video_resolution_val": 512, "video_num_frames_val": 120, "video_fps_val": 30,
            "video_quality_val": 7, "include_geometry_val": False,
            "mesh_simplify_val": 0.96, "texture_size_val": 512,
            "bake_resolution_val": 384, "bake_nviews_val": 40,
        },
    },
    8: {
        "peak_gb": 7.6,
        "peak_max_gb": 8.1,
        "note": "Everything works, but the mesh decode fills the card — close other GPU "
                "apps first. Drop to the 6 GB tier if you hit an out-of-memory error.",
        "values": {
            "precision_val": "fp16", "isolate_jobs_val": True, "generate_mesh_val": True,
            "ss_sampling_steps_val": 12, "slat_sampling_steps_val": 12,
            "video_resolution_val": 512, "video_num_frames_val": 150, "video_fps_val": 30,
            "video_quality_val": 8, "include_geometry_val": False,
            "mesh_simplify_val": 0.95, "texture_size_val": 1024,
            "bake_resolution_val": 512, "bake_nviews_val": 60,
        },
    },
    10: {
        "peak_gb": 7.6,
        "peak_max_gb": 8.1,
        "note": "Full feature set including the geometry pass in the preview video.",
        "values": {
            "precision_val": "fp16", "isolate_jobs_val": True, "generate_mesh_val": True,
            "ss_sampling_steps_val": 14, "slat_sampling_steps_val": 14,
            "video_resolution_val": 768, "video_num_frames_val": 180, "video_fps_val": 30,
            "video_quality_val": 8, "include_geometry_val": True,
            "mesh_simplify_val": 0.94, "texture_size_val": 1024,
            "bake_resolution_val": 768, "bake_nviews_val": 80,
        },
    },
    12: {
        "peak_gb": 7.6,
        "peak_max_gb": 8.1,
        "note": "Half precision keeps the mesh decoder at ~7.6 GB; fp32 would want ~12.1 GB "
                "and leave nothing for the desktop on a 12 GB card.",
        "values": {
            "precision_val": "fp16", "isolate_jobs_val": False, "generate_mesh_val": True,
            "ss_sampling_steps_val": 16, "slat_sampling_steps_val": 16,
            "video_resolution_val": 1024, "video_num_frames_val": 240, "video_fps_val": 60,
            "video_quality_val": 8, "include_geometry_val": True,
            "mesh_simplify_val": 0.92, "texture_size_val": 1024,
            "bake_resolution_val": 1024, "bake_nviews_val": 100,
        },
    },
    16: {
        "peak_gb": 12.1,
        "peak_max_gb": 12.9,
        "note": "Full fp32 precision — the reference quality setting.",
        "values": {
            "precision_val": "fp32", "isolate_jobs_val": False, "generate_mesh_val": True,
            "ss_sampling_steps_val": 18, "slat_sampling_steps_val": 18,
            "video_resolution_val": 1024, "video_num_frames_val": 240, "video_fps_val": 60,
            "video_quality_val": 9, "include_geometry_val": True,
            "mesh_simplify_val": 0.90, "texture_size_val": 1024,
            "bake_resolution_val": 1024, "bake_nviews_val": 100,
        },
    },
    24: {
        "peak_gb": 12.1,
        "peak_max_gb": 12.9,
        "note": "fp32 with a 2K baked texture and a denser mesh.",
        "values": {
            "precision_val": "fp32", "isolate_jobs_val": False, "generate_mesh_val": True,
            "ss_sampling_steps_val": 20, "slat_sampling_steps_val": 20,
            "video_resolution_val": 1024, "video_num_frames_val": 300, "video_fps_val": 60,
            "video_quality_val": 9, "include_geometry_val": True,
            "mesh_simplify_val": 0.88, "texture_size_val": 2048,
            "bake_resolution_val": 1024, "bake_nviews_val": 120,
        },
    },
    32: {
        "peak_gb": 12.1,
        "peak_max_gb": 12.9,
        "note": "Maximum quality — nothing is traded away for memory.",
        "values": {
            "precision_val": "fp32", "isolate_jobs_val": False, "generate_mesh_val": True,
            "ss_sampling_steps_val": 25, "slat_sampling_steps_val": 25,
            "video_resolution_val": 1536, "video_num_frames_val": 360, "video_fps_val": 60,
            "video_quality_val": 10, "include_geometry_val": True,
            "mesh_simplify_val": 0.85, "texture_size_val": 2048,
            "bake_resolution_val": 1024, "bake_nviews_val": 150,
        },
    },
}


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
        self.precision = cmd_args.precision

    # -- public ---------------------------------------------------------------------
    def ensure(self, job: Optional[JobState] = None, precision: Optional[str] = None):
        """Return the pipeline, building it first if needed.

        `precision` comes from the active preset, so switching between a fp32 and a fp16
        tier has to rebuild the pipeline - the weights themselves are cast in place.
        """
        want = precision if precision in ("fp16", "fp32") else self.precision
        if self.ready and want == self.precision:
            return self.pipeline
        with self.lock:
            if self.ready and want == self.precision:
                return self.pipeline
            if self.ready and want != self.precision:
                self._unload(job, f"switching precision {self.precision} -> {want}")
            self.precision = want
            self._load(job)
        return self.pipeline

    def _unload(self, job: Optional[JobState], why: str) -> None:
        msg = f"Releasing the current pipeline ({why})."
        if job is not None:
            job.log(msg, level="warn")
        else:
            cprint(f"[{stamp()}] {msg}", "yellow")
        self.ready = False
        self.pipeline = None
        import gc
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()

    def status_line(self) -> str:
        if self.ready:
            return (f"Engine ready ({self.device_name}, {self.precision}, "
                    f"loaded in {self.load_seconds:.1f}s)")
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

            if self.precision == "fp16":
                note("Converting the pipeline to half precision (fp16) - this is what keeps "
                     "the mesh decoder inside ~7.7 GB instead of ~12.1 GB.")
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
            # A missing CUDA .so is the one failure here with a one-command cure, so say
            # so instead of leaving the user with a bare loader message.  Empty on
            # Windows and for every other kind of error.
            hint = cuda_linux_deps.missing_library_hint(str(exc)) if cuda_linux_deps else ""
            if hint:
                note(hint, level="err")
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
    """`mesh` is None in Gaussian-only mode, which is what lets a 6 GB card finish a run -
    decoding the mesh alone needs ~7.6 GB and no setting brings that down."""
    state = {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
    }
    if mesh is not None:
        state["mesh"] = {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        }
    return state


def state_has_mesh(state: Optional[dict]) -> bool:
    return bool(state) and bool(state.get("mesh"))


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

    mesh = None
    if state_has_mesh(state):
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
    generate_mesh: bool = True,
    output_filename_prefix: Optional[str] = None,
    custom_output_dirs: Optional[dict] = None,
) -> Tuple[dict, Optional[str]]:
    """One image -> 3D run. Returns (state, video_path or None).

    With `generate_mesh=False` the structured latent is only decoded into a Gaussian
    splat. That skips the mesh decoder, which is the single biggest allocation in the
    whole pipeline (~7.6 GB in fp16, ~12.0 GB in fp32, and unaffected by every other
    setting) - so it is what makes a 6 GB card able to finish a run at all. The cost is
    that there is no mesh, hence no GLB and no geometry pass in the preview video.
    """
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

    formats = ["gaussian", "mesh"] if generate_mesh else ["gaussian"]
    include_geometry = bool(include_geometry) and generate_mesh

    job.stage(f"Sparse structure + latent sampling (seed {seed})", bars=2)
    job.set_detail(f"SS {ss_sampling_steps} steps @ cfg {ss_guidance_strength} · "
                   f"SLat {slat_sampling_steps} steps @ cfg {slat_guidance_strength}"
                   + ("" if generate_mesh else " · Gaussian only (low-VRAM mode)"))
    if is_multiimage:
        outputs = pipeline.run_multi_image(
            images, seed=int(seed), formats=formats, preprocess_image=False,
            sparse_structure_sampler_params=sampler_ss, slat_sampler_params=sampler_slat,
            mode=multiimage_algo, cancel_event=job.cancel_event,
        )
    else:
        outputs = pipeline.run(
            image, seed=int(seed), formats=formats, preprocess_image=False,
            sparse_structure_sampler_params=sampler_ss, slat_sampler_params=sampler_slat,
            cancel_event=job.cancel_event,
        )
    job.stage_done()
    job.raise_if_cancelled()

    gaussian = outputs["gaussian"][0]
    mesh = outputs["mesh"][0] if generate_mesh else None
    if mesh is not None:
        job.log(f"  Mesh: {mesh.vertices.shape[0]:,} vertices / {mesh.faces.shape[0]:,} triangles | "
                f"Gaussians: {gaussian._xyz.shape[0]:,}")
    else:
        job.log(f"  Gaussians: {gaussian._xyz.shape[0]:,} (mesh decoding skipped to save VRAM - "
                f"GLB extraction is not available for this run)", level="warn")

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
            "precision": ENGINE.precision,
            "mesh_generated": mesh is not None,
            "mesh_vertices": int(mesh.vertices.shape[0]) if mesh is not None else None,
            "mesh_triangles": int(mesh.faces.shape[0]) if mesh is not None else None,
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
                     save_metadata: bool, bake_resolution: int = 1024,
                     bake_nviews: int = 100) -> str:
    torch = ENGINE.torch
    if not state_has_mesh(state):
        raise RuntimeError(
            "This run was generated in Gaussian-only mode, so there is no mesh to turn into "
            "a GLB. Tick 'Also decode a mesh' (needs about 7.6 GB of VRAM in fp16) and "
            "generate again — or use the Gaussian .ply, which is already available.")
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
                   f"simplify {mesh_simplify} · texture {texture_size}px · "
                   f"bake {int(bake_resolution)}px × {int(bake_nviews)} views")
    glb_data = ENGINE.postprocessing_utils.to_glb(
        gs, mesh, simplify=float(mesh_simplify), texture_size=int(texture_size),
        verbose=True, cancel_event=job.cancel_event,
        bake_resolution=int(bake_resolution), bake_nviews=int(bake_nviews),
        # The hole-filling rasteriser does not need to be finer than the views we bake from.
        fill_holes_resolution=max(512, min(1024, int(bake_resolution))),
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
    "bake_resolution_val", "bake_nviews_val",
    "video_resolution_val", "video_num_frames_val", "video_fps_val",
    "video_quality_val", "include_geometry_val",
    "save_metadata_val",
    "precision_val", "isolate_jobs_val", "generate_mesh_val",
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
        "bake_resolution_val": 1024,
        "bake_nviews_val": 100,
        "video_resolution_val": 1024,
        "video_num_frames_val": 240,
        "video_fps_val": 60,
        "video_quality_val": 8,
        "include_geometry_val": True,
        "save_metadata_val": True,
        "precision_val": cmd_args.precision,
        "isolate_jobs_val": False,
        "generate_mesh_val": True,
        "batch_input_folder_val": DEFAULT_BATCH_INPUT,
        "batch_output_folder_val": BATCH_OUTPUT_DIR_BASE_DEFAULT,
        "batch_skip_existing_val": True,
        "batch_gen_video_cb_val": True,
        "batch_extract_glb_cb_val": True,
        "batch_extract_gaussian_cb_val": True,
    }


def vram_preset_values(tier_gb: int) -> Dict[str, Any]:
    """A full config for one VRAM tier: the defaults with that tier's overrides on top."""
    values = get_default_config_values()
    values.update(VRAM_PRESET_SPECS[tier_gb]["values"])
    return values


BUILTIN_PRESET_NAMES: List[str] = [DEFAULT_CONFIG_NAME] + [vram_preset_name(t) for t in VRAM_TIERS]


def write_builtin_presets() -> None:
    """Rewrite the protected folder from code on every start.

    These files exist so the user can read (and copy) them, not so they can edit them -
    anything they change here is overwritten next launch, which is exactly what makes
    them safe to fall back on.
    """
    os.makedirs(PRESET_DIR_BUILTIN, exist_ok=True)
    payloads = {DEFAULT_CONFIG_NAME: get_default_config_values()}
    for tier in VRAM_TIERS:
        payloads[vram_preset_name(tier)] = vram_preset_values(tier)
    for name, data in payloads.items():
        try:
            with open(os.path.join(PRESET_DIR_BUILTIN, f"{name}.json"), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
        except OSError as exc:                                     # pragma: no cover
            cprint(f"Could not write protected preset '{name}': {exc}", "yellow")

    # Presets from before the split lived directly in configs_trellis/. Move any that are
    # not ours into the user folder so nobody loses their saved settings on upgrade.
    try:
        for entry in os.listdir(CONFIG_DIR):
            if not entry.endswith(".json") or entry == os.path.basename(LAST_CONFIG_FILE):
                continue
            src = os.path.join(CONFIG_DIR, entry)
            if not os.path.isfile(src):
                continue
            name = os.path.splitext(entry)[0]
            if name in BUILTIN_PRESET_NAMES:
                os.remove(src)                                     # superseded by the protected copy
                continue
            dst = os.path.join(PRESET_DIR_USER, entry)
            if not os.path.exists(dst):
                os.replace(src, dst)
                cprint(f"[{stamp()}] Moved your preset '{name}' into presets_user/.", "cyan")
    except OSError:
        pass


def is_builtin_preset(name: Optional[str]) -> bool:
    return (name or "") in BUILTIN_PRESET_NAMES


def user_preset_names() -> List[str]:
    os.makedirs(PRESET_DIR_USER, exist_ok=True)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR_USER)
                  if f.endswith(".json"))


def get_config_list() -> List[str]:
    """Protected presets first (in tier order), then the user's own."""
    return BUILTIN_PRESET_NAMES + user_preset_names()


def preset_path(name: str) -> Optional[str]:
    for folder in (PRESET_DIR_BUILTIN, PRESET_DIR_USER):
        candidate = os.path.join(folder, f"{name}.json")
        if os.path.exists(candidate):
            return candidate
    return None


def read_preset(name: str) -> Optional[Dict[str, Any]]:
    path = preset_path(name)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _ordered_values(config_data: Dict[str, Any]) -> List[Any]:
    defaults = get_default_config_values()
    return [config_data.get(key, defaults[key]) for key in PRESET_KEYS]


def _remember_last(config_name: str, config_data: Dict[str, Any]) -> None:
    try:
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump({
                "last_config_name": config_name,
                "last_was_user_preset": not is_builtin_preset(config_name),
                "data": config_data,
            }, fh, indent=4)
    except OSError:
        pass


def save_config(config_name: str, *values):
    config_name = (config_name or "").strip()
    if not config_name:
        return "Please type a preset name first.", gr.update(choices=get_config_list())
    if re.search(r"[\\/:*?\"<>|]", config_name):
        return "Preset name contains invalid characters.", gr.update(choices=get_config_list())
    if is_builtin_preset(config_name):
        return (f"'{config_name}' is a protected preset and cannot be overwritten. "
                f"Pick another name — your presets are saved separately in presets_user/.",
                gr.update(choices=get_config_list()))

    config_data = {key: val for key, val in zip(PRESET_KEYS, values)}
    try:
        os.makedirs(PRESET_DIR_USER, exist_ok=True)
        with open(os.path.join(PRESET_DIR_USER, f"{config_name}.json"), "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=4)
        _remember_last(config_name, config_data)
        cprint(f"[{stamp()}] Preset '{config_name}' saved to presets_user/.", "green")
        return (f"Preset '{config_name}' saved to presets_user/.",
                gr.update(choices=get_config_list(), value=config_name))
    except Exception as exc:                                       # noqa: BLE001
        return f"Error saving preset: {exc}", gr.update(choices=get_config_list())


def load_config(config_name_to_load: Optional[str]):
    defaults = get_default_config_values()
    if not config_name_to_load:
        _remember_last(DEFAULT_CONFIG_NAME, defaults)
        return tuple(["Loaded built-in defaults."] + _ordered_values(defaults))

    config_data = read_preset(config_name_to_load)
    if config_data is None:
        return tuple([f"Preset '{config_name_to_load}' could not be read - loaded defaults."]
                     + _ordered_values(defaults))

    _remember_last(config_name_to_load, config_data)
    kind = "protected" if is_builtin_preset(config_name_to_load) else "your"
    cprint(f"[{stamp()}] Loaded {kind} preset '{config_name_to_load}'.", "cyan")
    return tuple([f"Loaded {kind} preset '{config_name_to_load}'."] + _ordered_values(config_data))


def delete_config(config_name: Optional[str]):
    if not config_name:
        return "Select a preset to delete.", gr.update(choices=get_config_list())
    if is_builtin_preset(config_name):
        return (f"'{config_name}' is protected and cannot be deleted.",
                gr.update(choices=get_config_list()))
    path = os.path.join(PRESET_DIR_USER, f"{config_name}.json")
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


def startup_preset_name() -> Tuple[str, str]:
    """Which preset the app should come up with, and why.

    A preset the user saved themselves always wins - if they went to the trouble of
    saving one, re-applying a VRAM tier over the top of it would throw their work away.
    Otherwise the tier that matches the detected card is used.
    """
    auto_tier = pick_vram_tier(GPU_VRAM_GB)
    auto_name = vram_preset_name(auto_tier)

    if os.path.exists(LAST_CONFIG_FILE):
        try:
            with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, json.JSONDecodeError):
            saved = {}
        candidate = saved.get("last_config_name")
        if candidate and not is_builtin_preset(candidate) and preset_path(candidate):
            return candidate, f"restored your last preset '{candidate}'"

    if GPU_VRAM_GB:
        return auto_name, (f"auto-selected for {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) "
                           f"-> {auto_tier} GB tier")
    return auto_name, "no GPU detected - using the 12 GB tier"


def vram_info_html(preset_name: Optional[str]) -> str:
    """The peak-VRAM card shown under the preset picker."""
    esc = html_mod.escape
    if GPU_VRAM_GB:
        gpu_line = f"{esc(GPU_NAME or 'GPU')} &middot; {GPU_VRAM_GB:.1f} GB detected"
    else:
        gpu_line = "No NVIDIA GPU detected (nvidia-smi not available)"

    tier = None
    for candidate in VRAM_TIERS:
        if preset_name == vram_preset_name(candidate):
            tier = candidate
            break

    if tier is None:
        values = read_preset(preset_name or "") or get_default_config_values()
        prec = values.get("precision_val", cmd_args.precision)
        if not values.get("generate_mesh_val", True):
            lo, hi, what = 5.1, 5.5, "Gaussian only"
        else:
            # fp32 keeps the flexicubes grid in the mesh decoder at full width, which is
            # the single biggest allocation in the whole app.
            lo, hi, what = (12.1, 12.9, "with mesh") if prec == "fp32" else (7.6, 8.1, "with mesh")
        body = (f"<b>{esc(str(preset_name or 'Custom'))}</b> — not a VRAM tier. "
                f"At <code>{esc(str(prec))}</code> precision, {what}, expect a peak of "
                f"<b>{lo:.1f} – {hi:.1f} GB</b> depending on how busy the subject is.")
        return (f'<div class="tp-vram"><div class="tp-vram-gpu">{gpu_line}</div>'
                f'<div class="tp-vram-body">{body}</div></div>')

    spec = VRAM_PRESET_SPECS[tier]
    values = spec["values"]
    peak = spec["peak_gb"]
    peak_max = spec.get("peak_max_gb", peak)
    # The tier's own budget is the honest bar, judged on the worst case rather than the
    # typical one: a 6 GB preset has to fit 6 GB even when previewed on a 32 GB card, and
    # even when the subject is a busy one.
    fits = peak_max <= tier + VRAM_TIER_MARGIN_GB
    klass = "tp-vram-ok" if fits else "tp-vram-warn"

    rows = (f"precision <code>{esc(values['precision_val'])}</code> &middot; "
            f"{'mesh + Gaussian' if values.get('generate_mesh_val', True) else '<b>Gaussian only</b>'}"
            f" &middot; video {values['video_resolution_val']}px / "
            f"{values['video_num_frames_val']} frames"
            f"{' + geometry pass' if values['include_geometry_val'] else ''} &middot; "
            f"texture {values['texture_size_val']}px &middot; "
            f"bake {values['bake_resolution_val']}px × {values['bake_nviews_val']} views &middot; "
            f"steps {values['ss_sampling_steps_val']}/{values['slat_sampling_steps_val']}"
            f"{' &middot; isolated subprocess' if values.get('isolate_jobs_val') else ''}")

    extra = ""
    if spec.get("mesh_peak_gb"):
        extra = (f'<div class="tp-vram-warnline">No GLB on this tier. Decoding a mesh peaks at '
                 f'<b>{spec["mesh_peak_gb"]:.1f} GB</b> regardless of every other setting — '
                 f'the decoder expands the latent into a 256³ grid — so this preset asks for '
                 f'the Gaussian splat only. You still get a <code>.ply</code> and the turntable '
                 f'video; tick “Also decode a mesh” if you want to try the GLB anyway.</div>')
    elif not fits:
        extra = (f'<div class="tp-vram-warnline">⚠ This preset peaks above its own {tier} GB '
                 f'budget. Pick a lower tier.</div>')

    return f"""
<div class="tp-vram {klass}">
  <div class="tp-vram-gpu">{gpu_line}</div>
  <div class="tp-vram-peak">Peak VRAM <b>{peak:.1f} – {peak_max:.1f} GB</b>
    <span>of {tier} GB &middot; typical → busiest subject</span></div>
  <div class="tp-vram-body">{rows}</div>
  <div class="tp-vram-note">{esc(spec['note'])}</div>
  {extra}
</div>""".strip()


def apply_vram_preset(tier_gb: int):
    """Load one VRAM tier into every control, and select it in the preset dropdown."""
    name = vram_preset_name(int(tier_gb))
    values = read_preset(name) or vram_preset_values(int(tier_gb))
    _remember_last(name, values)
    cprint(f"[{stamp()}] Applied protected preset '{name}'.", "cyan")
    return tuple([f"Applied '{name}'.", gr.update(value=name), vram_info_html(name)]
                 + _ordered_values(values))


def initial_load_config():
    """Restore the preset the app should start with (see startup_preset_name)."""
    name, why = startup_preset_name()
    config_data = read_preset(name) or get_default_config_values()
    choices = get_config_list()
    if name not in choices:
        name = DEFAULT_CONFIG_NAME
        config_data = read_preset(name) or get_default_config_values()

    tier = pick_vram_tier(GPU_VRAM_GB)
    cprint(f"[{stamp()}] UI ready - {why}.", "cyan")
    return tuple([gr.update(choices=choices, value=name),
                  gr.update(value=tier),
                  vram_info_html(name)] + _ordered_values(config_data))


# --------------------------------------------------------------------------------------
# Isolated jobs (each run in its own process, so VRAM goes back to zero afterwards)
# --------------------------------------------------------------------------------------
# Why a whole process rather than `torch.cuda.empty_cache()`: emptying the cache returns
# the allocator's blocks but leaves the CUDA context, the cuBLAS/cuDNN workspaces and
# whatever spconv and flash-attn have cached - measured at 1.25 GB on an RTX 3090 and
# 1.69 GB on an RTX 5090. Only tearing the process down gives that back, and on an 8 GB
# card it is the difference between the next run fitting and not. It also means a preset
# can pick fp16 or fp32 per run without restarting the server, and an out-of-memory kill
# takes the child down instead of the web UI.
WORKER_PROGRESS_PREFIX = "@@TRELLIS@@ "


def _worker_emit(kind: str, **payload) -> None:
    """Called inside the child process: one JSON event per line on stdout.

    The child's stderr is merged into the same pipe, and tqdm draws its bars there with
    a bare \\r and no newline - so an event written plainly would be glued onto the end
    of a half-drawn bar and never parse. The leading newline guarantees the marker starts
    a line, and writing the whole thing in one call keeps it atomic in the pipe.
    """
    sys.stdout.write("\n" + WORKER_PROGRESS_PREFIX + json.dumps({"kind": kind, **payload}) + "\n")
    sys.stdout.flush()


class _WorkerJobProxy(JobState):
    """A JobState whose updates are forwarded to the parent instead of a browser."""

    #  Rendering 240 frames fires a tqdm update per frame; at one pipe write each that is
    #  pure overhead, since the parent only repaints the browser every GEN_POLL seconds.
    BAR_MIN_INTERVAL = 0.15

    def __init__(self, title: str = "Working"):
        super().__init__(title)
        self._suppress_log = False
        self._last_bar_emit = 0.0

    def log(self, msg: str, level: str = "info", console: bool = True) -> None:
        # console=False: the parent prints every forwarded line itself, and the child's
        # stdout is piped into the parent, so echoing here would double every log line.
        super().log(msg, level=level, console=False)
        if not self._suppress_log:
            _worker_emit("log", msg=msg, level=level)

    def stage(self, name: str, index: Optional[int] = None, bars: int = 1) -> None:
        # JobState.stage() logs "> Step i/n" using the child's own plan, which is not the
        # plan the parent is tracking. Swallow that line and let the parent write its own.
        self._suppress_log = True
        try:
            super().stage(name, index=index, bars=bars)
        finally:
            self._suppress_log = False
        # No index is sent: the child counts its own stages from 1, while the parent is
        # tracking a plan that spans several child processes. The parent just advances.
        _worker_emit("stage", name=name, bars=bars)

    def bar_start(self, desc: str, total: Optional[int]) -> None:
        super().bar_start(desc, total)
        self._last_bar_emit = 0.0
        _worker_emit("bar_start", desc=desc or "", total=int(total or 0))

    def bar_update(self, n: int, total: Optional[int], rate: Optional[float]) -> None:
        super().bar_update(n, total, rate)
        now = time.time()
        final = bool(total) and int(n or 0) >= int(total)
        if final or now - self._last_bar_emit >= self.BAR_MIN_INTERVAL:
            self._last_bar_emit = now
            _worker_emit("bar", n=int(n or 0), total=int(total or 0), rate=rate)

    def set_detail(self, text: str) -> None:
        super().set_detail(text)
        _worker_emit("detail", text=text)

    def set_headline(self, text: str) -> None:
        super().set_headline(text)
        _worker_emit("headline", text=text)


def state_to_disk(state: dict, path: str) -> str:
    """Persist a packed generation state so another process can pick it up."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"filename_base": state.get("filename_base", "")}
    if state.get("custom_output_dirs"):
        meta["custom_output_dirs"] = state["custom_output_dirs"]
    for group in ("gaussian", "mesh"):
        if not state.get(group):                                   # Gaussian-only runs have no mesh
            continue
        for key, value in state[group].items():
            if isinstance(value, np.ndarray):
                flat[f"{group}.{key}"] = value
            else:
                meta[f"{group}.{key}"] = value
    np.savez(path, __meta__=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **flat)
    return path


def state_from_disk(path: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
        state: Dict[str, Any] = {"gaussian": {}}
        for key in data.files:
            if key == "__meta__":
                continue
            group, _, field = key.partition(".")
            state.setdefault(group, {})[field] = data[key]
    for key, value in meta.items():
        group, _, field = key.partition(".")
        if group in ("gaussian", "mesh") and field:
            state.setdefault(group, {})[field] = value
        else:
            state[key] = value
    return state


def _worker_argv(job_path: str) -> List[str]:
    argv = [sys.executable, os.path.abspath(__file__), "--worker", job_path]
    if cmd_args.attention:
        argv += ["--attention", cmd_args.attention]
    if cmd_args.xformers:
        argv.append("--xformers")
    if cmd_args.highvram:
        argv.append("--highvram")
    if cmd_args.no_tf32:
        argv.append("--no-tf32")
    return argv


def run_isolated(job: JobState, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one unit of work in a child process and mirror its progress into `job`.

    Returns the child's result dict. Raises on failure so the caller's normal error
    path reports it, and translates a cancel into JobCancelled.
    """
    token = f"{int(time.time() * 1000)}_{os.getpid()}"
    job_path = os.path.join(JOB_TMP_DIR, f"job_{token}.json")
    result_path = os.path.join(JOB_TMP_DIR, f"result_{token}.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump({"kind": kind, "payload": payload, "result_path": result_path}, fh)

    job.log(f"Starting an isolated worker process ({kind}) - VRAM returns to zero when it exits.")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        _worker_argv(job_path), cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=creationflags,
    )

    def watch_cancel():
        while proc.poll() is None:
            if job.cancel_event.wait(0.25):
                try:
                    proc.terminate()
                except OSError:                                    # pragma: no cover
                    pass
                return

    threading.Thread(target=watch_cancel, name="trellis-worker-cancel", daemon=True).start()

    tail: List[str] = []
    try:
        for line in proc.stdout:                                   # type: ignore[union-attr]
            line = line.rstrip("\n")
            # The marker is written at the start of its own line, but a stray \r-drawn
            # tqdm fragment can still share the buffer, so search rather than match.
            marker = line.find(WORKER_PROGRESS_PREFIX)
            if marker >= 0:
                before = line[:marker].rsplit("\r", 1)[-1].strip()
                if before:
                    cprint(f"  │ {before}", "dim")
                try:
                    evt = json.loads(line[marker + len(WORKER_PROGRESS_PREFIX):])
                except json.JSONDecodeError:                       # pragma: no cover
                    continue
                kind_ = evt.get("kind")
                if kind_ == "log":
                    JobState.log(job, evt.get("msg", ""), level=evt.get("level", "info"))
                elif kind_ == "stage":
                    JobState.stage(job, evt.get("name", ""), bars=int(evt.get("bars", 1) or 1))
                elif kind_ == "bar_start":
                    JobState.bar_start(job, evt.get("desc", ""), evt.get("total"))
                elif kind_ == "bar":
                    JobState.bar_update(job, evt.get("n", 0), evt.get("total"), evt.get("rate"))
                elif kind_ == "detail":
                    JobState.set_detail(job, evt.get("text", ""))
                elif kind_ == "headline":
                    JobState.set_headline(job, evt.get("text", ""))
            elif line.strip():
                # tqdm redraws with \r and only terminates the line when the bar closes,
                # so one "line" can hold a whole bar's history - show just the final state.
                shown = line.rsplit("\r", 1)[-1].strip()
                if shown:
                    cprint(f"  │ {shown}", "dim")
                tail.append(shown or line)
                del tail[:-40]
    finally:
        proc.stdout.close()                                        # type: ignore[union-attr]
        code = proc.wait()

    if job.cancelled:
        raise JobCancelled("Cancelled by user.")

    if code != 0 or not os.path.exists(result_path):
        detail = "\n".join(tail[-15:]) or f"exit code {code}"
        raise RuntimeError(f"The isolated worker failed (exit {code}).\n{detail}")

    with open(result_path, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    for path in (job_path, result_path):
        try:
            os.remove(path)
        except OSError:
            pass
    if result.get("error"):
        raise RuntimeError(result["error"])
    if result.get("peak_vram_gb"):
        job.log(f"  Worker peak VRAM: {result['peak_vram_gb']:.2f} GB "
                f"(process has exited - the GPU is idle again)", level="ok")
    return result


def worker_main(job_path: str) -> int:
    """Entry point of the child process (``--worker <jobfile>``)."""
    with open(job_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    kind = spec["kind"]
    payload = spec["payload"]
    result_path = spec["result_path"]

    job = _WorkerJobProxy(kind)
    _set_thread_job(job)
    out: Dict[str, Any] = {}
    peak_sampler_stop = threading.Event()
    peak = {"gb": 0.0}

    try:
        job.stage("Preparing the engine")
        ENGINE.ensure(job, precision=payload.get("precision"))
        job.stage_done()

        torch = ENGINE.torch
        if torch is not None and torch.cuda.is_available():
            def sample_peak():
                while not peak_sampler_stop.is_set():
                    try:
                        free, total = torch.cuda.mem_get_info()
                        used = (total - free) / 1024 ** 3
                        if used > peak["gb"]:
                            peak["gb"] = used
                    except Exception:                              # pragma: no cover
                        return
                    time.sleep(0.05)
            threading.Thread(target=sample_peak, daemon=True).start()

        if kind == "generate":
            kwargs = dict(payload["kwargs"])
            # The parent could only hand us file paths; generate_one's _ensure_pil opens
            # a str straight into a PIL image, so map them back onto its real arguments.
            kwargs["image"] = kwargs.pop("image_path", None)
            kwargs["multiimages"] = kwargs.pop("multiimage_paths", None)
            state, video_path = generate_one(job, **kwargs)
            state_path = payload["state_path"]
            state_to_disk(state, state_path)
            out = {"state_path": state_path, "video": video_path,
                   "filename_base": state.get("filename_base"),
                   "has_mesh": state_has_mesh(state)}
        elif kind == "extract_glb":
            state = state_from_disk(payload["state_path"])
            out = {"glb": extract_glb_file(job, state, **payload["kwargs"])}
        elif kind == "extract_gaussian":
            state = state_from_disk(payload["state_path"])
            out = {"gs": extract_gaussian_file(job, state)}
        else:
            raise ValueError(f"Unknown worker job kind: {kind!r}")
    except JobCancelled:
        out = {"error": "Cancelled by user."}
    except BaseException as exc:                                   # noqa: BLE001
        traceback.print_exc()
        out = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        peak_sampler_stop.set()
        _set_thread_job(None)

    out["peak_vram_gb"] = round(peak["gb"], 3)
    try:
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    except OSError as exc:                                         # pragma: no cover
        cprint(f"Worker could not write its result file: {exc}", "red")
        return 1
    return 1 if out.get("error") else 0


# --------------------------------------------------------------------------------------
# UI handlers
# --------------------------------------------------------------------------------------
# How often the streaming handlers push a status snapshot to the browser. Each push
# re-renders the status HTML, so this trades smoothness against DOM churn.
GEN_POLL = 0.4


def _skips(n: int):
    return tuple(gr.skip() for _ in range(n))


def explain_failure(error) -> str:
    """Turn a raw failure into something the user can act on.

    An out-of-memory here is almost always the mesh decoder, and the fix is a specific
    setting rather than 'try again' - so say which one instead of echoing a CUDA dump.
    """
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    lowered = text.lower()
    if "out of memory" in lowered or ("cuda error" in lowered and "memory" in lowered):
        tier = pick_vram_tier(GPU_VRAM_GB)
        hint = ("Ran out of VRAM. The mesh decoder is what needs the most (~7.6 GB in fp16, "
                "~12.1 GB in fp32) and no other setting reduces it. Try, in order: "
                "set precision to fp16; tick 'Run each job in its own process'; "
                "untick 'Also decode a mesh' to get the Gaussian splat and video only "
                "(under 5 GB); close other GPU applications.")
        if tier <= 8:
            hint += f" Your card is on the {tier} GB tier, where this is expected to be tight."
        return f"{hint}\n\nOriginal error: {text}"
    return text


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


# --------------------------------------------------------------------------------------
# One entry point per operation, so the in-process and isolated paths never diverge
# --------------------------------------------------------------------------------------
def _new_job_tmp(suffix: str) -> str:
    os.makedirs(JOB_TMP_DIR, exist_ok=True)
    return os.path.join(JOB_TMP_DIR, f"{int(time.time() * 1000)}_{os.getpid()}_{suffix}")


def sweep_job_tmp(max_age_hours: float = 24.0) -> None:
    """Drop handoff files left behind by a crash or a kill.

    A Gaussian state is ~40 MB, so without this the folder grows without bound. Anything
    still needed by the current session is younger than this cutoff by a wide margin.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    try:
        entries = os.listdir(JOB_TMP_DIR)
    except OSError:
        return
    for name in entries:
        path = os.path.join(JOB_TMP_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    if removed:
        cprint(f"[{stamp()}] Cleaned {removed} stale job file(s) from {JOB_TMP_DIR}.", "dim")


def _stash_images_for_worker(image, multiimages, is_multiimage) -> Tuple[Dict[str, Any], List[str]]:
    """A child process cannot be handed PIL objects, so put the pixels on disk.

    Returns the kwargs plus the list of files *we* created, so the caller can delete them
    again - a path the user already had on disk is passed straight through and must not
    end up in that list.
    """
    from PIL import Image as PILImage                              # cheap, pillow is already up
    created: List[str] = []
    if is_multiimage:
        paths = []
        for idx, item in enumerate(multiimages or []):
            raw = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(raw, str):
                paths.append(raw)
                continue
            path = _new_job_tmp(f"view{idx:02d}.png")
            (raw if isinstance(raw, PILImage.Image) else PILImage.fromarray(raw)).save(path)
            paths.append(path)
            created.append(path)
        return {"multiimage_paths": paths, "image_path": None}, created
    if isinstance(image, str):
        return {"image_path": image, "multiimage_paths": None}, created
    path = _new_job_tmp("input.png")
    (image if isinstance(image, PILImage.Image) else PILImage.fromarray(image)).save(path)
    created.append(path)
    return {"image_path": path, "multiimage_paths": None}, created


def state_ref(value) -> Tuple[Optional[dict], Optional[str]]:
    """Unpack whatever is sitting in the output gr.State into (state dict, npz path)."""
    if not value:
        return None, None
    if isinstance(value, dict) and "__state_path__" in value:
        return None, value["__state_path__"]
    return value, None


def _materialise_state(value) -> dict:
    state, path = state_ref(value)
    if state is not None:
        return state
    if path and os.path.exists(path):
        return state_from_disk(path)
    raise RuntimeError("The generation result is no longer available - please generate again.")


def _discard_state_file(value) -> None:
    """Delete the .npz handed between processes, if this state has one."""
    _, path = state_ref(value)
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def _state_on_disk(value) -> str:
    state, path = state_ref(value)
    if path and os.path.exists(path):
        return path
    if state is None:
        raise RuntimeError("The generation result is no longer available - please generate again.")
    return state_to_disk(state, _new_job_tmp("state.npz"))


def do_generate(job: JobState, *, isolate: bool, precision: str, image, multiimages,
                is_multiimage: bool, **kwargs):
    """Generate one asset. Returns (state-or-ref, video path)."""
    if not isolate:
        ENGINE.ensure(job, precision=precision)
        state, video_path = generate_one(job, image=image, multiimages=multiimages,
                                         is_multiimage=is_multiimage, **kwargs)
        return state, video_path

    # Everything below travels to a child process as JSON, so only plain data may be in
    # kwargs by this point - the images have already been written to disk.

    state_path = _new_job_tmp("state.npz")
    image_kwargs, stashed = _stash_images_for_worker(image, multiimages, is_multiimage)
    payload = {
        "precision": precision,
        "state_path": state_path,
        "kwargs": {**kwargs, **image_kwargs, "is_multiimage": bool(is_multiimage)},
    }
    try:
        result = run_isolated(job, "generate", payload)
    finally:
        # The child has read them by now (or died trying); either way they are ours to
        # remove, and at ~1 MB a run they add up over a batch.
        for path in stashed:
            try:
                os.remove(path)
            except OSError:
                pass
    ref = {"__state_path__": result["state_path"], "filename_base": result.get("filename_base"),
           "has_mesh": bool(result.get("has_mesh"))}
    return ref, result.get("video")


def result_has_mesh(value) -> bool:
    """Whether a generation result can produce a GLB (works for both state shapes)."""
    if not value:
        return False
    if isinstance(value, dict) and "__state_path__" in value:
        return bool(value.get("has_mesh"))
    return state_has_mesh(value)


def do_extract_glb(job: JobState, state_value, *, isolate: bool, precision: str,
                   mesh_simplify, texture_size, save_metadata, bake_resolution, bake_nviews) -> str:
    kwargs = {"mesh_simplify": float(mesh_simplify), "texture_size": int(texture_size),
              "save_metadata": bool(save_metadata), "bake_resolution": int(bake_resolution),
              "bake_nviews": int(bake_nviews)}
    if not isolate:
        ENGINE.ensure(job, precision=precision)
        return extract_glb_file(job, _materialise_state(state_value), **kwargs)
    result = run_isolated(job, "extract_glb", {
        "precision": precision, "state_path": _state_on_disk(state_value), "kwargs": kwargs})
    return result["glb"]


def do_extract_gaussian(job: JobState, state_value, *, isolate: bool, precision: str) -> str:
    if not isolate:
        ENGINE.ensure(job, precision=precision)
        return extract_gaussian_file(job, _materialise_state(state_value))
    result = run_isolated(job, "extract_gaussian", {
        "precision": precision, "state_path": _state_on_disk(state_value)})
    return result["gs"]


def run_generation(
    do_extract: bool,
    num_generations, image, multiimages, is_multiimage, image_is_clean,
    seed, randomize_seed,
    ss_guidance, ss_steps, slat_guidance, slat_steps, multiimage_algo,
    video_resolution, video_frames, video_fps, video_quality, include_geometry,
    save_metadata, mesh_simplify, texture_size,
    bake_resolution, bake_nviews, precision, isolate_jobs, generate_mesh,
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

    isolate = bool(isolate_jobs)
    generate_mesh = bool(generate_mesh)
    precision = precision if precision in ("fp16", "fp32") else cmd_args.precision
    # "Extract everything" means everything that exists: without a mesh there is no GLB to
    # build, so extract the Gaussian splat and say so rather than failing a minute later.
    want_glb = do_extract and generate_mesh
    extract_steps = (1 if want_glb else 0) + (1 if do_extract else 0)
    per_gen_stages = 4 + extract_steps
    job = JobState("Generation")
    # In isolated mode each operation spawns its own process, and each of those reports a
    # "Preparing the engine" stage of its own, so the plan has to make room for them.
    engine_stages = (1 + extract_steps) * num_gens if isolate else 1
    job.set_plan(engine_stages + per_gen_stages * num_gens)
    if do_extract and not generate_mesh:
        job.log("Gaussian-only mode: skipping GLB extraction (there is no mesh to build it "
                "from). The .ply and the preview video are still produced.", level="warn")
    job.set_headline(f"{num_gens} generation(s)" + (" + extraction" if do_extract else ""))
    register_job("generate", job)

    results: Dict[str, Any] = {"state": None, "video": None, "glb": None, "gs": None, "count": 0}

    def worker():
        _set_thread_job(job)
        if not isolate:
            job.stage("Preparing the engine")
            ENGINE.ensure(job, precision=precision)
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

                state, video_path = do_generate(
                    job, isolate=isolate, precision=precision,
                    image=image, multiimages=multiimages, is_multiimage=bool(is_multiimage),
                    image_is_clean=bool(image_is_clean), seed=seed_for_iter,
                    ss_guidance_strength=ss_guidance, ss_sampling_steps=ss_steps,
                    slat_guidance_strength=slat_guidance, slat_sampling_steps=slat_steps,
                    multiimage_algo=multiimage_algo,
                    video_resolution=video_resolution, video_num_frames=video_frames,
                    video_fps=video_fps, video_quality=video_quality,
                    include_geometry=bool(include_geometry),
                    save_metadata=bool(save_metadata),
                    generate_mesh=bool(generate_mesh),
                    output_filename_prefix=prefix,
                )
                # Only the newest run stays reachable from the UI, so an isolated run's
                # handoff file for the previous iteration is already garbage.
                if results["state"] is not None and results["state"] is not state:
                    _discard_state_file(results["state"])
                results["state"] = state
                results["video"] = video_path
                results["count"] = i + 1

                if want_glb:
                    job.raise_if_cancelled()
                    results["glb"] = do_extract_glb(
                        job, state, isolate=isolate, precision=precision,
                        mesh_simplify=mesh_simplify, texture_size=texture_size,
                        save_metadata=save_metadata, bake_resolution=bake_resolution,
                        bake_nviews=bake_nviews)
                if do_extract:
                    job.raise_if_cancelled()
                    results["gs"] = do_extract_gaussian(job, state, isolate=isolate,
                                                        precision=precision)

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
        job.log(f"FAILED: {explain_failure(error)}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"Generation failed: {explain_failure(error)}")
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
        # No mesh means no GLB, so leave that button disabled rather than letting the
        # click fail a minute later.
        gr.update(interactive=result_has_mesh(state)),
        gr.update(interactive=bool(state)),
        html,
        log_text,
    )


def run_extract_glb(state, mesh_simplify, texture_size, save_metadata,
                    bake_resolution, bake_nviews, precision, isolate_jobs):
    if not state:
        gr.Warning("Generate something first.")
        yield gr.skip(), gr.skip(), idle_status_html("Nothing to extract yet."), "No generation in memory."
        return

    isolate = bool(isolate_jobs)
    precision = precision if precision in ("fp16", "fp32") else cmd_args.precision
    job = JobState("GLB extraction")
    job.set_plan(2)
    register_job("generate", job)
    out: Dict[str, Any] = {}

    def worker():
        _set_thread_job(job)
        try:
            out["glb"] = do_extract_glb(
                job, state, isolate=isolate, precision=precision,
                mesh_simplify=mesh_simplify, texture_size=texture_size,
                save_metadata=save_metadata, bake_resolution=bake_resolution,
                bake_nviews=bake_nviews)
        finally:
            _set_thread_job(None)

    thread, box = _run_in_thread(worker, "trellis-glb")
    yield from _stream_status(job, thread, lambda html, log_text: (gr.skip(), gr.skip(), html, log_text))
    register_job("generate", None)

    error = box.get("error")
    if error is not None and not job.cancelled and not isinstance(error, JobCancelled):
        job.status = "error"
        job.log(f"FAILED: {explain_failure(error)}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"GLB extraction failed: {explain_failure(error)}")
    elif error is not None or job.cancelled:
        job.status = "cancelled"
    else:
        job.finish("GLB ready")

    html, log_text = job.snapshot()
    glb = out.get("glb")
    yield (glb if glb else gr.skip(),
           gr.update(value=glb, interactive=bool(glb)),
           html, log_text)


def run_extract_gaussian(state, precision, isolate_jobs):
    if not state:
        gr.Warning("Generate something first.")
        yield gr.skip(), gr.skip(), idle_status_html("Nothing to extract yet."), "No generation in memory."
        return

    isolate = bool(isolate_jobs)
    precision = precision if precision in ("fp16", "fp32") else cmd_args.precision
    job = JobState("Gaussian extraction")
    job.set_plan(2)
    register_job("generate", job)
    out: Dict[str, Any] = {}

    def worker():
        _set_thread_job(job)
        try:
            out["gs"] = do_extract_gaussian(job, state, isolate=isolate, precision=precision)
        finally:
            _set_thread_job(None)

    thread, box = _run_in_thread(worker, "trellis-gs")
    yield from _stream_status(job, thread, lambda html, log_text: (gr.skip(), gr.skip(), html, log_text))
    register_job("generate", None)

    error = box.get("error")
    if error is not None and not job.cancelled:
        job.status = "error"
        job.log(f"FAILED: {explain_failure(error)}", level="err")
        cprint(box.get("traceback", ""), "red")
        gr.Warning(f"Gaussian extraction failed: {explain_failure(error)}")
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
    include_geometry, save_metadata, bake_resolution, bake_nviews, precision, isolate_jobs,
    generate_mesh,
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

    isolate = bool(isolate_jobs)
    generate_mesh = bool(generate_mesh)
    precision = precision if precision in ("fp16", "fp32") else cmd_args.precision
    if extract_glb_cb and not generate_mesh:
        gr.Warning("Gaussian-only mode is on, so no GLB can be produced - unticking 'GLB mesh' "
                   "for this batch.")
        extract_glb_cb = False
    total_iterations = len(all_files) * num_gens_per_image
    # prepare image, sample, render video, write video (+ optional extractions)
    per_item_stages = 4 + (1 if extract_glb_cb else 0) + (1 if extract_gs_cb else 0)
    if isolate:
        # every isolated operation announces its own engine stage
        per_item_stages += 1 + (1 if extract_glb_cb else 0) + (1 if extract_gs_cb else 0)

    job = JobState("Batch")
    job.set_plan((0 if isolate else 1) + per_item_stages * total_iterations)
    job.set_headline(f"{len(all_files)} image(s) × {num_gens_per_image} generation(s)")
    register_job("batch", job)

    # Kept outside worker() so a cancel or a crash still reports what actually ran.
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    durations: List[float] = []

    def worker():
        _set_thread_job(job)
        try:
            if not isolate:
                job.stage("Preparing the engine")
                ENGINE.ensure(job, precision=precision)
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
                        # In isolated mode the worker opens the file itself, so hand it the
                        # path rather than decoding the image twice.
                        raw = image_path if isolate else ENGINE.Image.open(image_path).convert("RGBA")
                        state, video_path = do_generate(
                            job, isolate=isolate, precision=precision,
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
                            generate_mesh=bool(generate_mesh),
                            output_filename_prefix=output_prefix,
                            custom_output_dirs=custom_dirs,
                        )

                        try:
                            if extract_glb_cb:
                                job.raise_if_cancelled()
                                do_extract_glb(job, state, isolate=isolate, precision=precision,
                                               mesh_simplify=mesh_simplify, texture_size=texture_size,
                                               save_metadata=save_metadata,
                                               bake_resolution=bake_resolution,
                                               bake_nviews=bake_nviews)
                            if extract_gs_cb:
                                job.raise_if_cancelled()
                                do_extract_gaussian(job, state, isolate=isolate, precision=precision)
                        finally:
                            # A batch run never revisits an item, so the handoff file for an
                            # isolated generation would otherwise pile up on disk.
                            _discard_state_file(state)

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
                        job.log(f"ERROR on {output_prefix}: {explain_failure(exc)}", level="err")
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
        job.log(f"Batch failed: {explain_failure(error)}", level="err")
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
        ("Precision", f"{ENGINE.precision} (from the active preset; --precision "
                      f"{cmd_args.precision} at launch)"),
        ("High-VRAM mode", "on" if cmd_args.highvram else "off"),
        ("TF32", "off (--no-tf32)" if cmd_args.no_tf32 else "on"),
        ("Engine", ENGINE.status_line()),
        ("Detected GPU", f"{GPU_NAME} — {GPU_VRAM_GB:.1f} GB → {pick_vram_tier(GPU_VRAM_GB)} GB tier"
                         if GPU_VRAM_GB else "nvidia-smi reported nothing"),
    ]
    if ENGINE.torch is not None:
        torch = ENGINE.torch
        rows.append(("Torch", torch.__version__))
        if torch.cuda.is_available():
            rows.append(("GPU", f"{ENGINE.device_name} — {ENGINE.total_vram_gb:.1f} GB"))
            rows.append(("VRAM", ENGINE.vram_report()))
    rows.append(("Outputs folder", OUTPUT_DIR_BASE))
    rows.append(("Protected presets", f"{PRESET_DIR_BUILTIN} (rewritten every start)"))
    rows.append(("Your presets", f"{PRESET_DIR_USER} ({len(user_preset_names())} saved)"))

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
/* fill_width=True is what gives the two output columns room to breathe, but it also turns
   off Gradio's own centering: .fillable only gets its responsive max-width (and the auto
   margins that go with it) while :not(.fill_width). The cap below therefore has to bring
   its own centering, or the whole app sits flush against the left edge on any display
   wider than the cap. */
.gradio-container {
  max-width: 1680px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

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

/* ---------- VRAM preset panel ---------- */
#tp-vram-box {
  border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 10px 12px 4px 12px; margin-bottom: 10px;
  background: linear-gradient(140deg, rgba(79,70,229,.10) 0%, rgba(6,182,212,.08) 100%);
}
#tp-vram-box h3 { margin: 0 0 6px 0 !important; font-size: 1.0rem; }
#tp-vram-info > * { margin: 0 !important; }
.tp-vram {
  border-radius: 10px; padding: 10px 12px; font-size: .84rem;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
}
.tp-vram-gpu { font-size: .78rem; opacity: .75; margin-bottom: 4px; }
.tp-vram-peak { font-size: 1.02rem; font-weight: 700; margin-bottom: 6px; }
.tp-vram-peak span { font-weight: 500; opacity: .6; font-size: .84rem; }
.tp-vram-ok   .tp-vram-peak { color: #10b981; }
.tp-vram-warn .tp-vram-peak { color: #f43f5e; }
.tp-vram-body { line-height: 1.55; opacity: .9; }
.tp-vram-body code { font-size: .8rem; }
.tp-vram-note { margin-top: 6px; font-size: .79rem; opacity: .72; }
.tp-vram-warnline {
  margin-top: 8px; padding: 7px 9px; border-radius: 8px; font-size: .79rem;
  background: rgba(217,119,6,.14); border: 1px solid rgba(217,119,6,.4);
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
HEAD_HTML = r"""
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

(function () {
  // Brighten the GLB preview.
  //
  // gr.Model3D renders glTF through the Babylon viewer, whose default environment sits at
  // intensity 1.0 -- correct for a photographic asset, far too dim for our baked-albedo
  // meshes, which end up much darker than the colour half of the turntable clip. Babylon
  // exposes viewer.environmentConfig, but Gradio 6 offers no way to reach it from Python,
  // so patch the viewer prototype instead. Everything here is best-effort: if a Gradio or
  // Babylon upgrade moves any of it, the patch quietly does nothing and the preview simply
  // stays at stock brightness.
  var INTENSITY = 2.4;   // ~1.8x mean luminance, still under 1% clipped highlights
  var boosted = new WeakSet();
  var patched = false;

  function boost(viewer) {
    if (!viewer || boosted.has(viewer)) return;
    boosted.add(viewer);                       // before the write: setting the config marks
    try {                                      // the scene dirty, which re-enters the hook
      var cfg = viewer.environmentConfig;
      if (cfg && typeof cfg.intensity === "number") {
        viewer.environmentConfig = Object.assign({}, cfg, { intensity: INTENSITY });
      }
    } catch (e) {}
  }

  function patch(Viewer) {
    var proto = Viewer.prototype;

    // Normal path: every model Gradio shows arrives through loadModel.
    var load = proto.loadModel;
    if (typeof load === "function") {
      proto.loadModel = function () {
        var r = load.apply(this, arguments);
        var self = this;
        if (r && r.then) return r.then(function (v) { boost(self); return v; });
        boost(self);
        return r;
      };
    }

    // Fallback for a viewer built before this patch landed: _shouldRender is read once per
    // frame, so an in-flight load still gets caught on its next redraw.
    var d = Object.getOwnPropertyDescriptor(proto, "_shouldRender");
    if (d && d.get && d.configurable) {
      var get = d.get;
      Object.defineProperty(proto, "_shouldRender", {
        configurable: true,
        enumerable: d.enumerable,
        get: function () { boost(this); return get.call(this); },
      });
    }
  }

  // The Babylon bundle is a hashed, lazily imported chunk, so it can only be found once the
  // browser has fetched it. Resource timings list it only after that, which means import()
  // here always hits the module cache rather than costing another download. Poll often: the
  // window between that fetch and the first loadModel is short, and losing the race only
  // falls back to the per-frame hook.
  var tried = {};
  var timer = setInterval(look, 150);

  function look() {
    if (patched) { clearInterval(timer); return; }
    performance.getEntriesByType("resource").forEach(function (entry) {
      var url = entry.name;
      if (tried[url] || !/\/assets\/index-[^/]*\.js$/.test(url)) return;
      tried[url] = true;
      import(url).then(function (mod) {
        if (patched || !mod || !mod.cx || typeof mod.cx.Viewer !== "function") return;
        patched = true;
        patch(mod.cx.Viewer);
      }).catch(function () {});
    });
  }
  document.addEventListener("DOMContentLoaded", look);
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
    <span class="tp-chip">{'high-VRAM' if cmd_args.highvram else 'model offloading'}</span>
    <span class="tp-chip">{(GPU_NAME + f' · {GPU_VRAM_GB:.1f} GB') if GPU_VRAM_GB else 'no GPU detected'}</span>
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

                        # ---------------- VRAM presets (above Generation settings) ----
                        with gr.Group(elem_id="tp-vram-box"):
                            gr.Markdown("### 🎛️  VRAM preset")
                            vram_tier_radio = gr.Radio(
                                choices=[(f"{t} GB", t) for t in VRAM_TIERS],
                                value=pick_vram_tier(GPU_VRAM_GB),
                                label="Pick the tier that matches your GPU",
                                info="Auto-selected from the detected card on start. Every tier "
                                     "sets precision, sampling steps, video and extraction "
                                     "quality together.",
                            )
                            vram_info = gr.HTML(vram_info_html(vram_preset_name(pick_vram_tier(GPU_VRAM_GB))),
                                                elem_id="tp-vram-info")
                            with gr.Row(elem_classes=["tp-actions"]):
                                precision_radio = gr.Radio(
                                    ["fp32", "fp16"], value=cmd_args.precision, label="Model precision",
                                    info="fp16 roughly halves the mesh decoder's peak (12.1 → 7.7 GB). "
                                         "Changing it rebuilds the pipeline once.")
                                isolate_jobs_checkbox = gr.Checkbox(
                                    label="Run each job in its own process", value=False,
                                    info="Frees the CUDA context (1.2–1.7 GB) between runs instead of "
                                         "just the tensor cache. Costs one model load per job.")
                            generate_mesh_checkbox = gr.Checkbox(
                                label="Also decode a mesh (required for GLB)", value=True,
                                info="The mesh decoder alone peaks at ~7.6 GB in fp16 / ~12.1 GB in "
                                     "fp32 and ignores every other setting. Untick it and you get "
                                     "the Gaussian .ply plus the video for under 5 GB — but no GLB.")

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
                            with gr.Row():
                                bake_resolution_slider = gr.Slider(
                                    256, 1024, label="Texture-bake view size (px)", value=1024, step=128,
                                    info="Views are held on the GPU while baking — this is the "
                                         "biggest VRAM term of GLB extraction.")
                                bake_nviews_slider = gr.Slider(
                                    20, 200, label="Texture-bake views", value=100, step=10,
                                    info="Fewer views = less VRAM, but unseen patches may stay blank.")
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
                            "A preset stores every slider, checkbox and batch path in this app.\n\n"
                            "* **🔒 Protected** — `Default` and the seven `VRAM … GB` tiers. They live in "
                            "`configs_trellis/presets_builtin/` and are rewritten from code on every "
                            "start, so they can never be overwritten, deleted or corrupted.\n"
                            "* **Yours** — everything you save, in `configs_trellis/presets_user/`. "
                            "The app never touches this folder.\n\n"
                            "On start-up the app loads **your** last-used preset if you have one; "
                            "otherwise it picks the VRAM tier that matches the detected GPU."
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
                        with gr.Row(elem_classes=["tp-actions"]):
                            user_folder_button = gr.Button("📂  Open my presets folder",
                                                           elem_classes=["tb-cyan"])
                            config_folder_button = gr.Button("🔒  Open protected folder",
                                                             elem_classes=["tb-slate"])
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### What each VRAM tier costs\n"
                            "Peak VRAM is measured on a real run (device occupancy sampled every "
                            "15 ms, so the CUDA context is included), not estimated.\n\n"
                            + "| Tier | Precision | Mesh | Peak VRAM | Video | Texture |\n"
                              "|---|---|---|---|---|---|\n"
                            + "\n".join(
                                f"| **{t} GB** | `{VRAM_PRESET_SPECS[t]['values']['precision_val']}` "
                                f"| {'yes' if VRAM_PRESET_SPECS[t]['values']['generate_mesh_val'] else '—'} "
                                f"| {VRAM_PRESET_SPECS[t]['peak_gb']:.1f}–"
                                f"{VRAM_PRESET_SPECS[t].get('peak_max_gb', VRAM_PRESET_SPECS[t]['peak_gb']):.1f} GB "
                                f"| {VRAM_PRESET_SPECS[t]['values']['video_resolution_val']}px / "
                                f"{VRAM_PRESET_SPECS[t]['values']['video_num_frames_val']}f "
                                f"| {VRAM_PRESET_SPECS[t]['values']['texture_size_val']}px |"
                                for t in VRAM_TIERS)
                            + "\n\nThe range is typical subject → busiest subject: the mesh decoder's "
                              "cost scales with how much of the volume the object fills."
                        )
                        gr.Markdown(
                            "### Quality cheat-sheet\n"
                            "| Goal | What to change |\n|---|---|\n"
                            "| Sharper geometry | Sparse-structure steps 20-25, guidance 7.5 |\n"
                            "| Cleaner texture | Texture size 2048, simplification 0.7-0.8 |\n"
                            "| Faster iteration | 10-12 steps, video 512 px / 120 frames, geometry pass off |\n"
                            "| Lower VRAM | fp16 precision — it is worth more than every other knob combined |\n"
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
            bake_resolution_slider, bake_nviews_slider,
            video_resolution_slider, video_num_frames_slider, video_fps_slider,
            video_quality_slider, include_geometry_checkbox,
            save_metadata_checkbox,
            precision_radio, isolate_jobs_checkbox, generate_mesh_checkbox,
            batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
            batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox,
        ]
        assert len(preset_ui_components) == len(PRESET_KEYS), (
            f"preset component/key mismatch: {len(preset_ui_components)} components "
            f"vs {len(PRESET_KEYS)} keys")

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
            bake_resolution_slider, bake_nviews_slider, precision_radio, isolate_jobs_checkbox,
            generate_mesh_checkbox,
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
            inputs=[output_buf, mesh_simplify_slider, texture_size_slider, save_metadata_checkbox,
                    bake_resolution_slider, bake_nviews_slider, precision_radio,
                    isolate_jobs_checkbox],
            outputs=[model_output, download_glb, gen_status, gen_log],
            show_progress="hidden",
        )
        extract_gs_btn.click(
            run_extract_gaussian, inputs=[output_buf, precision_radio, isolate_jobs_checkbox],
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
                bake_resolution_slider, bake_nviews_slider, precision_radio,
                isolate_jobs_checkbox, generate_mesh_checkbox,
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

        # -- VRAM presets
        # The radio owns the whole control panel: picking a tier writes every slider and
        # also selects the matching protected preset in the Presets tab, so the two views
        # never disagree about what is currently loaded.
        vram_tier_radio.change(
            apply_vram_preset, inputs=[vram_tier_radio],
            outputs=[config_status_textbox, config_load_dropdown, vram_info] + preset_ui_components,
            **LIGHT)

        # -- presets
        def _load_config_and_info(name):
            values = load_config(name)
            return (values[0], vram_info_html(name)) + tuple(values[1:])

        preset_outputs = [config_status_textbox, vram_info] + preset_ui_components

        config_save_button.click(save_config,
                                 inputs=[config_save_name_textbox] + preset_ui_components,
                                 outputs=[config_status_textbox, config_load_dropdown], **LIGHT)
        config_load_button.click(_load_config_and_info, inputs=[config_load_dropdown],
                                 outputs=preset_outputs, **LIGHT)
        config_load_dropdown.change(_load_config_and_info, inputs=[config_load_dropdown],
                                    outputs=preset_outputs, **LIGHT)
        config_delete_button.click(delete_config, inputs=[config_load_dropdown],
                                   outputs=[config_status_textbox, config_load_dropdown], **LIGHT)
        config_reset_button.click(lambda: (reset_to_defaults()[0], vram_info_html(DEFAULT_CONFIG_NAME))
                                  + tuple(reset_to_defaults()[1:]),
                                  inputs=None, outputs=preset_outputs, **LIGHT)
        config_folder_button.click(lambda: open_folder(CONFIG_DIR), outputs=None, **LIGHT)
        user_folder_button.click(lambda: open_folder(PRESET_DIR_USER), outputs=None, **LIGHT)

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
                  outputs=[config_load_dropdown, vram_tier_radio, vram_info] + preset_ui_components,
                  **LIGHT)

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
    write_builtin_presets()
    sweep_job_tmp()
    startup_name, startup_why = startup_preset_name()

    banner(f"SECourses TRELLIS Studio {APP_VERSION}")
    cprint(f"  Attention backend : {ATTENTION_BACKEND} ({_ATTENTION_NOTE})", "cyan")
    if GPU_VRAM_GB:
        tier = pick_vram_tier(GPU_VRAM_GB)
        cprint(f"  GPU               : {GPU_NAME} ({GPU_VRAM_GB:.1f} GB) -> {tier} GB VRAM tier "
               f"(±{VRAM_TIER_MARGIN_GB} GB margin)", "cyan")
        cprint(f"  Expected peak     : ~{VRAM_PRESET_SPECS[tier]['peak_gb']:.1f} GB", "cyan")
    else:
        cprint("  GPU               : not detected via nvidia-smi", "yellow")
    cprint(f"  Startup preset    : {startup_name} ({startup_why})", "cyan")
    cprint(f"  Launch precision  : {cmd_args.precision} (presets can change this per run)", "cyan")
    cprint(f"  High-VRAM mode    : {'on' if cmd_args.highvram else 'off (models offload to RAM)'}", "cyan")
    cprint(f"  Outputs           : {OUTPUT_DIR_BASE}", "cyan")
    cprint(f"  Presets           : {PRESET_DIR_BUILTIN} (protected)", "dim")
    cprint(f"                      {PRESET_DIR_USER} (yours)", "dim")
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
    if cmd_args.worker:
        # Child process: do one job, write the result, exit. Exiting is the point - it is
        # what hands the CUDA context back to the driver.
        sys.exit(worker_main(cmd_args.worker))
    main()
