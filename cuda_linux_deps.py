"""Linux-only CUDA shared-library plumbing for the bundled CUDA extensions.

Why this file exists
--------------------
On Windows, PyTorch ships the entire CUDA runtime *inside* ``torch\\lib``
(``nvrtc64_130_0.dll``, ``nvrtc-builtins64_130.dll``, ``cudart64_13.dll``, ...) and
registers that folder with :func:`os.add_dll_directory` when it is imported.  Every
CUDA extension loaded afterwards - cumm, spconv, flash-attn, kaolin - finds its DLLs
for free.  That is why ``Pip_Freeze.txt`` on Windows contains no ``nvidia-*`` packages
at all.

The Linux wheels do not work that way.  ``torch`` depends on separate ``nvidia-*-cu13``
distributions that unpack into ``site-packages/nvidia/<component>/lib``, and torch
pre-loads only the handful of libraries *it* needs, by absolute path.  Nothing puts
that directory on the dynamic loader's search path, and ``LD_LIBRARY_PATH`` is read
once when the process starts, so Python cannot repair it from the inside.

The gap that actually bites is NVRTC.  torch globs ``libnvrtc.so.*[0-9]``, which never
matches the ``libnvrtc-builtins.so.<major>.<minor>`` sibling that NVRTC pulls in - and
the soname carries the exact CUDA *minor* version, so ``.so.13.0`` and ``.so.13.1`` are
different files.  A prebuilt ``cumm``/``spconv`` linked on a CUDA 13.1 box therefore
dies on a CUDA 13.0 install with::

    ImportError: libnvrtc-builtins.so.13.1: cannot open shared object file

:func:`preload` fixes that the same way torch fixes it for itself: ``dlopen`` the
libraries with ``RTLD_GLOBAL`` *before* the extension is imported, which puts their
sonames into the global symbol namespace so the extension's own ``dlopen`` resolves
against them.  It searches every plausible location and globs versions, so it does not
care which CUDA minor the wheels were built against.

:func:`repair` goes one step further - it imports the extensions in a child process,
reads the missing soname straight out of the ``ImportError`` and pip-installs the
matching ``nvidia-*`` wheel.  That is for the install scripts, not for app start-up.

Everything here is inert on Windows and macOS: :func:`preload` returns immediately and
:func:`repair` reports that there is nothing to do.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Libraries the loader has to resolve on its own.  All of them are small, so pre-loading
# costs nothing measurable; the big ones (cudnn, cublas) are left alone because torch
# already pre-loads those itself.
_PRELOAD_GLOBS = (
    "libnvrtc-builtins.so*",
    "libnvrtc.so*",
    "libnvJitLink.so*",
    "libcudart.so*",
)

# Soname stem -> pip distribution, minus the "-cu<major>" suffix that gets appended from
# the soname's own version.  Used by repair() to install a library that is missing
# outright, which no amount of search-path fixing can solve.
_PIP_DISTRIBUTION = {
    "libcublas": "nvidia-cublas",
    "libcublasLt": "nvidia-cublas",
    "libcudart": "nvidia-cuda-runtime",
    "libcudnn": "nvidia-cudnn",
    "libcufft": "nvidia-cufft",
    "libcurand": "nvidia-curand",
    "libcusolver": "nvidia-cusolver",
    "libcusparse": "nvidia-cusparse",
    "libnccl": "nvidia-nccl",
    "libnvJitLink": "nvidia-nvjitlink",
    "libnvrtc": "nvidia-cuda-nvrtc",
    "libnvrtc-builtins": "nvidia-cuda-nvrtc",
}

# "libnvrtc-builtins.so.13.1: cannot open shared object file: No such file or directory"
_MISSING_SO_RE = re.compile(r"(?P<stem>lib[A-Za-z0-9_+-]+)\.so\.(?P<version>\d+(?:\.\d+)*)")

# Probed in this order.  `import cumm` is what actually blows up and it does not drag
# torch in, so the probe stays fast.
_PROBE_MODULES = ("cumm", "spconv")


# --------------------------------------------------------------------------------------
# Search paths
# --------------------------------------------------------------------------------------
def candidate_dirs() -> list[str]:
    """Directories that may hold CUDA runtime libraries, in loader-preference order."""
    import importlib.util

    dirs: list[str] = []

    def add(path: str) -> None:
        if path and path not in dirs and os.path.isdir(path):
            dirs.append(path)

    def search_locations(module: str) -> list[str]:
        # find_spec locates a package without executing it.  Importing torch here would
        # drag the slow CUDA init into start-up, which the app deliberately defers.
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            return []
        return list(getattr(spec, "submodule_search_locations", None) or [])

    # The nvidia-*-cu1X wheels torch depends on.  `nvidia` is a namespace package, so its
    # search locations are the site-packages roots that contain the components.
    for root in search_locations("nvidia"):
        for lib in sorted(glob.glob(os.path.join(root, "*", "lib"))):
            add(lib)

    for root in search_locations("torch"):
        add(os.path.join(root, "lib"))

    # A system toolkit, if one happens to be installed.  Searched last so the wheels that
    # torch was built against always win.
    for env in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        base = os.environ.get(env)
        if base:
            add(os.path.join(base, "lib64"))
            add(os.path.join(base, "lib"))
    for base in ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        add(os.path.join(base, "lib64"))
        for lib in sorted(glob.glob(os.path.join(base, "targets", "*", "lib"))):
            add(lib)

    return dirs


# --------------------------------------------------------------------------------------
# The actual fix
# --------------------------------------------------------------------------------------
def preload() -> list[str]:
    """dlopen the CUDA libraries with RTLD_GLOBAL so later extension imports resolve.

    Returns the files that were loaded.  Cheap and side-effect-free to call on any
    platform - on anything other than Linux it returns an empty list immediately, since
    Windows and macOS resolve these through mechanisms that already work.
    """
    if sys.platform != "linux":
        return []

    import ctypes

    loaded: list[str] = []
    seen: set[str] = set()
    for directory in candidate_dirs():
        for pattern in _PRELOAD_GLOBS:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                soname = os.path.basename(path)
                if soname in seen:
                    continue                  # first directory wins, as the loader would
                seen.add(soname)
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue                  # wrong arch, dangling symlink - skip it
                loaded.append(path)
    return loaded


def missing_library_hint(error: str) -> str:
    """A short, actionable message for a "cannot open shared object file" ImportError.

    Returns "" when the error is something else, so callers can append unconditionally.
    """
    if sys.platform != "linux" or "cannot open shared object file" not in error:
        return ""
    match = _MISSING_SO_RE.search(error)
    soname = match.group(0) if match else "a CUDA library"
    return (f"{soname} is missing from this environment. Run "
            f"`python {os.path.join(HERE, 'cuda_linux_deps.py')}` to install the matching "
            f"NVIDIA wheel, then start the app again.")


# --------------------------------------------------------------------------------------
# Self-repair - used by the install scripts
# --------------------------------------------------------------------------------------
_PROBE_SOURCE = """\
import importlib.util, sys
sys.path.insert(0, {here!r})
import cuda_linux_deps
cuda_linux_deps.preload()
for name in {modules!r}:
    if importlib.util.find_spec(name) is None:
        continue
    try:
        __import__(name)
    except ImportError as exc:
        print("MISSING\\t%s\\t%s" % (name, exc))
        raise SystemExit(0)
print("OK")
"""


def _probe() -> tuple[str, str] | None:
    """Import the CUDA extensions in a child process.

    Returns ``(module, error)`` for the first one that fails, or None if they all import.
    A child process is used because a failed dlopen cannot be meaningfully retried inside
    the process that already attempted it.
    """
    source = _PROBE_SOURCE.format(here=HERE, modules=list(_PROBE_MODULES))
    proc = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("MISSING\t"):
            _, module, error = line.split("\t", 2)
            return module, error
        if line.strip() == "OK":
            return None
    return "probe", (proc.stderr.strip() or "the import probe produced no output")


def _pip_target(error: str) -> str | None:
    """Map a missing soname to the pip requirement that provides it."""
    match = _MISSING_SO_RE.search(error)
    if not match:
        return None
    distribution = _PIP_DISTRIBUTION.get(match.group("stem"))
    if distribution is None:
        return None
    parts = match.group("version").split(".")
    major = parts[0]
    # libfoo.so.13.1 names the exact CUDA minor release; libfoo.so.13 only the major.
    if len(parts) >= 2:
        return f"{distribution}-cu{major}=={major}.{parts[1]}.*"
    return f"{distribution}-cu{major}"


def _install(requirement: str) -> bool:
    """Install one NVIDIA wheel over whatever torch pinned.

    --no-deps is deliberate: torch pins the nvidia-* wheels exactly, so a normal install
    would be rejected as a version conflict.  NVRTC and friends are backward compatible
    within a CUDA major version, so torch keeps working against the newer one.
    """
    for command in ([sys.executable, "-m", "uv", "pip", "install", "--no-deps", requirement],
                    [sys.executable, "-m", "pip", "install", "--no-deps", requirement]):
        print("  $ " + " ".join(command), flush=True)
        try:
            if subprocess.run(command).returncode == 0:
                return True
        except OSError:
            continue
    return False


def _report_environment() -> None:
    print("\nSearched these directories:")
    for directory in candidate_dirs():
        found = sorted(os.path.basename(p)
                       for pattern in _PRELOAD_GLOBS
                       for p in glob.glob(os.path.join(directory, pattern)))
        print(f"  {directory}")
        if found:
            print("      " + ", ".join(found))


def repair(max_rounds: int = 4) -> int:
    """Import the CUDA extensions, installing whatever library they turn out to need.

    Returns a process exit code: 0 when the extensions import, 1 otherwise.
    """
    if sys.platform != "linux":
        print("Not Linux - torch bundles the CUDA runtime here, nothing to do.")
        return 0

    attempted: set[str] = set()
    for _ in range(max_rounds):
        failure = _probe()
        if failure is None:
            print("CUDA extensions import cleanly.")
            return 0

        module, error = failure
        print(f"{module} cannot load: {error}")

        requirement = _pip_target(error)
        if requirement is None or requirement in attempted:
            break
        attempted.add(requirement)

        print(f"Installing {requirement} to provide it ...", flush=True)
        if not _install(requirement):
            print(f"Could not install {requirement}.")
            break

    _report_environment()
    print("\nThe CUDA extensions still do not import. Please report the output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(repair())
