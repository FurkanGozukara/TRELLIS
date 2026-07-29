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

# The one library the loader can genuinely fail to find.  Every other CUDA soname carries
# only the major version, so torch's own pre-loading and the wheels' RUNPATH both resolve
# them.  `libnvrtc-builtins` is the sole exception - its soname carries major.minor and
# nothing pre-loads it by name.
#
# Keeping the list to this single 6 MB file is deliberate: `libnvrtc.so.13` alone is a
# 114 MB mapping, and secourses_trellis.py re-executes this module in every per-job worker
# subprocess, so anything listed here is paid for again on every generation.
_PRELOAD_GLOBS = ("libnvrtc-builtins.so*",)

# Wider net, used only when printing a diagnostic after something has already failed.
_REPORT_GLOBS = ("libnvrtc*.so*", "libcudart.so*", "libnvJitLink.so*", "libcublas*.so*")

# Soname stem -> (distributions to try in order, may the soname's version pin the wheel?)
#
# NVIDIA changed the scheme at CUDA 13.  The 12.x wheels are "nvidia-<part>-cu12" and
# unpack into "nvidia/<part>/lib"; the 13.x wheels dropped the suffix entirely and share
# one "nvidia/cu13/lib" directory.  The "-cu13" names still exist on PyPI but are
# placeholders stuck at 0.0.1 with no Linux wheel, so a version-pinned install of one
# just fails and falls through to the next candidate - no version sniffing needed.
#
# cuDNN and NCCL version independently of CUDA, so their soname major says nothing about
# the wheel version; those are installed unpinned, newest CUDA generation first.
_PIP_CANDIDATES = {
    "libcublas": (("nvidia-cublas", "nvidia-cublas-cu{major}"), True),
    "libcublasLt": (("nvidia-cublas", "nvidia-cublas-cu{major}"), True),
    "libcudart": (("nvidia-cuda-runtime", "nvidia-cuda-runtime-cu{major}"), True),
    "libcufft": (("nvidia-cufft", "nvidia-cufft-cu{major}"), True),
    "libcurand": (("nvidia-curand", "nvidia-curand-cu{major}"), True),
    "libcusolver": (("nvidia-cusolver", "nvidia-cusolver-cu{major}"), True),
    "libcusparse": (("nvidia-cusparse", "nvidia-cusparse-cu{major}"), True),
    "libnvJitLink": (("nvidia-nvjitlink", "nvidia-nvjitlink-cu{major}"), True),
    "libnvrtc": (("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu{major}"), True),
    "libnvrtc-builtins": (("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu{major}"), True),
    "libcudnn": (("nvidia-cudnn-cu13", "nvidia-cudnn-cu12"), False),
    "libnccl": (("nvidia-nccl-cu13", "nvidia-nccl-cu12"), False),
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
# Wheel audit - catches a stale cached wheel, the one failure preload() cannot paper over
# --------------------------------------------------------------------------------------
# A DT_NEEDED whose soname carries a minor (libnvrtc-builtins.so.13.1) welds the build
# box's exact CUDA minor into the wheel and is the bug documented in
# BUILD_NOTES_CUDA_WHEELS.md.  Correct wheels reference only major-versioned sonames.
_MINOR_VERSIONED_RE = re.compile(r"^lib[A-Za-z0-9_+-]+\.so\.\d+\.\d+")


def elf_needed(path: str) -> list[str]:
    """Return the DT_NEEDED entries of an ELF64 shared object; [] if it is not one.

    Hand-rolled rather than shelling out to readelf, because binutils is frequently absent
    from slim container images - and this has to work on exactly those.
    """
    import struct

    try:
        with open(path, "rb") as fh:
            header = fh.read(64)
            if header[:4] != b"\x7fELF" or header[4] != 2:      # not ELF, or not 64-bit
                return []
            e_phoff, = struct.unpack_from("<Q", header, 32)
            e_phentsize, e_phnum = struct.unpack_from("<HH", header, 54)

            fh.seek(e_phoff)
            phdrs = fh.read(e_phentsize * e_phnum)

            loads, dynamic = [], None
            for index in range(e_phnum):
                base = index * e_phentsize
                p_type, = struct.unpack_from("<I", phdrs, base)
                p_offset, p_vaddr = struct.unpack_from("<QQ", phdrs, base + 8)
                p_filesz, = struct.unpack_from("<Q", phdrs, base + 32)
                if p_type == 1:                                 # PT_LOAD
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:                               # PT_DYNAMIC
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return []

            fh.seek(dynamic[0])
            entries = fh.read(dynamic[1])

            needed, strtab_vaddr, strsz = [], None, 0
            for offset in range(0, len(entries) - 15, 16):
                d_tag, d_val = struct.unpack_from("<qQ", entries, offset)
                if d_tag == 0:                                  # DT_NULL, end of table
                    break
                if d_tag == 1:                                  # DT_NEEDED, strtab index
                    needed.append(d_val)
                elif d_tag == 5:                                # DT_STRTAB, virtual addr
                    strtab_vaddr = d_val
                elif d_tag == 10:                               # DT_STRSZ
                    strsz = d_val
            if strtab_vaddr is None or not needed:
                return []

            # DT_STRTAB is a virtual address; map it back through the PT_LOAD segments.
            file_offset = None
            for p_vaddr, p_offset, p_filesz in loads:
                if p_vaddr <= strtab_vaddr < p_vaddr + p_filesz:
                    file_offset = strtab_vaddr - p_vaddr + p_offset
                    break
            if file_offset is None:
                return []

            fh.seek(file_offset)
            strtab = fh.read(strsz or 65536)
    except OSError:
        return []

    names = []
    for index in needed:
        end = strtab.find(b"\0", index)
        if end > index:
            names.append(strtab[index:end].decode("utf-8", "replace"))
    return names


def audit_installed() -> dict[str, list[str]]:
    """Installed extension .so files that hard-link a minor-versioned CUDA soname.

    Returns ``{path: [offending sonames]}``; an empty dict means the wheels are clean.

    This is what detects a stale wheel served from the uv/pip cache.  The rebuilt cumm was
    re-uploaded under its original filename, so a machine that installed the earlier build
    already has that filename cached and will happily reuse it - the install log looks
    identical, and only the binary differs.
    """
    import importlib.util

    findings: dict[str, list[str]] = {}
    for package in _PROBE_MODULES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            continue
        for root in list(getattr(spec, "submodule_search_locations", None) or []):
            for so in sorted(glob.glob(os.path.join(root, "*.so"))):
                offenders = [n for n in elf_needed(so) if _MINOR_VERSIONED_RE.match(n)]
                if offenders:
                    findings[so] = offenders
    return findings


_STALE_WHEEL_ADVICE = """\
That is the old wheel.  It was re-uploaded under the same filename, so pip/uv will reuse
the cached copy unless told otherwise.  Force a fresh download:

    uv cache clean cumm spconv
    uv pip install -r requirements_trellis.txt --index-strategy unsafe-best-match \\
        --reinstall-package cumm --reinstall-package spconv
"""


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


def _pip_targets(error: str) -> list[str]:
    """Map a missing soname to the pip requirements that might provide it, best first."""
    match = _MISSING_SO_RE.search(error)
    if not match:
        return []
    entry = _PIP_CANDIDATES.get(match.group("stem"))
    if entry is None:
        return []
    distributions, versioned = entry

    parts = match.group("version").split(".")
    major = parts[0]
    # libfoo.so.13.3 names an exact release; libfoo.so.13 only constrains the major.
    version = f"=={major}.{parts[1]}.*" if len(parts) >= 2 else f"=={major}.*"

    return [d.format(major=major) + (version if versioned else "") for d in distributions]


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
                       for pattern in _REPORT_GLOBS
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

    stale = audit_installed()
    if stale:
        print("These installed binaries hard-link an exact CUDA minor version:")
        for path, sonames in stale.items():
            print(f"  {path}")
            print(f"      {', '.join(sonames)}")
        print("\n" + _STALE_WHEEL_ADVICE)
    else:
        print("Installed CUDA extensions link only major-versioned sonames - good.")

    attempted: set[str] = set()
    for _ in range(max_rounds):
        failure = _probe()
        if failure is None:
            print("CUDA extensions import cleanly.")
            return 0

        module, error = failure
        print(f"{module} cannot load: {error}")

        candidates = [r for r in _pip_targets(error) if r not in attempted]
        if not candidates:
            break
        attempted.update(candidates)

        for requirement in candidates:
            print(f"Installing {requirement} to provide it ...", flush=True)
            if _install(requirement):
                break
        else:
            print("None of " + ", ".join(candidates) + " could be installed.")
            break

    _report_environment()
    print("\nThe CUDA extensions still do not import. Please report the output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(repair())
