# Build notes for the in-house CUDA wheels (read before rebuilding cumm / spconv)

Scope: the Linux `cumm` and `spconv` wheels pinned in `../requirements_trellis.txt`.
Companion runtime code: `cuda_linux_deps.py` in this folder.

---

## 0. STATUS - the rebuilt `cumm` wheel is verified fixed

The wheel now at the pinned URL (`cumm-0.8.2.post1+cu130.torch2.13.gd971d02...`) was
re-downloaded and every `core_cc_py3XX.so` in it re-read. It is correct:

```
cumm/core_cc_py310.so … py313.so    (2.8 MB each)
    RUNPATH: $ORIGIN/../nvidia/cu13/lib
    NEEDED : libcudart.so.13, libnvrtc.so.13, libstdc++.so.6, libgcc_s.so.1,
             libc.so.6, ld-linux-x86-64.so.2
```

Two changes versus the broken build:

1. **The `libnvrtc-builtins.so.13.1` `DT_NEEDED` is gone.** Only major-versioned CUDA
   sonames remain, so no CUDA 13.x minor can ever fail to resolve.
2. **A `RUNPATH` was added** - and it is right. The full chain now closes on its own:

   ```
   cumm/core_cc_py312.so
     --RUNPATH $ORIGIN/../nvidia/cu13/lib-->  nvidia/cu13/lib/libnvrtc.so.13
       --dlopen, RUNPATH $ORIGIN----------->  nvidia/cu13/lib/libnvrtc-builtins.so.13.<own minor>
   ```

   Verified against `nvidia_cuda_nvrtc-13.3.33-...manylinux_2_12_x86_64.whl`:
   `libnvrtc.so.13` carries `RUNPATH: $ORIGIN` and does **not** list the builtins in
   `DT_NEEDED` at all - it `dlopen`s them from its own directory at run time, so the
   builtins always match the NVRTC that loaded them. This is the mechanism section 1
   predicted, now confirmed on the actual binary.

`spconv` was correctly left alone: its `core_cc_py3XX.so` still link `libcudart.so.13`
only, and they picked up the same `RUNPATH`.

Consequence: `cuda_linux_deps.preload()` is **no longer load-bearing**. Keep it as
redundancy - it costs a few milliseconds and protects against a regressed wheel or a
stale cached one - but the wheels resolve without it now. It was narrowed to pre-load
only `libnvrtc-builtins.so*` (6 MB), because `secourses_trellis.py` re-executes it in
every per-job worker subprocess and `libnvrtc.so.13` alone is a 114 MB mapping.

### The one thing left that can still bite: the cache

The rebuilt wheels were re-uploaded to the **same filenames**. pip and uv key their
caches on that filename, so a machine that installed the earlier build will reuse the
broken binary and the install log will look completely normal. Both install scripts now
do:

```bash
uv cache clean cumm spconv || true
uv pip install -r requirements_trellis.txt --index-strategy unsafe-best-match \
    --reinstall-package cumm --reinstall-package spconv
```

and `cuda_linux_deps.py` then runs `audit_installed()`, which reads the `DT_NEEDED` of
every installed `cumm`/`spconv` `.so` with a built-in ELF parser (no binutils needed) and
names any minor-versioned soname it finds. That is section 5's preflight, run on the
deployment box instead of the build box. A clean install prints:

```
Installed CUDA extensions link only major-versioned sonames - good.
CUDA extensions import cleanly.
```

---

## 1. The mistake that must not be repeated

A `cumm` wheel built on a box with the **CUDA 13.1** toolkit, shipped next to
**torch 2.13.0+cu130**, dies at import with:

```
ImportError: libnvrtc-builtins.so.13.1: cannot open shared object file: No such file or directory
  ... venv/lib/python3.12/site-packages/cumm/__init__.py line 42, in <module>
      core_cc = importlib.util.module_from_spec(_native_spec)
```

### Why

Every other CUDA soname carries only the **major** version, so it survives a minor
bump. `libnvrtc-builtins` is the one exception - its soname carries **major.minor**:

| library             | soname on disk                | minor in soname? |
| ------------------- | ----------------------------- | ---------------- |
| `libcudart`         | `libcudart.so.13`             | no               |
| `libnvrtc`          | `libnvrtc.so.13`              | no               |
| `libnvrtc-builtins` | `libnvrtc-builtins.so.13.1`   | **YES**          |

`torch 2.13.0+cu130` pins an NVRTC wheel from a **different 13.x minor** than the build
box had, and that wheel only ships its own `libnvrtc-builtins.so.13.<its minor>`. A wheel
that hard-links `.so.13.1` can never resolve against it.

> **CUDA 13 renamed and relocated these wheels - do not copy the CUDA 12 spelling.**
> Checked against PyPI:
>
> | | CUDA 12 | CUDA 13 |
> | --- | --- | --- |
> | distribution | `nvidia-cuda-nvrtc-cu12` | `nvidia-cuda-nvrtc` (**no suffix**) |
> | install path | `nvidia/cuda_nvrtc/lib/` | `nvidia/cu13/lib/` (shared by all parts) |
>
> `nvidia-cuda-nvrtc-cu13` **does** exist on PyPI but is a placeholder pinned at `0.0.1`
> with no Linux wheel - installing it gets you nothing. The same holds for
> `nvidia-cuda-runtime-cu13`. `cuDNN` and `NCCL` are the exception: they version
> independently of CUDA and still use `nvidia-cudnn-cu13` / `nvidia-nccl-cu13`.
>
> `cuda_linux_deps.py` handles this by trying the unsuffixed name first and the
> `-cu<major>` name second, so it works on both generations without sniffing.

### Where it comes from

`cumm/common.py`, class `CummNVRTCLink` (around line 470), forces the dependency in:

```python
nvrtc_builtins_lib_name = get_cuda_lib_link_name_linux(lib, "nvrtc-builtins")
...
self.build_meta.add_ldflags("g++",    "-Wl,--no-as-needed", f"-l{nvrtc_builtins_lib_name}")
self.build_meta.add_ldflags("clang++","-Wl,--no-as-needed", f"-l{nvrtc_builtins_lib_name}")
self.build_meta.add_ldflags("nvcc",   "-Wl,--no-as-needed", f"-l{nvrtc_builtins_lib_name}")
```

`get_cuda_lib_link_name_linux` globs the **build machine's** CUDA lib dir, and
`--no-as-needed` forces a `DT_NEEDED` entry **even though not one symbol is used from
it**. So the build box's CUDA minor gets welded into the wheel.

Verified against `cumm/core_cc_py312.so` from the currently pinned wheel - the only
NVRTC symbols it imports are the 12 public entry points, all exported by
`libnvrtc.so.13`, none by `libnvrtc-builtins`:

```
nvrtcAddNameExpression  nvrtcCompileProgram   nvrtcCreateProgram  nvrtcDestroyProgram
nvrtcGetCUBIN           nvrtcGetCUBINSize     nvrtcGetErrorString nvrtcGetLoweredName
nvrtcGetPTX             nvrtcGetPTXSize       nvrtcGetProgramLog  nvrtcGetProgramLogSize
```

**The `libnvrtc-builtins` `DT_NEEDED` is dead weight. NVRTC loads its own builtins at
runtime.**

---

## 2. Only `cumm` is affected - audit of all 11 pinned Linux wheels

Every `.so` in every Linux wheel in `requirements_trellis.txt` was downloaded and its
`DT_NEEDED` list read. Result:

| wheel                          | CUDA sonames linked                                              | verdict |
| ------------------------------ | ---------------------------------------------------------------- | ------- |
| **cumm-0.8.2.post1**           | `libcudart.so.13`, `libnvrtc.so.13`, ~~`libnvrtc-builtins.so.13.1`~~ | **was BROKEN, rebuilt - see section 0** |
| spconv-2.3.8                   | `libcudart.so.13`                                                  | clean |
| flash_attn-2.8.3.post1         | `libcudart.so.13` + torch libs                                     | clean |
| xformers-0.0.35.dev1135        | `libcudart.so.13` + torch libs                                     | clean |
| sageattention-2.2.0.post38     | `libcudart.so.13`, `libcuda.so.1` + torch libs                     | clean |
| torchao-0.18.0                 | `libcudart.so.13` + torch libs                                     | clean |
| kaolin-0.18.0.post41           | torch libs only (`libc10_cuda.so`)                                 | clean |
| nvdiffrast-0.4.0               | torch libs only (`libc10_cuda.so`)                                 | clean |
| diff_gaussian_rasterization    | torch libs only                                                    | clean |
| diffoctreerast-0.0.0           | torch libs only                                                    | clean |
| vox2seq-0.0.0                  | torch libs only                                                    | clean |

`cumm` is the **only** wheel in the whole set with a minor-versioned `DT_NEEDED`, and it
carries it in all four builds (`core_cc_py310.so` through `core_cc_py313.so`).

**Do not waste a rebuild on `spconv`.** Its `core_cc_py3XX.so` files link nothing but
`libcudart.so.13` - it never links NVRTC at all. The existing `spconv` wheel is correct
as shipped; rebuilding it changes nothing about this failure.

---

## 3. Fastest correct fix - no recompile, same filename (seconds)

Because the entry provides zero symbols, strip it. Run on the Linux build box:

```bash
pip install patchelf                       # or: apt-get install -y patchelf
WHL=cumm-0.8.2.post1+cu130.torch2.13.gd971d02.sm80.sm86.sm89.sm90a.sm100a.sm103a.sm120a.sm121a-py310-none-linux_x86_64.whl

rm -rf /tmp/cummfix && mkdir -p /tmp/cummfix
cd /tmp/cummfix && unzip -q "$OLDPWD/$WHL"

for so in cumm/core_cc_py*.so; do
    # remove whatever minor got baked in, not just .13.1
    for dep in $(patchelf --print-needed "$so" | grep '^libnvrtc-builtins\.so\.'); do
        patchelf --remove-needed "$dep" "$so"
        echo "stripped $dep from $so"
    done
done

# repack under the IDENTICAL name so requirements_trellis.txt needs no edit
rm -f "$OLDPWD/$WHL" && zip -qr9 "$OLDPWD/$WHL" .
cd "$OLDPWD"

# verify: must print libcudart.so.13 and libnvrtc.so.13 only
unzip -p "$WHL" cumm/core_cc_py312.so > /tmp/c.so && readelf -d /tmp/c.so | grep NEEDED
```

Then re-upload to the same HF path. `requirements_trellis.txt` is unchanged.

Caveat, and the reason the upstream flag exists at all: with the `DT_NEEDED` gone,
`libnvrtc-builtins.so.*` must be loadable when NVRTC first compiles a kernel. This repo
already guarantees that - `cuda_linux_deps.preload()` `dlopen`s it `RTLD_GLOBAL` from the
`nvidia/*/lib` wheel dirs before anything imports `cumm`, and `secourses_trellis.py`
calls `preload()` at start-up. Keep that call. If it is ever removed, use section 4.

---

## 4. If you do rebuild `cumm` from source

Same CUDA archs, same wheel name, same everything - change **only** the toolkit:

```bash
# torch 2.13.0+cu130 ships nvrtc 13.0.x, so build against the 13.0 toolkit, NOT 13.1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export CUMM_CUDA_VERSION=13.0
export CUMM_DISABLE_JIT=1
export CUMM_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0a;10.0a;10.3a;12.0a;12.1a"
```

The arch list above reproduces the `sm80.sm86.sm89.sm90a.sm100a.sm103a.sm120a.sm121a`
suffix in the pinned wheel name. Keep it identical or the filename stops matching
`requirements_trellis.txt`.

The rule, stated generally, so it survives the next CUDA bump:

> The CUDA **minor** of the build toolkit must equal the CUDA **minor** of the NVRTC
> wheel that the target torch build pins.

Read the target minor off the deployment box - never assume. This globs both wheel
layouts, so it works on CUDA 12 (`nvidia/cuda_nvrtc/lib`) and CUDA 13 (`nvidia/cu13/lib`)
alike:

```bash
python -c "import glob,nvidia; print(glob.glob(nvidia.__path__[0]+'/*/lib/libnvrtc-builtins*'))"
```

Better still, make the toolkit minor stop mattering - which is what the current wheel
does. See section 0: with the `DT_NEEDED` dropped and `RUNPATH` set to
`$ORIGIN/../nvidia/cu13/lib`, the wheel loads on any CUDA 13.x.

A cheaper, version-proof alternative: patch out the three `--no-as-needed` lines in
`cumm/common.py::CummNVRTCLink` before building. Then the toolkit minor stops mattering
entirely and the wheel loads on any CUDA 13.x.

---

## 5. Mandatory preflight before uploading any Linux CUDA wheel

One command. If it prints anything, the wheel is not shippable:

```bash
for f in *.whl; do
  d=$(mktemp -d); unzip -qo "$f" -d "$d"
  find "$d" -name '*.so' -exec sh -c \
    'readelf -d "$1" 2>/dev/null | grep -o "lib[A-Za-z0-9_+-]*\.so\.[0-9]*\.[0-9][0-9.]*"' _ {} \; \
    | sort -u | sed "s|^|$f: MINOR-VERSIONED DEP |"
  rm -rf "$d"
done
```

It flags any `DT_NEEDED` with a `major.minor` soname - the whole class of bug, not just
this one instance. Applies equally to the other in-house wheels
(`vox2seq`, `diffoctreerast`, `diff_gaussian_rasterization`, `kaolin`, `nvdiffrast`).

---

## 6. Windows is immune - and here is why, so nobody "fixes" it by accident

This bug **cannot** reach a Windows end user. Every `.pyd`/`.dll` in the shipped
`venv/Lib/site-packages` was scanned for CUDA imports. Across the entire distribution
exactly **one** binary names a CUDA DLL at all:

```
cumm\core_cc.cp312-win_amd64.pyd  ->  nvrtc64_130_0.dll     (present in torch\lib)
```

Every other native module - spconv, flash_attn, xformers, sageattention, kaolin,
nvdiffrast, torchao, diff_gaussian_rasterization, diffoctreerast, vox2seq - imports only
`c10.dll` / `torch_cpu.dll` / `torch_cuda.dll`, which are already loaded by the time they
import. Zero unresolved CUDA dependencies in the whole venv.

Four reasons it stays that way, all verified on the shipped venv with the CUDA toolkit
scrubbed from `PATH`:

1. **The minor never enters a Windows DLL name.** NVIDIA freezes it: both the CUDA 13.0
   and 13.1 toolkits ship `nvrtc64_130_0.dll`. (The 13.1 toolkit does ship
   `nvrtc-builtins64_131.dll` - but see point 2.)

2. **`nvrtc-builtins` is never in an import table.** Import tables of the shipped
   binaries:

   ```
   cumm/core_cc.cp312-win_amd64.pyd     -> nvrtc64_130_0.dll   (+ CRT only)
   spconv/core_cc.cp312-win_amd64.pyd   -> no CUDA DLL at all
   ```

   The builtins DLL is loaded by `nvrtc64_130_0.dll` itself at runtime out of its own
   directory, so it always matches the NVRTC that loaded it. The Linux `--no-as-needed`
   hack from section 1 is guarded by `if compat.InLinux:` and never runs here.

3. **`cumm` points itself at bundled torch.** `cumm/__init__.py` defines
   `_determine_windows_cuda_dll_dir()`: pip `nvidia-*` packages first, then
   `os.add_dll_directory(<torch>/lib)`. Confirmed live - with `PATH` reduced to
   `C:\Windows\system32;C:\Windows` and no system CUDA reachable, `import cumm`
   succeeded and the resolved module was
   `venv\Lib\site-packages\torch\lib\nvrtc64_130_0.dll`. It uses `find_spec`, so it works
   even when `torch` has not been imported yet.

4. **`torch\lib` ships the matched set** we distribute inside the venv:
   `nvrtc64_130_0.dll`, `nvrtc-builtins64_130.dll`, `cudart64_13.dll`,
   `nvJitLink_130_0.dll`. Consistent by construction, and there are no `nvidia-*` wheels
   in the Windows venv at all.

Net: a Windows user with no CUDA toolkit, or with a different one installed, resolves
everything out of the bundled `torch\lib`. Nothing to fix, and `cuda_linux_deps.preload()`
correctly no-ops on Windows.

**The one way to break Windows:** building `cumm`/`spconv` against a different CUDA
**major** than the bundled torch (e.g. a CUDA 12 build would import `nvrtc64_120_0.dll`,
which `torch\lib` for cu130 does not contain). Match the major, always.
