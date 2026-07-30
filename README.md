# SECourses TRELLIS Studio V10

## One-click image-to-3D AI generator for Windows, RunPod, SimplePod, Massed Compute, and Linux

<p align="center">
  <img src="assets/logo.webp" width="100%" alt="TRELLIS Structured 3D Latents">
</p>

<p align="center">
  <a href="https://www.patreon.com/SECourses/posts/trellis-1-click-117470976"><strong>V10 installers, tutorials, screenshots, and downloads</strong></a>
  ·
  <a href="https://discord.com/servers/software-engineering-courses-secourses-772774097734074388">SECourses Discord</a>
  ·
  <a href="https://arxiv.org/abs/2412.01506">Research paper</a>
  ·
  <a href="https://trellis3d.github.io">Original project page</a>
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch 2.13" src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="CUDA 13" src="https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white">
  <img alt="Gradio 6.20" src="https://img.shields.io/badge/Gradio-6.20-FF7C00?logo=gradio&logoColor=white">
  <img alt="Windows and Linux" src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux-2563EB">
</p>

SECourses TRELLIS Studio V10 is a complete implementation of **TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation**. It converts a single image or 2–4 views of the same object into:

- A textured **GLB 3D mesh**
- A **3D Gaussian splat PLY**
- An **MP4 turntable preview**
- Reproducible generation metadata

V10 combines a modern Gradio interface with automatic background removal, multi-view image conditioning, low-VRAM presets, batch image-to-3D processing, resumable model downloads, and one-click installers for local and cloud NVIDIA GPUs.

> [!IMPORTANT]
> Download the current `trellis_v10.zip` installer bundle and read the platform-specific instructions on the [SECourses Patreon post](https://www.patreon.com/SECourses/posts/trellis-1-click-117470976). The installer bundle contains the pinned Torch 2.13/CUDA 13 dependency file, model downloader, and launch scripts described below.

## V10 technology stack

| Component | V10 version or implementation |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0 + CUDA 13.0 |
| Torchvision | 0.28.0 + CUDA 13.0 |
| Gradio | 6.20.0 |
| 3D viewer | Native `gr.Model3D` |
| Primary interface | `secourses_trellis.py` |
| Image conditioner | DINOv2 ViT-L/14 with registers |
| Background removal | rembg + U2NET |
| Attention | FlashAttention, xFormers, SDPA, or naive |
| Sparse backend | spconv |
| Main outputs | Textured GLB, Gaussian PLY, MP4, metadata |

The V10 installer provides precompiled Torch 2.13/CUDA 13 wheels for the performance-critical packages and TRELLIS CUDA extensions, including:

- xFormers
- FlashAttention
- SageAttention
- TorchAO
- Triton for Windows
- nvdiffrast
- diff-gaussian-rasterization
- vox2seq
- diffoctreerast
- Kaolin
- spconv and cumm

These extensions do not need to compile during a normal V10 installation.

## Core single-image and multi-view image-to-3D features

### Single image to 3D

Upload or paste an object image, optionally remove the background, and generate a complete 3D asset. The input component accepts uploaded files and clipboard images.

### Multi-image to 3D

Provide 2–4 clean views of the **same object** and choose one of the supported multi-image algorithms:

- `stochastic`
- `multidiffusion`

The interface includes bundled multi-view example sets. Multi-image conditioning is experimental; single-image mode will usually produce the sharpest result when the source image is already strong.

### Two-stage TRELLIS generation controls

The interface exposes both stages of the image-to-3D pipeline:

1. **Sparse structure**
   - Guidance strength
   - Sampling steps
2. **Structured latent**
   - Guidance strength
   - Sampling steps

You can also use a fixed or randomized seed, repeat a generation any number of times, and reproduce previous assets from their saved metadata.

### Automatic background removal

The bundled U2NET/rembg workflow prepares images that do not already have a clean alpha channel. The app chooses the fastest available ONNX Runtime provider:

1. CUDA
2. DirectML
3. CPU

### Live progress, ETA, and cancellation

Every long operation reports:

- Current processing stage
- Percentage and iteration progress
- Iteration speed
- Elapsed time
- Estimated time remaining
- VRAM status
- Detailed browser and console logs

Generation, extraction, and batch jobs can be cancelled without closing the interface.

### Lazy engine loading

The heavy PyTorch, CUDA, spconv, Kaolin, and TRELLIS modules are not loaded while the web interface starts. The engine loads on the first processing action, or it can be preloaded in the background with `--preload`.

## Low-VRAM NVIDIA GPU presets

TRELLIS Studio detects the active NVIDIA GPU and its VRAM with `nvidia-smi`, then automatically selects one of seven protected presets.

| Preset | Precision | Mesh | Measured peak VRAM | Default video | Texture |
|---|---:|:---:|---:|---:|---:|
| 6 GB | fp16 | No | 5.1–5.5 GB | 512 px / 120 frames | 512 px |
| 8 GB | fp16 | Yes | 7.6–8.1 GB | 512 px / 150 frames | 1024 px |
| 10 GB | fp16 | Yes | 7.6–8.1 GB | 768 px / 180 frames | 1024 px |
| 12 GB | fp16 | Yes | 7.6–8.1 GB | 1024 px / 240 frames | 1024 px |
| 16 GB | fp32 | Yes | 12.1–12.9 GB | 1024 px / 240 frames | 1024 px |
| 24 GB | fp32 | Yes | 12.1–12.9 GB | 1024 px / 300 frames | 2048 px |
| 32 GB | fp32 | Yes | 12.1–12.9 GB | 1536 px / 360 frames | 2048 px |

The measurements include real device occupancy, the CUDA context, allocator blocks, and driver overhead. The range represents a typical subject through a subject that fills more of the 3D volume.

### How the 6 GB preset works

The structured-latent mesh decoder is the largest allocation in the application. It needs approximately:

- 7.6–8.1 GB in fp16
- 12.1–12.9 GB in fp32

The 6 GB preset skips mesh decoding and generates:

- A Gaussian splat PLY
- A color turntable video

Because there is no decoded mesh, GLB export and the geometry preview pass are unavailable for that run.

### Isolated job mode

Low-VRAM presets can run each operation in a separate process. When the operation finishes, the process exits and releases the complete CUDA context rather than only cached tensors. This can recover approximately 1.2–1.7 GB between jobs and prevents a worker out-of-memory error from taking down the web interface.

> [!CAUTION]
> The bundled `Windows_Start.bat` currently uses `--highvram` for maximum speed. High-VRAM mode needs approximately 10 GB or more because it keeps the models on the GPU. On a smaller GPU, remove `--highvram` from the final launch command and select the matching 6 GB or 8 GB preset inside the interface.

## Textured GLB, Gaussian PLY, and video outputs

### MP4 turntable preview

| Control | Range |
|---|---:|
| Resolution | 256–2048 px |
| Frames | 30–480 |
| Frame rate | 10–120 FPS |
| Encoder quality | 1–10 |
| Geometry pass | Optional |

The preview can show color only or a side-by-side color and geometry/normal pass. Disabling the geometry pass makes preview rendering approximately twice as fast.

Video files use H.264, YUV420p, and fast-start metadata for broad browser and media-player compatibility.

### Textured GLB mesh export

The GLB extraction pipeline:

1. Simplifies the generated mesh
2. Fills holes
3. Unwraps UV coordinates
4. Renders multiple texture-bake views
5. Bakes the Gaussian appearance into a texture
6. Exports a portable GLB asset

| GLB control | Range |
|---|---:|
| Mesh simplification factor | 0.20–0.99 |
| Texture size | 512, 1024, or 2048 px |
| Texture-bake view size | 256–1024 px |
| Texture-bake views | 20–200 |

GLB extraction is normally the slowest step. Higher texture-bake resolution and more views improve coverage but increase processing time and VRAM use.

### 3D Gaussian splat export

The generated Gaussian representation can be saved as a `.ply` file and previewed in the native browser 3D viewer. Gaussian PLY files are commonly around 50 MB, so large assets may take a moment to appear.

### Reproducible metadata

Optional metadata is saved beside the generated outputs and records:

- Application and code version
- Seed
- Single-image or multi-image mode
- Sparse-structure guidance and steps
- Structured-latent guidance and steps
- Multi-image algorithm
- Precision and attention backend
- Video settings
- Mesh vertex and triangle counts
- Number of Gaussians
- Generation duration
- Output filename and timestamp

## Batch image-to-3D processing

The Batch tab can process an entire folder containing:

- PNG
- JPG or JPEG
- WebP
- BMP
- TIF or TIFF

Batch features include:

- One or multiple generations per source image
- Selectable MP4, GLB, and Gaussian PLY outputs
- Reuse of every generation and extraction setting from the Generate tab
- Natural filename sorting
- Dedicated video, GLB, Gaussian, and metadata folders
- Image counting before a run
- Skip-existing support
- Processed, skipped, and failed totals
- Per-item timing, average speed, and ETA
- Live logs and cancellation

When **skip existing** is enabled, an image is skipped only when all outputs requested for that batch item already exist. This makes interrupted or incremental batch processing practical.

## Preset system

V10 stores every generation, video, extraction, precision, process-isolation, and batch option in presets.

### Protected presets

The built-in `Default` and seven VRAM presets live in:

```text
configs_trellis/presets_builtin/
```

They are regenerated from the source on every start and cannot be overwritten, deleted, or corrupted through the UI.

### User presets

Your custom presets live in:

```text
configs_trellis/presets_user/
```

The application never overwrites this folder. If a user preset was selected most recently, V10 restores it on the next start. Otherwise, the detected GPU determines the startup VRAM preset.

## System and engine tools

The System tab displays:

- Application, Python, Torch, and platform versions
- Active attention and sparse backends
- Precision and high-VRAM status
- Detected GPU and VRAM
- Selected VRAM tier
- Engine load state
- Current and peak VRAM use
- Output and preset paths

It also provides controls to:

- Load the engine immediately
- Release cached VRAM
- Refresh system information
- Open the application folder

## V10 one-click installer bundle

The current installer ZIP is available from the [TRELLIS V10 Patreon post](https://www.patreon.com/SECourses/posts/trellis-1-click-117470976).

| File | Purpose |
|---|---|
| `Windows_Install_Or_Update.bat` | Clone/update TRELLIS, create the Python 3.12 venv, install pinned packages, and download models |
| `Windows_Start.bat` | Activate the environment and launch TRELLIS Studio |
| `Windows_Download_Resume_Models.bat` | Resume and verify model downloads without reinstalling |
| `RunPod_Trellis_Install.sh` | Install on RunPod or SimplePod |
| `Massed_Compute_Install.sh` | Install on Massed Compute |
| `DownloadModels.py` | Resumable, multi-connection, SHA256-verified model downloader |
| `requirements_trellis.txt` | Pinned Torch 2.13/CUDA 13 dependencies and precompiled wheels |
| `RunPod_SimplePod_Instructions_READ.txt` | RunPod and SimplePod setup/start instructions |
| `Massed_Compute_Instructions_READ.txt` | Massed Compute setup/start instructions |
| `Pip_Freeze.txt` | Reference V10 Windows environment |

Keep `requirements_trellis.txt` and `DownloadModels.py` beside the matching installer script.

## Requirements and compatibility

### NVIDIA GPU

CUDA 13 requires:

- NVIDIA driver **580 or newer**
- Compute capability **sm_75 or newer**

Supported architecture families include compatible:

- Turing
- Ampere
- Ada
- Hopper
- Blackwell

In consumer GPU terms, use an RTX 20, 30, 40, or 50 series GPU or newer compatible hardware. More VRAM enables fp32, higher sampling steps, denser meshes, larger previews, and higher-resolution textures.

### Windows requirements

- Python 3.12.10
- Git with Git LFS
- FFmpeg
- CUDA 13
- cuDNN 9.17 or newer
- Visual Studio Community with the C++ workload and options
- Current NVIDIA driver

Follow the requirements video and written tutorial linked in the [Patreon post](https://www.patreon.com/SECourses/posts/trellis-1-click-117470976) before running the installer.

### Folder requirements

- Extract V10 into a fresh folder.
- Use a short, normal path.
- Avoid spaces and special characters in installer paths.
- Do not install from OneDrive, Dropbox, Google Drive, or another synchronization folder.
- Keep all installer-bundle files together.

## Quick start: Windows

1. Install the Windows requirements.
2. Download `trellis_v10.zip` from the [V10 post](https://www.patreon.com/SECourses/posts/trellis-1-click-117470976).
3. Extract the ZIP into a new local folder.
4. Run:

   ```text
   Windows_Install_Or_Update.bat
   ```

5. Wait for dependencies and all models to finish downloading.
6. Start the application with:

   ```text
   Windows_Start.bat
   ```

If a model download is interrupted, run:

```text
Windows_Download_Resume_Models.bat
```

The resume script installs nothing. It verifies completed files, skips valid files, and continues partial downloads.

## Quick start: RunPod and SimplePod

Use the recommended CUDA 13 template and persistent storage described in `RunPod_SimplePod_Instructions_READ.txt`.

Place the extracted installer files in `/workspace`, then run:

```bash
cd /workspace
export HF_HOME="/workspace"
export TORCH_HOME="/workspace/torch"
export U2NET_HOME="/workspace/u2net"
export UV_CONCURRENT_INSTALLS=4
chmod +x RunPod_Trellis_Install.sh
./RunPod_Trellis_Install.sh
```

Start the application in a new terminal:

```bash
unset LD_LIBRARY_PATH
cd /workspace/TRELLIS
export HF_HOME="/workspace"
export TORCH_HOME="/workspace/torch"
export U2NET_HOME="/workspace/u2net"
export PYTHONWARNINGS=ignore
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0
source ./venv/bin/activate
python secourses_trellis.py --share --highvram
```

Use the same `HF_HOME`, `TORCH_HOME`, and `U2NET_HOME` values on every start. Otherwise, DINOv2 or U2NET may download again outside the persistent volume.

Remove `--highvram` when using a GPU that does not have enough VRAM to keep the complete model stack resident.

## Quick start: Massed Compute

Extract the V10 installer bundle into a normal folder outside any synchronized directory:

```bash
chmod +x Massed_Compute_Install.sh
./Massed_Compute_Install.sh
```

The installer:

- Reuses a compatible Python 3.12 environment when available
- Installs a stable Python 3.12 build when required
- Falls back to a uv-managed Python 3.12 build when needed
- Creates the virtual environment
- Installs the pinned CUDA stack
- Downloads and verifies every model
- Installs FFmpeg and FFprobe

Start the application:

```bash
cd TRELLIS
export PYTHONWARNINGS=ignore
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0
source ./venv/bin/activate
python3 secourses_trellis.py --share --highvram
```

## Resumable model downloader

`DownloadModels.py` replaces a basic snapshot download with a robust cross-platform downloader.

### Download behavior

- Up to 16 parallel connections
- Byte-range resume
- Automatic retries and exponential backoff
- File-size checks
- SHA256 verification
- Verified-file cache
- Safe repeated execution
- Hugging Face token support
- Separate third-party download session so a Hugging Face token is not sent to other hosts

### Models downloaded

| Model | Approximate size | Destination |
|---|---:|---|
| TRELLIS checkpoints | 4 GB | `TRELLIS/models/` |
| DINOv2 ViT-L/14 with registers | 1.2 GB | `TORCH_HOME/hub/checkpoints/` |
| U2NET background-removal model | 176 MB | `U2NET_HOME/` |

Pre-downloading DINOv2 and U2NET prevents the first generation from pausing for another large download inside the web interface.

### Downloader commands

Run from the V10 installer directory with the TRELLIS virtual environment active:

```bash
python DownloadModels.py
python DownloadModels.py --model trellis
python DownloadModels.py --model dinov2 rembg
python DownloadModels.py --all
python DownloadModels.py --list
python DownloadModels.py --dry-run
```

If a transfer fails, run the same command again. Completed files are skipped and partial files resume.

## Using TRELLIS Studio V10

1. Start `secourses_trellis.py`.
2. Confirm that the automatically selected VRAM tier matches the active GPU.
3. Upload one image or switch to the multi-image tab and provide 2–4 views.
4. Use **Remove background & preview** when the source is not already a clean cutout.
5. Choose a random or fixed seed.
6. Adjust sparse-structure and structured-latent settings if needed.
7. Select:
   - **Generate preview video** for a fast first look, or
   - **Generate + extract everything** for MP4, GLB, and PLY outputs.
8. Download the assets from the interface or open the output folder.

### Quality guidance

| Goal | Suggested change |
|---|---|
| Sharper geometry | Use 20–25 sparse-structure sampling steps |
| Cleaner texture | Use a 2048 px texture and a lower simplification value such as 0.7–0.8 |
| Faster iteration | Use 10–12 steps, 512 px / 120-frame video, and disable the geometry pass |
| Lower VRAM | Use fp16; disable mesh generation when necessary |
| Reproducible assets | Disable random seed and keep the generated metadata |

## Command-line options

```text
--precision fp32|fp16
--attention flash_attn|xformers|sdpa|naive
--xformers
--share
--highvram
--preload
--no-tf32
--no-browser
--port PORT
--listen
```

| Option | Description |
|---|---|
| `--precision` | Select fp32 or lower-VRAM fp16 model weights |
| `--attention` | Force an attention backend |
| `--xformers` | Shortcut for `--attention xformers` |
| `--share` | Create a public Gradio share URL |
| `--highvram` | Keep every model resident on the GPU for maximum speed |
| `--preload` | Load the engine in the background immediately after UI startup |
| `--no-tf32` | Disable TF32 matrix multiplication |
| `--no-browser` | Do not open a local browser automatically |
| `--port` | Select the Gradio server port |
| `--listen` | Bind to `0.0.0.0` for LAN or remote access |

Without an explicit attention option, V10 uses:

1. FlashAttention when available
2. xFormers when FlashAttention is unavailable
3. PyTorch SDPA as the next fallback

## Output folders

```text
TRELLIS/
├── outputs_trellis/
│   ├── video/
│   ├── glb/
│   ├── gaussian/
│   └── metadata/
├── batch_input_images/
├── batch_outputs_trellis/
│   ├── video/
│   ├── glb/
│   ├── gaussian/
│   └── metadata/
├── configs_trellis/
│   ├── presets_builtin/
│   └── presets_user/
└── tmp/
    └── jobs/
```

Output filenames are reserved with a file lock and a collision-safe numeric sequence. Multi-generation runs receive an additional per-generation suffix.

## Python examples

The repository includes:

- [`example.py`](example.py) for single-image generation
- [`example_multi_image.py`](example_multi_image.py) for multi-view generation
- [`app.py`](app.py) for the original lightweight Gradio demo
- [`api_spz/main_api.py`](api_spz/main_api.py) for FastAPI integration

The full V10 Studio interface is:

```bash
python secourses_trellis.py
```

After the V10 installer has placed the models in `TRELLIS/models`, custom code can load the local pipeline with:

```python
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("models")
```

Refer to the bundled examples for complete rendering, GLB export, PLY export, and multi-image calls.

## Upstream TRELLIS project

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="TRELLIS generated 3D assets">
</p>

TRELLIS is a large 3D asset generation model. Its unified Structured LATent (SLAT) representation can decode into multiple 3D formats, including Radiance Fields, 3D Gaussians, and meshes. Rectified Flow Transformers provide the generative backbones.

The upstream project provides pretrained models with up to two billion parameters, trained on a large dataset of 500,000 diverse 3D objects.

### Upstream resources

- [TRELLIS paper](https://arxiv.org/abs/2412.01506)
- [TRELLIS project page](https://trellis3d.github.io)
- [Original Microsoft TRELLIS repository](https://github.com/microsoft/TRELLIS)
- [Original Hugging Face demo](https://huggingface.co/spaces/JeffreyXiang/TRELLIS)
- [TRELLIS-image-large model](https://huggingface.co/JeffreyXiang/TRELLIS-image-large)

## Dataset

TRELLIS-500K contains 500,000 3D assets curated from:

- [Objaverse XL](https://objaverse.allenai.org/)
- [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
- [HSSD](https://huggingface.co/datasets/hssd/hssd-models)
- [Toys4K](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k)

See [`DATASET.md`](DATASET.md) for dataset and preprocessing details.

## License

TRELLIS models and the majority of the code are released under the [`LICENSE`](LICENSE) included in this repository. Individual submodules and third-party dependencies may use different licenses.

Review the licenses of all models, submodules, and dependencies before commercial use or redistribution.

## Citation

If you use TRELLIS in academic work, cite the original paper:

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```

## Support and updates

- [TRELLIS V10 Patreon post](https://www.patreon.com/SECourses/posts/trellis-1-click-117470976)
- [SECourses Discord](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388)
- [SECourses Stable Diffusion and Generative AI repository](https://github.com/FurkanGozukara/Stable-Diffusion)
- [Patreon posts and scripts index](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Patreon-Posts-Index.md)

