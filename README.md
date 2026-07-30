# This repo is made for SECourses Premium App download and use from here : https://www.patreon.com/SECourses/posts/trellis-1-click-117470976

## App download and installer here : https://www.patreon.com/SECourses/posts/trellis-1-click-117470976

Video tutorial : [https://youtu.be/EhU7Jil9WAk](https://youtu.be/EhU7Jil9WAk)

-   We have TRELLIS 2 app too but since someone requested I updated this amazing app
    
    -   TRELLIS 2 : [https://www.patreon.com/SECourses/posts/trellis-2-app-1-147686623](https://www.patreon.com/SECourses/posts/trellis-2-app-1-147686623)
        
### 30 July 2026 V10 Update

-   We have completey remade the app and read below to understand how it works and full features
    
-   Make a fresh install and read the below carefully please
    
-   We use latest pre-compiled wheels and Torch 2.13 and CUDA 13
    

### Windows Requirements

-   Python 3.12.10, FFmpeg, CUDA 13, cuDNN 9.17 or above, Visual Studio Community Edition with all C++ options selected
    
    -   Don't worry CUDA 13 works with all GPUs - make sure you have updated NVIDIA driver
        
    -   Follow this requirements tutorial video exactly : [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)
        
    -   Follow its updated post with links and screenshots exactly : [https://www.patreon.com/SECourses/posts/requirements-written-tutorial-111553210](https://www.patreon.com/SECourses/posts/requirements-written-tutorial-111553210)
        

### RunPod, SimplePod, Massed Compute and Linux Users

-   RunPod and SimplePod please follow : RunPod\_SimplePod\_Instructions\_READ.txt
    
    -   [https://get.runpod.io/955rkuppqv4h](https://get.runpod.io/955rkuppqv4h)
        
    -   [https://simplepod.ai/ref?user=secourses](https://simplepod.ai/ref?user=secourses)
        
-   Massed Compute and Local Linux please follow : Massed\_Compute\_Instructions\_READ.txt
    
    -   [https://vm.massedcompute.com/signup?linkId=lp\_034338&sourceId=secourses&tenantId=massed-compute](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute)
        

### TRELLIS 1 FEATURES

-   1-Click installers with following pre-compiled wheels for both Windows and Linux
    
<img width="1714" height="1333" alt="image" src="https://github.com/user-attachments/assets/0819eaf9-1763-4630-9bbc-c69c152637a5" />

<img width="2173" height="1000" alt="image" src="https://github.com/user-attachments/assets/49737ba1-a015-4f9d-a055-62304d3a1954" />

-   Modern Torch 2.13 and CUDA 13 installation with Python 3.12 VENV
    
-   Latest version modern looking Gradio 6.20.0 interface
    
-   Fully working fully optimized very best configuration set TRELLIS 1 app
    
-   Click to see full size screenshot and all features below
    
<img width="3842" height="6072" alt="image" src="https://github.com/user-attachments/assets/4b223c65-9134-459f-9835-2bacea90f1dc" />

Supporting multiple image input for better quality generation see below image

<img width="3842" height="6406" alt="image" src="https://github.com/user-attachments/assets/53cea5db-b0f2-4788-8954-9293a4af8a10" />

Fully supported batch folder processing

<img width="3840" height="2420" alt="image" src="https://github.com/user-attachments/assets/93ae1e39-3eb1-4437-af60-15d4adccfe3a" />

Fully supported preset system

<img width="3840" height="2719" alt="image" src="https://github.com/user-attachments/assets/7be4764a-f2f5-43ce-bbca-e2ce6e476b24" />


### TRELLIS V10 Image-to-3D AI Generator – Complete Features

SECourses TRELLIS Studio V10 is a one-click implementation of TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation for Windows, RunPod, SimplePod, Massed Compute, and Linux. This AI 3D model generator converts a single image or 2–4 multi-view images of the same object into a textured GLB mesh, a 3D Gaussian splat PLY file, and an MP4 turntable preview. V10 uses Python 3.12, PyTorch 2.13.0, CUDA 13, torchvision 0.28.0, and Gradio 6.20.0.

### Core Single-Image and Multi-View Image-to-3D Features

-   Single image to 3D: upload or paste an object image, optionally remove its background, and generate a complete 3D asset.
    
-   Multi-image to 3D: provide 2–4 clean views of the same object and choose the stochastic or multidiffusion algorithm. Built-in multi-view example sets are included. Multi-image mode is experimental; single-image mode is usually sharper.
    
-   Two-stage generation controls: separately control sparse-structure guidance and sampling steps, structured-latent guidance and sampling steps, random or fixed seeds, and any number of repeated generations.
    
-   Automatic background removal: the bundled U2NET/rembg workflow prepares images without an alpha channel and automatically uses CUDA, DirectML, or CPU when available.
    
-   Fast workflow buttons: generate only the preview first, or use “Generate + extract everything” to create the preview video, textured GLB mesh, and Gaussian splat in one run.
    
-   Live progress and cancellation: every long operation reports the current stage, percentage, iteration speed, elapsed time, ETA, VRAM status, and a live log in both the browser and console. Generation and batch jobs can be cancelled.
    
-   Modern native 3D preview: V10 uses the built-in Gradio Model3D viewer for GLB and PLY previews, with direct download buttons and no custom viewer component.
    

### Low-VRAM NVIDIA GPU Presets and Performance

-   The app detects the NVIDIA GPU and VRAM with nvidia-smi, then automatically selects one of seven protected presets: 6, 8, 10, 12, 16, 24, or 32 GB. Each preset changes precision, sampling steps, video quality, texture quality, bake settings, mesh generation, and process isolation together.
    
-   6 GB preset: approximately 5.1–5.5 GB measured peak VRAM in fp16. It produces the Gaussian splat and turntable video but skips mesh decoding, so GLB export is unavailable.
    
-   8–12 GB presets: approximately 7.6–8.1 GB measured peak VRAM in fp16 with the mesh and Gaussian enabled. The 10 GB tier adds the geometry pass; the 12 GB tier uses a 1024 px, 240-frame preview by default.
    
-   16–32 GB presets: approximately 12.1–12.9 GB measured peak VRAM in fp32, with progressively higher sampling, video, mesh, and texture quality.
    
-   Isolated job mode runs each task in its own process so the CUDA context is fully released afterward. This can recover roughly 1.2–1.7 GB between runs and protects the web interface if a worker runs out of memory.
    
-   FlashAttention, xFormers, SDPA, and naive attention backends are supported. The app automatically chooses the best installed backend, enables TF32 on supported NVIDIA GPUs, and includes a dedicated high-VRAM mode for maximum speed.
    

Important low-VRAM Windows note: the bundled Windows\_Start.bat currently launches with the --highvram flag, which is the fastest mode and needs about 10 GB of VRAM. If your GPU has less VRAM, remove --highvram from the final launch command and then select the matching 6 GB or 8 GB preset inside the app.

### Textured GLB, Gaussian Splat PLY, and Video Outputs

-   MP4 turntable preview: adjustable 256–2048 px resolution, 30–480 frames, 10–120 FPS, encoder quality 1–10, and an optional side-by-side geometry/normal pass. Disabling the geometry pass makes preview rendering about twice as fast.
    
-   Textured GLB mesh export: the extraction pipeline simplifies the mesh, fills holes, unwraps UVs, and bakes the Gaussian appearance into a texture. Controls include a 0.20–0.99 simplification factor, 512/1024/2048 px textures, 256–1024 px bake views, and 20–200 texture-bake views.
    
-   3D Gaussian splat export: save the generated Gaussian representation as a PLY file and preview it directly in the browser. PLY files are commonly around 50 MB, so the viewer can take a moment to display them.
    
-   Reproducible metadata: optional TXT/JSON metadata records the seed, stage guidance and steps, precision, attention backend, input mode, video settings, mesh vertex/triangle counts, number of Gaussians, timing, and output filename.
    
-   Outputs are organized automatically into separate video, GLB, Gaussian, and metadata folders with collision-safe numeric filenames.
    

### Batch Image-to-3D Processing

-   Process an entire folder of PNG, JPG, JPEG, WebP, BMP, TIF, or TIFF images.
    
-   Create one or multiple generations per image and choose any combination of MP4 preview, textured GLB mesh, and Gaussian PLY output.
    
-   Reuse every generation, video, extraction, precision, and VRAM setting from the Generate tab.
    
-   Resume practical batch work with “skip existing”: an image is skipped when all requested output files already exist.
    
-   Natural filename sorting, dedicated output subfolders, image counting, processed/skipped/failed totals, per-item timing, average speed, ETA, live logs, and batch cancellation are included.
    

### One-Click Installers and Resumable Model Downloads

-   Windows\_Install\_Or\_Update.bat clones or updates the SECourses TRELLIS repository and submodules, creates a Python 3.12 virtual environment, installs the pinned dependencies with uv, and downloads every required model automatically.
    
-   RunPod\_Trellis\_Install.sh supports RunPod and SimplePod with a managed Python 3.12 environment, persistent Hugging Face/Torch/U2NET caches, pinned packages, automatic model downloads, and a bundled Linux FFmpeg/FFprobe installation.
    
-   Massed\_Compute\_Install.sh installs or reuses a stable Python 3.12 environment, falls back to uv-managed Python when needed, installs the CUDA stack, downloads the models, and installs FFmpeg.
    
-   Precompiled Torch 2.13/CUDA 13 wheels are supplied for xFormers, FlashAttention, SageAttention, TorchAO, nvdiffrast, diff-gaussian-rasterization, vox2seq, diffoctreerast, Kaolin, spconv, and cumm. The TRELLIS CUDA extensions do not need to compile during installation.
    
-   The custom model downloader uses up to 16 connections, automatic retry and backoff, byte-range resume, file-size checks, and SHA256 verification. Verified files are skipped safely, so interrupted downloads continue instead of restarting.
    
-   The installer pre-downloads roughly 4 GB of TRELLIS checkpoints, the 1.2 GB DINOv2 ViT-L/14 image conditioner, and the 176 MB U2NET background-removal model so the first generation does not stop for another large download.
    
-   Windows\_Download\_Resume\_Models.bat only resumes/verifies model files; it does not reinstall the application. It is safe to run repeatedly.
    

### Compatibility and Installation Notes

-   Windows requirements: Python 3.12.10, Git with Git LFS, FFmpeg, CUDA 13, cuDNN 9.17 or newer, Visual Studio Community with the C++ workload/options, and an up-to-date NVIDIA driver.
    
-   CUDA 13 compatibility: use NVIDIA driver 580 or newer and a GPU with compute capability sm\_75 or newer—RTX 20, 30, 40, and 50 series plus compatible Turing, Ampere, Ada, Hopper, and Blackwell GPUs.
    
-   Keep the ZIP files together: requirements\_trellis.txt and DownloadModels.py must remain beside the matching installer script. Use a short normal folder path, avoid spaces/special characters, and do not install from a cloud-synchronization folder.
    
-   For RunPod and SimplePod, extract the files in /workspace and use persistent storage plus the same HF\_HOME, TORCH\_HOME, and U2NET\_HOME values on every start, otherwise large models may download again.
    
-   Make a fresh V10 installation. If a cloud storage error corrupts the virtual environment, remove only the TRELLIS venv and rerun the installer; completed model files will be verified and skipped.
    

### Recommended TRELLIS V10 Workflow

1.  Extract the latest trellis\_v10.zip into a new folder and follow the requirements tutorial linked above.
    
2.  Run the installer for Windows, RunPod/SimplePod, or Massed Compute and allow the automatic model verification/download to finish.
    
3.  If a download is interrupted, rerun the installer or use Windows\_Download\_Resume\_Models.bat; completed files are skipped and partial files resume.
    
4.  Start SECourses TRELLIS Studio, confirm the automatically selected VRAM preset, and lower the tier if another GPU application is using VRAM.
    
5.  Use a clean object image with clear separation from the background, or provide 2–4 consistent views of the same object. Generate a preview first, then extract GLB/PLY—or use “Generate + extract everything.”
    
6.  For reproducible AI 3D assets, disable random seed and save the metadata file. For sharper geometry, try 20–25 sampling steps; for cleaner textures, use 2048 px texture size when VRAM allows.
    

In short: TRELLIS Studio V10 is a complete single-image and multi-view image-to-3D workflow with one-click Windows and cloud installers, low-VRAM presets, batch 3D asset generation, textured GLB export, 3D Gaussian splat PLY export, automatic background removal, resumable verified model downloads, and a modern Gradio interface

System info

<img width="3840" height="3212" alt="image" src="https://github.com/user-attachments/assets/c58891a5-c99b-4ad3-9e96-c73f799d5d68" />


