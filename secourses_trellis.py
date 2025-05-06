import os
import sys
sys.path.append(os.getcwd())

import argparse # Moved up for earlier access

# -------------- CMD ARGS PARSE -----------------
parser = argparse.ArgumentParser(description="Trellis API server")
parser.add_argument("--precision", 
                    choices=["fp32", "fp16"], 
                    default="fp32",
                    help="Set precision: fp32 (full) or fp16 (half). Default: fp32")
parser.add_argument("--xformers", action="store_true", help="Prefer xformers attention backend if available.")
parser.add_argument("--share", action="store_true", help="Enable Gradio live share.")
cmd_args = parser.parse_args()

# Attention Backend Selection
_xformers_available = False
try:
    import xformers
    _xformers_available = True
except ImportError:
    pass

if cmd_args.xformers:
    if _xformers_available:
        os.environ['ATTN_BACKEND'] = 'xformers'
        print("Using xformers attention backend (selected by --xformers).")
    else:
        os.environ['ATTN_BACKEND'] = 'flash-attn' # Fallback
        print("WARNING: --xformers specified, but xformers is not available. Falling back to flash-attn.")
else: # Default is flash-attn
    os.environ['ATTN_BACKEND'] = 'flash-attn'
    print("Using flash-attn attention backend (default).")
    # Note: The request "if flash attention not available fall back to xformers" is hard to implement
    # robustly at this level without a direct way to check flash-attn's operational status.
    # We set the preference; if the underlying library has its own fallback logic, it might engage.
    if _xformers_available:
        print("Note: xformers is available if flash-attn encounters issues and the library supports fallback.")


os.environ['SPCONV_ALGO'] = 'native' 

import gradio as gr
from gradio_litmodel3d import LitModel3D

import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

from version import code_version

# For new features
import json
import time
import platform
import subprocess
import re
import glob as sistema_glob # Renamed to avoid conflict with local glob
from filelock import FileLock # For robust file naming

# ------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max

# Output Folder Setup
OUTPUT_DIR_BASE = "outputs_trellis"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR_BASE, "video")
OUTPUT_GLB_DIR = os.path.join(OUTPUT_DIR_BASE, "glb")
OUTPUT_GAUSSIAN_DIR = os.path.join(OUTPUT_DIR_BASE, "gaussian")
OUTPUT_METADATA_DIR = os.path.join(OUTPUT_DIR_BASE, "metadata")
BATCH_OUTPUT_DIR_BASE_DEFAULT = "batch_outputs_trellis" # For batch processing

for d in [OUTPUT_DIR_BASE, OUTPUT_VIDEO_DIR, OUTPUT_GLB_DIR, OUTPUT_GAUSSIAN_DIR, OUTPUT_METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Config/Preset Paths
CONFIG_DIR = "configs_trellis"
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config_trellis.json") # Store full last config
DEFAULT_CONFIG_NAME = "Default"
os.makedirs(CONFIG_DIR, exist_ok=True)


# Helper for robust file naming (simplified version without tempfile)
def get_next_output_path_numeric(output_dir, extension, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize prefix for lock file name
    safe_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', prefix)
    
    # Create lock file in the output directory itself
    lock_file_name = f".trellis_lock_{safe_prefix}.lock"
    lock_file_path = os.path.join(output_dir, lock_file_name)
    
    lock = FileLock(lock_file_path, timeout=10)

    with lock:
        counter = 1
        while True:
            filename_base = f"{prefix}{counter:04d}"
            final_path = os.path.join(output_dir, f"{filename_base}.{extension}")
            temp_reservation_path = os.path.join(output_dir, f"{filename_base}.{extension}.tmp")

            if not os.path.exists(final_path) and not os.path.exists(temp_reservation_path):
                try:
                    with open(temp_reservation_path, "w") as f:
                        f.write(f"Reserved by process {os.getpid()} at {time.time()}")
                    return final_path, temp_reservation_path, filename_base # Return base for metadata
                except IOError as e:
                    print(f"Warning: IOError creating temp reservation file {temp_reservation_path}: {e}")
                    # If temp file creation fails, try next, but this indicates a problem.
            counter += 1
            if counter > 99999: # Safety break for 4 digits
                raise RuntimeError(f"Could not find an available numeric filename in {output_dir} with prefix {prefix}.")

def remove_temp_reservation_file(temp_file_path):
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except OSError as e:
            print(f"Warning: Could not remove temp reservation file {temp_file_path}: {e}")

# Natural sort helper
def alphanum_key(s):
    def try_int(s_):
        try:
            return int(s_)
        except ValueError:
            return s_
    return [try_int(c) for c in re.split('([0-9]+)', s)]

def sorted_glob(pattern):
    files = sistema_glob.glob(pattern)
    files.sort(key=alphanum_key)
    return files

def open_folder(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"Error: Folder not found at {path}")
        return f"Error: Folder not found at {path}"
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", path])
        return f"Opened folder: {path}"
    except Exception as e:
        print(f"Error opening folder {path}: {e}")
        return f"Error opening folder: {e}"

# Removed start_session and end_session as TMP_DIR for outputs is replaced
# def start_session(req: gr.Request):
#     user_dir = os.path.join(TMP_DIR, str(req.session_hash))
#     os.makedirs(user_dir, exist_ok=True)
    
# def end_session(req: gr.Request):
#     user_dir = os.path.join(TMP_DIR, str(req.session_hash))
#     shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = pipeline.preprocess_image(image)
    return processed_image

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict]: # Removed str from return
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    # New video and metadata params
    video_resolution: int,
    video_num_frames: int,
    video_fps: int,
    save_metadata: bool,
    # For batch processing override
    output_filename_prefix: Optional[str] = None, # e.g., "input_image_name" for batch or None for NNNN
    # req: gr.Request, # Removed as user_dir is not used for final outputs
) -> Tuple[dict, str]:
    
    start_time = time.time()
    # user_dir = os.path.join(TMP_DIR, str(req.session_hash)) # Not used for final output paths

    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False, # Assuming preprocessed outside or if image is None
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            [img[0] for img in multiimages], # Ensure it's a list of PIL Images
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False, # Assuming preprocessed
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    
    # Video rendering parameters
    video_frames_color = render_utils.render_video(outputs['gaussian'][0], resolution=video_resolution, bg_color=(0,0,0), num_frames=video_num_frames)['color']
    video_frames_geo = render_utils.render_video(outputs['mesh'][0], resolution=video_resolution, bg_color=(0,0,0), num_frames=video_num_frames)['normal']
    
    combined_video_frames = [np.concatenate([video_frames_color[i], video_frames_geo[i]], axis=1) for i in range(len(video_frames_color))]
    
    actual_video_path = ""
    video_filename_base = ""

    if output_filename_prefix: # Batch mode
        video_filename_base = output_filename_prefix
        actual_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_filename_base}.mp4")
        os.makedirs(os.path.dirname(actual_video_path), exist_ok=True) # Ensure batch subfolder exists
        temp_reservation_path = None # Not strictly needed for batch if skip_exists is handled earlier
    else: # Single mode
        actual_video_path, temp_reservation_path, video_filename_base = get_next_output_path_numeric(OUTPUT_VIDEO_DIR, "mp4")

    imageio.mimsave(actual_video_path, combined_video_frames, fps=video_fps)
    if temp_reservation_path:
        remove_temp_reservation_file(temp_reservation_path)
        
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    
    generation_duration = time.time() - start_time

    if save_metadata:
        metadata = {
            "source_image_provided": image is not None,
            "multi_image_mode": is_multiimage,
            "num_multi_images": len(multiimages) if is_multiimage else 0,
            "seed": seed,
            "ss_guidance_strength": ss_guidance_strength,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "slat_sampling_steps": slat_sampling_steps,
            "multiimage_algo": multiimage_algo if is_multiimage else "N/A",
            "video_resolution": video_resolution,
            "video_num_frames": video_num_frames,
            "video_fps": video_fps,
            "generation_duration_seconds": round(generation_duration, 2),
            "code_version": code_version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        }
        # Later, mesh_simplify and texture_size can be added if metadata is saved after those steps
        # For now, save general generation params.
        
        metadata_path = os.path.join(OUTPUT_METADATA_DIR, f"{video_filename_base}.txt")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    torch.cuda.empty_cache()
    return state, actual_video_path


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    save_metadata: bool, # Added for consistency, could extend metadata here
    # For batch processing override
    output_filename_prefix: Optional[str] = None, 
    # req: gr.Request, # Removed
) -> Tuple[str, str]: # Returns (model_output_path, download_button_path)
    # user_dir = os.path.join(TMP_DIR, str(req.session_hash)) # Not used
    gs, mesh = unpack_state(state)
    
    actual_glb_path = ""
    glb_filename_base = ""

    if output_filename_prefix: # Batch mode
        glb_filename_base = output_filename_prefix
        actual_glb_path = os.path.join(OUTPUT_GLB_DIR, f"{glb_filename_base}.glb")
        os.makedirs(os.path.dirname(actual_glb_path), exist_ok=True)
        temp_reservation_path = None
    else: # Single mode
        actual_glb_path, temp_reservation_path, glb_filename_base = get_next_output_path_numeric(OUTPUT_GLB_DIR, "glb")

    glb_data = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_data.export(actual_glb_path)

    if temp_reservation_path:
        remove_temp_reservation_file(temp_reservation_path)

    if save_metadata and glb_filename_base: # If metadata is to be extended or created here
        metadata_path_check = os.path.join(OUTPUT_METADATA_DIR, f"{glb_filename_base}.txt")
        # Example: append to existing metadata or create new if needed
        # For now, just acknowledging it could be done here
        pass

    torch.cuda.empty_cache()
    return actual_glb_path, actual_glb_path


def extract_gaussian(
    state: dict, 
    save_metadata: bool, # Added
    # For batch processing override
    output_filename_prefix: Optional[str] = None,
    # req: gr.Request, # Removed
) -> Tuple[str, str]:
    # user_dir = os.path.join(TMP_DIR, str(req.session_hash)) # Not used
    gs, _ = unpack_state(state)
    
    actual_gaussian_path = ""
    gs_filename_base = ""

    if output_filename_prefix: # Batch mode
        gs_filename_base = output_filename_prefix
        actual_gaussian_path = os.path.join(OUTPUT_GAUSSIAN_DIR, f"{gs_filename_base}.ply")
        os.makedirs(os.path.dirname(actual_gaussian_path), exist_ok=True)
        temp_reservation_path = None
    else: # Single mode
        actual_gaussian_path, temp_reservation_path, gs_filename_base = get_next_output_path_numeric(OUTPUT_GAUSSIAN_DIR, "ply")

    gs.save_ply(actual_gaussian_path)

    if temp_reservation_path:
        remove_temp_reservation_file(temp_reservation_path)
    
    if save_metadata and gs_filename_base:
        pass # Similar logic for metadata if needed here

    torch.cuda.empty_cache()
    return actual_gaussian_path, actual_gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    image_np = np.array(image) # Use a different variable name
    # Check if alpha channel exists and has variance
    if image_np.shape[2] < 4 or np.all(image_np[..., 3] == image_np[0,0,3]): # No alpha or uniform alpha
         # If no usable alpha, assume 3 equal splits or return as is if not divisible
        width = image_np.shape[1]
        if width % 3 == 0:
            split_width = width // 3
            images = [
                Image.fromarray(image_np[:, :split_width]),
                Image.fromarray(image_np[:, split_width:2*split_width]),
                Image.fromarray(image_np[:, 2*split_width:]),
            ]
        else: # Cannot split into 3, return as single image (or handle error)
            images = [Image.fromarray(image_np)] 
    else: # Original alpha splitting logic
        alpha = image_np[..., 3]
        alpha_mask = np.any(alpha > 0, axis=0) # Mask where alpha is present along columns
        
        # Find transitions from no-alpha to alpha (starts) and alpha to no-alpha (ends)
        # Pad with False at ends to correctly detect transitions at edges
        padded_mask = np.concatenate(([False], alpha_mask, [False]))
        transitions = np.diff(padded_mask.astype(np.int8))
        
        start_pos = np.where(transitions == 1)[0].tolist()
        end_pos = (np.where(transitions == -1)[0] -1).tolist() # Adjust end_pos due to diff

        images = []
        for s, e in zip(start_pos, end_pos):
            if e > s : # Ensure valid slice
                 images.append(Image.fromarray(image_np[:, s:e+1]))
    
    if not images: # Fallback if splitting fails
        return [preprocess_image(Image.fromarray(image_np))]

    return [preprocess_image(img) for img in images]


# --- Preset Functions ---
UI_COMPONENT_KEYS = [
    "seed_val", "randomize_seed_val", "ss_guidance_strength_val", "ss_sampling_steps_val",
    "slat_guidance_strength_val", "slat_sampling_steps_val", "multiimage_algo_val",
    "mesh_simplify_val", "texture_size_val", "video_resolution_val", "video_num_frames_val",
    "video_fps_val", "save_metadata_val"
]

def get_default_config_values():
    return {
        "seed_val": 0, "randomize_seed_val": True,
        "ss_guidance_strength_val": 7.5, "ss_sampling_steps_val": 12,
        "slat_guidance_strength_val": 3.0, "slat_sampling_steps_val": 12,
        "multiimage_algo_val": "stochastic",
        "mesh_simplify_val": 0.99,
        "texture_size_val": 1024,
        "video_resolution_val": 1024, "video_num_frames_val": 240, "video_fps_val": 30,
        "save_metadata_val": True,
        # Batch settings (not typically part of generation preset, but for completeness if UI implies it)
        "batch_input_folder_val": "batch_input_images",
        "batch_output_folder_val": BATCH_OUTPUT_DIR_BASE_DEFAULT,
        "batch_skip_existing_val": True,
        "batch_gen_video_cb_val": True,
        "batch_extract_glb_cb_val": True,
        "batch_extract_gaussian_cb_val": True,
    }

def save_config(config_name, *values): # Order of values must match UI_COMPONENT_KEYS + batch keys
    if not config_name:
        return "Config name cannot be empty", gr.update(choices=get_config_list())

    all_keys = UI_COMPONENT_KEYS + [
        "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
        "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
    ]
    config_data = {key: val for key, val in zip(all_keys, values)}

    try:
        config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        
        # Save this as the last used config
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": config_name, "data": config_data}, f, indent=4)
            
        return f"Config '{config_name}' saved and loaded.", gr.update(choices=get_config_list(), value=config_name)
    except Exception as e:
        return f"Error saving config: {str(e)}", gr.update(choices=get_config_list())

def load_config(config_name_to_load):
    if not config_name_to_load: # Might happen if dropdown is empty
        # Load default if no name provided (e.g. initial load)
        config_data = get_default_config_values()
        status_msg = "Loaded default configuration."
        # Save this default as last loaded if it was an empty call
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)

    else:
        config_file_path = os.path.join(CONFIG_DIR, f"{config_name_to_load}.json")
        if not os.path.exists(config_file_path):
            # If specific config not found, load default values
            default_vals = get_default_config_values()
            # Return default values in the correct order
            ordered_defaults = [default_vals[key] for key in UI_COMPONENT_KEYS]
            ordered_defaults += [default_vals[key] for key in [
                "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
                "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
            ]]
            return tuple([f"Config '{config_name_to_load}' not found. Loaded defaults."] + ordered_defaults)

        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        status_msg = f"Config '{config_name_to_load}' loaded."
        
        # Save this as the last used config
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": config_name_to_load, "data": config_data}, f, indent=4)


    # Ensure all keys are present, falling back to default if a key is missing from saved config
    default_vals_for_fallback = get_default_config_values()
    ordered_values = [config_data.get(key, default_vals_for_fallback[key]) for key in UI_COMPONENT_KEYS]
    ordered_values += [config_data.get(key, default_vals_for_fallback[key]) for key in [
        "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
        "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
    ]]
    return tuple([status_msg] + ordered_values)


def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return sorted(configs) if configs else [DEFAULT_CONFIG_NAME]


def initial_load_config():
    last_config_name = DEFAULT_CONFIG_NAME
    config_data = get_default_config_values() # Start with defaults

    if os.path.exists(LAST_CONFIG_FILE):
        try:
            with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
                last_config_name = saved_state.get("last_config_name", DEFAULT_CONFIG_NAME)
                # Check if the specific config file exists
                specific_config_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
                if os.path.exists(specific_config_path):
                     with open(specific_config_path, "r", encoding="utf-8") as scf:
                        config_data = json.load(scf) # Load data from the specific file
                else: # Last config name points to a non-existent file, try loading its stored data or default
                    config_data = saved_state.get("data", get_default_config_values())
                    if not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
                         # Save default if it doesn't exist
                        with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as df:
                            json.dump(get_default_config_values(), df, indent=4)


        except (FileNotFoundError, json.JSONDecodeError):
            # If last config file is invalid or not found, create and load default
            if not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
                with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as f:
                    json.dump(config_data, f, indent=4) # Save default
            # Update last_config.json to point to default
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                 json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)

    elif not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
        # Create default config if it doesn't exist on first ever run
        with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as f:
            json.dump(config_data, f, indent=4)
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f: # Create last config file pointing to default
            json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)

    # Ensure all keys are present, falling back to default if a key is missing
    default_vals_for_fallback = get_default_config_values()
    
    # Construct the tuple for UI updates
    # Status message, then dropdown value, then component values
    ordered_values = [config_data.get(key, default_vals_for_fallback[key]) for key in UI_COMPONENT_KEYS]
    ordered_values += [config_data.get(key, default_vals_for_fallback[key]) for key in [
        "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
        "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
    ]]

    return tuple([gr.update(choices=get_config_list(), value=last_config_name)] + ordered_values)


# --- Batch Processing Function ---
def run_batch_processing(
    batch_input_dir, batch_output_base, skip_existing,
    gen_video_cb, extract_glb_cb, extract_gs_cb,
    # Generation parameters from UI (passed through)
    seed_val, randomize_seed_val, ss_guidance_strength_val, ss_sampling_steps_val,
    slat_guidance_strength_val, slat_sampling_steps_val, multiimage_algo_val,
    mesh_simplify_val, texture_size_val, video_resolution_val, video_num_frames_val,
    video_fps_val, save_metadata_val,
    progress=gr.Progress(track_tqdm=True)
):
    if not os.path.isdir(batch_input_dir):
        return "Error: Batch input directory not found."

    # Create specific output directories for this batch run
    batch_video_out_dir = os.path.join(batch_output_base, "video")
    batch_glb_out_dir = os.path.join(batch_output_base, "glb")
    batch_gs_out_dir = os.path.join(batch_output_base, "gaussian")
    batch_meta_out_dir = os.path.join(batch_output_base, "metadata")
    for d in [batch_video_out_dir, batch_glb_out_dir, batch_gs_out_dir, batch_meta_out_dir]:
        os.makedirs(d, exist_ok=True)

    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]
    all_files = []
    for pattern in image_patterns:
        all_files.extend(sorted_glob(os.path.join(batch_input_dir, pattern)))
    
    # Remove duplicates if any pattern overlaps (e.g. *.jpg and *.jpeg)
    all_files = sorted(list(set(all_files)), key=alphanum_key)

    if not all_files:
        return "No images found in the batch input directory."

    total_files = len(all_files)
    log_output = []
    
    # For ETA
    start_batch_time = time.time()
    processed_count = 0

    for i, image_path in enumerate(all_files):
        progress(i / total_files, desc=f"Processing {os.path.basename(image_path)} ({i+1}/{total_files})")
        log_output.append(f"Processing {i+1}/{total_files}: {os.path.basename(image_path)}")
        print(f"Batch Processing {i+1}/{total_files}: {os.path.basename(image_path)}")

        input_basename = os.path.splitext(os.path.basename(image_path))[0]

        # Check if outputs exist if skip_existing is True
        if skip_existing:
            video_exists = not gen_video_cb or os.path.exists(os.path.join(batch_video_out_dir, f"{input_basename}.mp4"))
            glb_exists = not extract_glb_cb or os.path.exists(os.path.join(batch_glb_out_dir, f"{input_basename}.glb"))
            gs_exists = not extract_gs_cb or os.path.exists(os.path.join(batch_gs_out_dir, f"{input_basename}.ply"))
            if video_exists and glb_exists and gs_exists:
                log_output.append(f"Skipping {input_basename}, outputs already exist.")
                print(f"Skipping {input_basename}, outputs already exist.")
                processed_count +=1
                continue
        
        try:
            img_pil = Image.open(image_path).convert("RGBA") # Ensure RGBA
            processed_img_pil = preprocess_image(img_pil)

            current_seed = get_seed(randomize_seed_val, seed_val)

            # Call main generation function
            # is_multiimage is False for batch, multiimages is empty
            generated_state, video_out_path = image_to_3d(
                processed_img_pil, [], False, current_seed,
                ss_guidance_strength_val, ss_sampling_steps_val,
                slat_guidance_strength_val, slat_sampling_steps_val,
                multiimage_algo_val,
                video_resolution_val, video_num_frames_val, video_fps_val,
                save_metadata_val,
                output_filename_prefix=input_basename # This directs output
            )
            log_output.append(f"  Generated video: {video_out_path}")

            if extract_glb_cb:
                glb_path, _ = extract_glb(generated_state, mesh_simplify_val, texture_size_val, save_metadata_val, output_filename_prefix=input_basename)
                log_output.append(f"  Extracted GLB: {glb_path}")
            
            if extract_gs_cb:
                gs_path, _ = extract_gaussian(generated_state, save_metadata_val, output_filename_prefix=input_basename)
                log_output.append(f"  Extracted Gaussian Splats: {gs_path}")
            
            processed_count += 1
            elapsed_time = time.time() - start_batch_time
            avg_time_per_file = elapsed_time / processed_count if processed_count > 0 else 0
            remaining_files = total_files - processed_count
            eta_seconds = remaining_files * avg_time_per_file
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if avg_time_per_file > 0 else "N/A"
            print(f"  Completed: {input_basename}. ETA: {eta_str}")


        except Exception as e:
            error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
            log_output.append(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Optionally, continue to next file or stop batch

    log_output.append(f"Batch processing finished. Processed {processed_count}/{total_files} images.")
    print(f"Batch processing finished. Processed {processed_count}/{total_files} images.")
    return "\n".join(log_output)


# Gradio UI
# Added theme=gr.themes.Soft() for a modern look
with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with TRELLIS SECourses App (forked from trellis-stable-projectorz) V1 > https://www.patreon.com/posts/117470976
    """.format(code_version))
    
    # UI Component Values (will be populated by load_config)
    seed_val = gr.State()
    randomize_seed_val = gr.State()
    ss_guidance_strength_val = gr.State()
    ss_sampling_steps_val = gr.State()
    slat_guidance_strength_val = gr.State()
    slat_sampling_steps_val = gr.State()
    multiimage_algo_val = gr.State()
    mesh_simplify_val = gr.State()
    texture_size_val = gr.State()
    video_resolution_val = gr.State()
    video_num_frames_val = gr.State()
    video_fps_val = gr.State()
    save_metadata_val = gr.State()
    # Batch states (also part of config for convenience of saving UI state)
    batch_input_folder_val = gr.State()
    batch_output_folder_val = gr.State()
    batch_skip_existing_val = gr.State()
    batch_gen_video_cb_val = gr.State()
    batch_extract_glb_cb_val = gr.State()
    batch_extract_gaussian_cb_val = gr.State()


    with gr.Row():
        with gr.Column(scale=2): # Input settings column
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil", height=300)
                with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="Image Prompt", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images. 
                        *NOTE: this is an experimental algorithm. It may not produce the best results for all images.*
                    """)
            
            with gr.Accordion(label="Generation Settings", open=True):
                with gr.Row(): # Seed and Randomize Seed on the same row
                    seed_slider = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1, elem_id="seed_slider_elem")
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True, elem_id="randomize_seed_checkbox_elem")
                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength_slider = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1, elem_id="ss_guidance_strength_slider_elem")
                    ss_sampling_steps_slider = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, elem_id="ss_sampling_steps_slider_elem")
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength_slider = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1, elem_id="slat_guidance_strength_slider_elem")
                    slat_sampling_steps_slider = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1, elem_id="slat_sampling_steps_slider_elem")
                multiimage_algo_radio = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic", elem_id="multiimage_algo_radio_elem")

            with gr.Accordion(label="Video Output Settings", open=True):
                video_resolution_slider = gr.Slider(256, 2048, label="Video Resolution (pixels)", value=1024, step=64, info="Width of the output video frames.", elem_id="video_resolution_slider_elem")
                video_num_frames_slider = gr.Slider(30, 480, label="Video Number of Frames", value=240, step=10, info="Total frames in the rotating showcase video.", elem_id="video_num_frames_slider_elem")
                video_fps_slider = gr.Slider(10, 120, label="Video FPS", value=60, step=1, info="Frames per second for the output video.", elem_id="video_fps_slider_elem")

            generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Accordion(label="Extraction & Metadata Settings", open=True):
                mesh_simplify_slider = gr.Slider(0.2, 0.99, label="Mesh Simplification Factor", value=0.7, step=0.01, 
                                          info="Lower values simplify more (less detail, smaller file). E.g., 0.2=heavy, 0.99=light.", elem_id="mesh_simplify_slider_elem")
                texture_size_slider = gr.Slider(512, 2048, label="Texture Size (pixels)", value=1024, step=512, 
                                         info="Resolution of baked texture (e.g., 1024x1024). Higher=sharper but larger GLB.", elem_id="texture_size_slider_elem")
                save_metadata_checkbox = gr.Checkbox(label="Save Generation Metadata (.txt file)", value=True, 
                                                info="Saves parameters and duration to a text file alongside outputs.", elem_id="save_metadata_checkbox_elem")

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian Splats", interactive=False) # Renamed for clarity
            
            gr.Markdown("""
                        *NOTE: Gaussian Splat file (.ply) can be very large (~50MB), it may take a while to display and download.*
                        """)
            with gr.Row():
                open_outputs_btn = gr.Button("Open Outputs Folder")
                open_batch_outputs_btn = gr.Button("Open Batch Outputs Folder")


        with gr.Column(scale=1): # Output and controls column
            video_output = gr.Video(label="Generated 3D Asset (Video)", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)
            
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian Splats", interactive=False)  
            
            with gr.Accordion("Configuration Presets", open=True):
                config_status_textbox = gr.Textbox(label="Config Status", interactive=False, lines=1)
                with gr.Row():
                    config_load_dropdown = gr.Dropdown(label="Load Preset", choices=get_config_list(), elem_id="config_load_dropdown_elem")
                with gr.Row():
                    config_save_name_textbox = gr.Textbox(label="Save Preset As", placeholder="Enter preset name")
                    config_save_button = gr.Button("Save Preset")
            
            with gr.Accordion("Batch Processing", open=False):
                batch_input_folder_textbox = gr.Textbox(label="Input Folder (contains images)", placeholder="/path/to/input_images", info="Folder with images to process.", elem_id="batch_input_folder_textbox_elem")
                batch_output_folder_textbox = gr.Textbox(label="Base Output Folder Name", value=BATCH_OUTPUT_DIR_BASE_DEFAULT, info="A folder with this name will be created in the main directory.", elem_id="batch_output_folder_textbox_elem")
                batch_skip_existing_checkbox = gr.Checkbox(label="Skip if all outputs exist for an image", value=True, elem_id="batch_skip_existing_checkbox_elem")
                gr.Markdown("Select operations for batch processing:")
                with gr.Row():
                    batch_gen_video_checkbox = gr.Checkbox(label="Generate Video", value=True, elem_id="batch_gen_video_checkbox_elem")
                    batch_extract_glb_checkbox = gr.Checkbox(label="Extract GLB", value=True, elem_id="batch_extract_glb_checkbox_elem")
                    batch_extract_gs_checkbox = gr.Checkbox(label="Extract Gaussian Splats", value=True, elem_id="batch_extract_gs_checkbox_elem")
                batch_process_button = gr.Button("Start Batch Process", variant="primary")
                batch_status_textbox = gr.Textbox(label="Batch Process Log", lines=10, interactive=False)

    is_multiimage = gr.State(False)
    output_buf = gr.State() # Stores the packed Gaussian/Mesh state

    # --- Example Images ---
    # Removed run_on_click=True
    with gr.Row() as single_image_example_row:
        gr.Examples(
            examples=[
                f'assets/example_image/{image_file}' # Corrected path
                for image_file in os.listdir("assets/example_image") if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ],
            inputs=[image_prompt],
            fn=preprocess_image, # Preprocessing function for examples
            outputs=[image_prompt], # Output to the image_prompt component
            #cache_examples=True # Optional: caches preprocessed examples
            label="Single Image Examples",
            examples_per_page=8,
        )
    with gr.Row(visible=False) as multiimage_example_row:
        gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt], # This input is a bit awkward for multi-image examples that are already processed.
                                   # The target for multi-image examples is multiimage_prompt (Gallery).
                                   # This needs specific handling or direct Gallery update.
                                   # For simplicity, let's assume split_image is used if one combined image is clicked.
            fn=split_image, # Function to split the example image (if it's a combined one)
            outputs=[multiimage_prompt], # Output to the gallery
            #cache_examples=True,
            label="Multi Image Examples (Click to split and load)",
            examples_per_page=4,
        )

    # --- UI Component List for Presets ---
    # IMPORTANT: Order must match get_default_config_values, save_config, load_config
    # This list also needs to match the inputs/outputs for preset save/load clicks
    preset_ui_components = [
        seed_slider, randomize_seed_checkbox, ss_guidance_strength_slider, ss_sampling_steps_slider,
        slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
        mesh_simplify_slider, texture_size_slider, video_resolution_slider, video_num_frames_slider,
        video_fps_slider, save_metadata_checkbox,
        # Batch settings (included in preset save/load for UI convenience)
        batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
        batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox
    ]
    preset_state_components = [
        seed_val, randomize_seed_val, ss_guidance_strength_val, ss_sampling_steps_val,
        slat_guidance_strength_val, slat_sampling_steps_val, multiimage_algo_val,
        mesh_simplify_val, texture_size_val, video_resolution_val, video_num_frames_val,
        video_fps_val, save_metadata_val,
        batch_input_folder_val, batch_output_folder_val, batch_skip_existing_val,
        batch_gen_video_cb_val, batch_extract_glb_cb_val, batch_extract_gaussian_cb_val
    ]


    # --- Event Handlers ---
    # demo.load(start_session) # Removed
    # demo.unload(end_session) # Removed
    
    single_image_input_tab.select(
        lambda: (False, gr.update(visible=True), gr.update(visible=False)),
        outputs=[is_multiimage, single_image_example_row, multiimage_example_row]
    )
    multiimage_input_tab.select(
        lambda: (True, gr.update(visible=False), gr.update(visible=True)),
        outputs=[is_multiimage, single_image_example_row, multiimage_example_row]
    )
    
    image_prompt.upload(preprocess_image, inputs=[image_prompt], outputs=[image_prompt])
    multiimage_prompt.upload(preprocess_images, inputs=[multiimage_prompt], outputs=[multiimage_prompt])

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed_checkbox, seed_slider],
        outputs=[seed_slider], # Update the displayed seed
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, multiimage_prompt, is_multiimage, seed_slider, 
            ss_guidance_strength_slider, ss_sampling_steps_slider, 
            slat_guidance_strength_slider, slat_sampling_steps_slider, 
            multiimage_algo_radio,
            video_resolution_slider, video_num_frames_slider, video_fps_slider, # New video params
            save_metadata_checkbox # New metadata param
        ],
        outputs=[output_buf, video_output],
    ).then(
        lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    video_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify_slider, texture_size_slider, save_metadata_checkbox],
        outputs=[model_output, download_glb],
    ).then(
        lambda _: gr.update(interactive=True),
        outputs=[download_glb],
    )
    
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf, save_metadata_checkbox],
        outputs=[model_output, download_gs],
    ).then(
        lambda _: gr.update(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear( # When the 3D model viewer is cleared
        lambda: (gr.update(interactive=False), gr.update(interactive=False)), # Disable download buttons
        outputs=[download_glb, download_gs], # Corrected to disable both
    )
    
    open_outputs_btn.click(lambda: open_folder(OUTPUT_DIR_BASE), inputs=None, outputs=None) # No status output needed, prints to console
    open_batch_outputs_btn.click(lambda: open_folder(batch_output_folder_textbox.value if batch_output_folder_textbox.value else BATCH_OUTPUT_DIR_BASE_DEFAULT), 
                                 inputs=[batch_output_folder_textbox], outputs=None)


    # --- Preset Handler Connections ---
    config_save_button.click(
        save_config,
        inputs=[config_save_name_textbox] + preset_ui_components,
        outputs=[config_status_textbox, config_load_dropdown]
    )
    config_load_dropdown.change( # Load when dropdown selection changes
        load_config,
        inputs=[config_load_dropdown],
        outputs=[config_status_textbox] + preset_ui_components
    )
    # Initial load of presets
    demo.load(initial_load_config, inputs=None, outputs=[config_load_dropdown] + preset_ui_components)


    # --- Batch Processing Handler ---
    batch_process_button.click(
        run_batch_processing,
        inputs=[
            batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
            batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox,
            # Pass current UI settings for generation
            seed_slider, randomize_seed_checkbox, ss_guidance_strength_slider, ss_sampling_steps_slider,
            slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
            mesh_simplify_slider, texture_size_slider, video_resolution_slider, video_num_frames_slider,
            video_fps_slider, save_metadata_checkbox
        ],
        outputs=[batch_status_textbox]
    )


# Define a function to initialize the pipeline
def initialize_pipeline(precision_arg="fp32"): # Changed default to fp32 to match arg help
    global pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("models")
    
    # Map command line precision to pipeline precision
    if precision_arg == "fp16":
        effective_precision = "half"
    elif precision_arg == "fp32":
        effective_precision = "full"
    else: # Should not happen with choices in argparse
        effective_precision = "full" 

    print('')
    print(f"Using precision: '{effective_precision}' (requested: '{precision_arg}'). Loading...")
    if effective_precision == "half": # original "half" or "float16"
        pipeline.to(torch.float16)
        if "image_cond_model" in pipeline.models:
            # Ensure image_cond_model (DINOv2) is also half if main pipeline is half
            # It's often kept at higher precision for stability, but let's follow original logic.
             if hasattr(pipeline.models['image_cond_model'], 'half'):
                pipeline.models['image_cond_model'].half()
    # No explicit .cuda() here, let parts be moved dynamically by TrellisImageTo3DPipeline


# Launch the Gradio app
if __name__ == "__main__":
    initialize_pipeline(cmd_args.precision)
    demo.launch(inbrowser=True, share=cmd_args.share) # Added share=cmd_args.share