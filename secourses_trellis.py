
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

# Global flag for cancellation
CANCEL_REQUESTED = False

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
    safe_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', prefix)
    
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
                    return final_path, temp_reservation_path, filename_base 
                except IOError as e:
                    print(f"Warning: IOError creating temp reservation file {temp_reservation_path}: {e}")
            counter += 1
            if counter > 99999: 
                raise RuntimeError(f"Could not find an available numeric filename in {output_dir} with prefix {prefix}.")

def remove_temp_reservation_file(temp_file_path):
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except OSError as e:
            print(f"Warning: Could not remove temp reservation file {temp_file_path}: {e}")

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
        elif platform.system() == "Darwin": 
            subprocess.run(["open", path])
        else: 
            subprocess.run(["xdg-open", path])
        return f"Opened folder: {path}"
    except Exception as e:
        print(f"Error opening folder {path}: {e}")
        return f"Error opening folder: {e}"

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
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict]: 
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
    video_resolution: int,
    video_num_frames: int,
    video_fps: int,
    save_metadata: bool,
    output_filename_prefix: Optional[str] = None,
) -> Tuple[dict, str]:
    
    start_time = time.time()

    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False, 
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
            [img[0] for img in multiimages], 
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False, 
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
    
    video_frames_color = render_utils.render_video(outputs['gaussian'][0], resolution=video_resolution, bg_color=(0,0,0), num_frames=video_num_frames)['color']
    video_frames_geo = render_utils.render_video(outputs['mesh'][0], resolution=video_resolution, bg_color=(0,0,0), num_frames=video_num_frames)['normal']
    
    combined_video_frames = [np.concatenate([video_frames_color[i], video_frames_geo[i]], axis=1) for i in range(len(video_frames_color))]
    
    actual_video_path = ""
    video_filename_base_final = "" # This will be stored in state
    temp_reservation_path = None

    if output_filename_prefix: 
        video_filename_base_final = output_filename_prefix
        actual_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{video_filename_base_final}.mp4")
        # Ensure directory for batch subfolder exists (if prefix contains path separators, common in batch)
        os.makedirs(os.path.dirname(actual_video_path), exist_ok=True) 
    else: 
        # For single generations (1 or N), get_next_output_path_numeric provides the base NNNN
        # Any iteration suffix (like _0001) should have been incorporated into output_filename_prefix by the caller if N > 1.
        # If output_filename_prefix is None, it means it's a single (1-shot) generation.
        actual_video_path, temp_reservation_path, video_filename_base_final = get_next_output_path_numeric(OUTPUT_VIDEO_DIR, "mp4")

    imageio.mimsave(actual_video_path, combined_video_frames, fps=video_fps)
    if temp_reservation_path: # Only exists if get_next_output_path_numeric was called (i.e. not batch/N-gen with prefix)
        remove_temp_reservation_file(temp_reservation_path)
        
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    state['filename_base'] = video_filename_base_final # This base now includes any N-gen iter suffixes
    
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
            "output_filename_base": video_filename_base_final,
        }
        
        metadata_path = os.path.join(OUTPUT_METADATA_DIR, f"{video_filename_base_final}.txt")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True) # Ensure subfolder for batch
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    torch.cuda.empty_cache()
    return state, actual_video_path


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    save_metadata: bool, 
    # output_filename_prefix: Optional[str] = None, # Not needed, uses state['filename_base']
) -> Tuple[str, str]: 
    gs, mesh = unpack_state(state)
    
    vertex_count = mesh.vertices.shape[0]
    face_count = mesh.faces.shape[0]
    print(f"Mesh stats - Vertices: {vertex_count}, Triangles: {face_count}")
    
    # GLB filename is derived from the state, which already includes any iteration suffixes
    glb_filename_base = state.get('filename_base', '')
    if not glb_filename_base: # Fallback, should ideally not happen
        # This fallback would generate a new NNNN, not related to the video's NNNN.
        # It's better to error or ensure filename_base is always in state.
        print("Error: filename_base not found in state for GLB extraction. Generating new name.")
        _, _, glb_filename_base = get_next_output_path_numeric(OUTPUT_GLB_DIR, "glb")
        # No temp_reservation_path to clean here as it's a fallback.

    actual_glb_path = os.path.join(OUTPUT_GLB_DIR, f"{glb_filename_base}.glb")
    os.makedirs(os.path.dirname(actual_glb_path), exist_ok=True) # For batch subfolders

    glb_data = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_data.export(actual_glb_path)

    # No temp_reservation_path to remove here as it's based on state['filename_base']

    if save_metadata and glb_filename_base: 
        metadata_path_check = os.path.join(OUTPUT_METADATA_DIR, f"{glb_filename_base}.txt")
        if os.path.exists(metadata_path_check):
            try:
                with open(metadata_path_check, 'r') as f:
                    metadata = json.load(f)
                metadata["vertex_count"] = int(vertex_count)
                metadata["triangle_count"] = int(face_count)
                metadata["mesh_simplify_factor"] = mesh_simplify
                metadata["texture_size"] = texture_size
                with open(metadata_path_check, 'w') as f:
                    json.dump(metadata, f, indent=4)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not update metadata with mesh stats: {e}")
        else:
            print(f"Warning: Metadata file {metadata_path_check} not found for GLB, cannot update stats.")


    torch.cuda.empty_cache()
    return actual_glb_path, actual_glb_path


def extract_gaussian(
    state: dict, 
    save_metadata: bool, 
    # output_filename_prefix: Optional[str] = None, # Not needed, uses state['filename_base']
) -> Tuple[str, str]:
    gs, _ = unpack_state(state)
    
    gs_filename_base = state.get('filename_base', '')
    if not gs_filename_base: 
        print("Error: filename_base not found in state for Gaussian extraction. Generating new name.")
        _, _, gs_filename_base = get_next_output_path_numeric(OUTPUT_GAUSSIAN_DIR, "ply")

    actual_gaussian_path = os.path.join(OUTPUT_GAUSSIAN_DIR, f"{gs_filename_base}.ply")
    os.makedirs(os.path.dirname(actual_gaussian_path), exist_ok=True) # For batch subfolders

    gs.save_ply(actual_gaussian_path)
    
    # No temp_reservation_path to remove here

    if save_metadata and gs_filename_base:
        # Gaussian specific metadata could be added here if needed, similar to GLB
        pass

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
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


# --- Preset Functions ---
UI_COMPONENT_KEYS = [
    "seed_val", "randomize_seed_val", "ss_guidance_strength_val", "ss_sampling_steps_val",
    "slat_guidance_strength_val", "slat_sampling_steps_val", "multiimage_algo_val",
    "num_generations_val", # Added
    "mesh_simplify_val", "texture_size_val", "video_resolution_val", "video_num_frames_val",
    "video_fps_val", "save_metadata_val"
]

def get_default_config_values():
    return {
        "seed_val": 0, "randomize_seed_val": True,
        "ss_guidance_strength_val": 7.5, "ss_sampling_steps_val": 20,
        "slat_guidance_strength_val": 3.0, "slat_sampling_steps_val": 20,
        "multiimage_algo_val": "stochastic",
        "num_generations_val": 1, # Added
        "mesh_simplify_val": 0.9,
        "texture_size_val": 1024,
        "video_resolution_val": 1024, "video_num_frames_val": 240, "video_fps_val": 30,
        "save_metadata_val": True,
        "batch_input_folder_val": "batch_input_images",
        "batch_output_folder_val": BATCH_OUTPUT_DIR_BASE_DEFAULT,
        "batch_skip_existing_val": True,
        "batch_gen_video_cb_val": True,
        "batch_extract_glb_cb_val": True,
        "batch_extract_gaussian_cb_val": True,
    }

def save_config(config_name, *values): 
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
        
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": config_name, "data": config_data}, f, indent=4)
            
        return f"Config '{config_name}' saved and loaded.", gr.update(choices=get_config_list(), value=config_name)
    except Exception as e:
        return f"Error saving config: {str(e)}", gr.update(choices=get_config_list())

def load_config(config_name_to_load):
    if not config_name_to_load: 
        config_data = get_default_config_values()
        status_msg = "Loaded default configuration."
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)
    else:
        config_file_path = os.path.join(CONFIG_DIR, f"{config_name_to_load}.json")
        if not os.path.exists(config_file_path):
            default_vals = get_default_config_values()
            ordered_defaults = [default_vals[key] for key in UI_COMPONENT_KEYS]
            ordered_defaults += [default_vals[key] for key in [
                "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
                "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
            ]]
            return tuple([f"Config '{config_name_to_load}' not found. Loaded defaults."] + ordered_defaults)

        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        status_msg = f"Config '{config_name_to_load}' loaded."
        
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_config_name": config_name_to_load, "data": config_data}, f, indent=4)

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
    config_data = get_default_config_values() 

    if os.path.exists(LAST_CONFIG_FILE):
        try:
            with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
                last_config_name = saved_state.get("last_config_name", DEFAULT_CONFIG_NAME)
                specific_config_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
                if os.path.exists(specific_config_path):
                     with open(specific_config_path, "r", encoding="utf-8") as scf:
                        config_data = json.load(scf) 
                else: 
                    config_data = saved_state.get("data", get_default_config_values())
                    if not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
                        with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as df:
                            json.dump(get_default_config_values(), df, indent=4)
        except (FileNotFoundError, json.JSONDecodeError):
            if not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
                with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as f:
                    json.dump(config_data, f, indent=4) 
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                 json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)
    elif not os.path.exists(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")):
        with open(os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json"), "w") as f:
            json.dump(config_data, f, indent=4)
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f: 
            json.dump({"last_config_name": DEFAULT_CONFIG_NAME, "data": config_data}, f, indent=4)

    default_vals_for_fallback = get_default_config_values()
    
    ordered_values = [config_data.get(key, default_vals_for_fallback[key]) for key in UI_COMPONENT_KEYS]
    ordered_values += [config_data.get(key, default_vals_for_fallback[key]) for key in [
        "batch_input_folder_val", "batch_output_folder_val", "batch_skip_existing_val",
        "batch_gen_video_cb_val", "batch_extract_glb_cb_val", "batch_extract_gaussian_cb_val"
    ]]

    return tuple([gr.update(choices=get_config_list(), value=last_config_name)] + ordered_values)

# --- Core Generation Loop for UI ---
def perform_generations_and_optional_extractions(
    num_gens_from_slider, # str or float from gr.Number
    extract_models_after_each, # Boolean
    # UI inputs for generation
    image_prompt_pil, multiimages_list, is_multiimage_bool,
    initial_seed_int, randomize_seed_bool,
    ss_guidance_strength_float, ss_sampling_steps_int,
    slat_guidance_strength_float, slat_sampling_steps_int,
    multiimage_algo_str,
    video_resolution_int, video_num_frames_int, video_fps_int,
    save_metadata_bool,
    mesh_simplify_float, texture_size_int,
    progress=gr.Progress(track_tqdm=True)
):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False # Reset at the start of a new task sequence

    try:
        num_gens = int(float(num_gens_from_slider))
    except ValueError:
        num_gens = 1
    if num_gens <= 0: num_gens = 1

    base_NNNN_for_series = None
    temp_reservation_to_clean_after_series = None

    # For N-generations (num_gens > 1) from the UI (not batch), we need a common base NNNN prefix.
    if num_gens > 1:
        try:
            # This reserves "base_NNNN_for_series.mp4.tmp"
            _, temp_reservation_to_clean_after_series, base_NNNN_for_series = get_next_output_path_numeric(OUTPUT_VIDEO_DIR, "mp4", prefix="")
        except RuntimeError as e:
            error_msg = f"Error getting unique series base name: {e}"
            print(error_msg)
            return None, None, None, gr.update(value=None, interactive=False), gr.update(value=None, interactive=False), gr.update(interactive=False), gr.update(interactive=False), error_msg


    current_seed = initial_seed_int
    
    last_generated_state = None
    last_video_path = None
    last_glb_path = None 
    last_gs_path = None 
    
    all_outputs_log = []

    for i in range(num_gens):
        if CANCEL_REQUESTED:
            msg = f"Generation {i+1}/{num_gens} cancelled by user request."
            all_outputs_log.append(msg)
            print(msg)
            break # Exit the loop
        
        progress((i) / num_gens, desc=f"Processing Generation {i+1}/{num_gens}")
        
        seed_for_this_iter = get_seed(randomize_seed_bool, current_seed)
        if not randomize_seed_bool:
            current_seed += 1
        
        iter_output_filename_prefix = None
        if num_gens > 1:
            iter_suffix = f"_{i+1:04d}"
            if base_NNNN_for_series: 
                iter_output_filename_prefix = base_NNNN_for_series + iter_suffix
            else: # Should not happen if num_gens > 1 due to logic above, but as a fallback:
                # This would mean each of N gens gets a *new* NNNN prefix, not sharing one.
                # This path is unlikely given the base_NNNN_for_series logic.
                pass 
        # If num_gens == 1, iter_output_filename_prefix remains None.
        # image_to_3d will then use its internal get_next_output_path_numeric.

        try:
            generated_state_iter, video_path_iter = image_to_3d(
                image_prompt_pil, multiimages_list, is_multiimage_bool, seed_for_this_iter,
                ss_guidance_strength_float, ss_sampling_steps_int,
                slat_guidance_strength_float, slat_sampling_steps_int,
                multiimage_algo_str,
                video_resolution_int, video_num_frames_int, video_fps_int,
                save_metadata_bool,
                output_filename_prefix=iter_output_filename_prefix 
            )
            last_generated_state = generated_state_iter
            last_video_path = video_path_iter
            all_outputs_log.append(f"Gen {i+1}/{num_gens}: Video '{os.path.basename(video_path_iter)}' created.")

            if extract_models_after_each and last_generated_state:
                if CANCEL_REQUESTED: break # Check before potentially long extraction
                glb_path_iter, _ = extract_glb(last_generated_state, mesh_simplify_float, texture_size_int, save_metadata_bool)
                last_glb_path = glb_path_iter 
                all_outputs_log.append(f"  Gen {i+1}: GLB '{os.path.basename(glb_path_iter)}' extracted.")
                
                if CANCEL_REQUESTED: break 
                gs_path_iter, _ = extract_gaussian(last_generated_state, save_metadata_bool)
                last_gs_path = gs_path_iter 
                all_outputs_log.append(f"  Gen {i+1}: Gaussian '{os.path.basename(gs_path_iter)}' extracted.")
                
        except Exception as e:
            error_msg = f"Error during generation {i+1}/{num_gens}: {str(e)}"
            all_outputs_log.append(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            if num_gens > 1: 
                all_outputs_log.append("Continuing to next generation if any.")
                continue 
            else: 
                break 
        
        # Yield intermediate results to update UI gradually (optional, adds complexity)
        # For now, updates UI at the end with the last successful set.

    if temp_reservation_to_clean_after_series: 
        remove_temp_reservation_file(temp_reservation_to_clean_after_series)
        
    if CANCEL_REQUESTED:
        all_outputs_log.append("Task was cancelled by user.")
        CANCEL_REQUESTED = False # Reset the flag

    model_to_show_in_ui = None
    if extract_models_after_each: # Only show model if auto-extraction was on
        if last_glb_path: model_to_show_in_ui = last_glb_path
        elif last_gs_path: model_to_show_in_ui = last_gs_path

    enable_extract_buttons = gr.update(interactive=True) if last_generated_state else gr.update(interactive=False)
    
    # Prepare download button updates (active only if corresponding file was made)
    download_glb_val = gr.update(value=last_glb_path, interactive=bool(last_glb_path)) if extract_models_after_each else gr.update(value=None, interactive=False)
    download_gs_val = gr.update(value=last_gs_path, interactive=bool(last_gs_path)) if extract_models_after_each else gr.update(value=None, interactive=False)
    
    return (
        last_generated_state, 
        last_video_path,      
        model_to_show_in_ui,  
        download_glb_val,    
        download_gs_val,     
        enable_extract_buttons, 
        enable_extract_buttons, 
        "\n".join(all_outputs_log) 
    )

# --- Batch Processing Function ---
def run_batch_processing(
    num_generations_from_slider, # From UI
    batch_input_dir, batch_output_base_name, skip_existing,
    gen_video_cb, extract_glb_cb, extract_gs_cb,
    seed_val, randomize_seed_val, ss_guidance_strength_val, ss_sampling_steps_val,
    slat_guidance_strength_val, slat_sampling_steps_val, multiimage_algo_val,
    mesh_simplify_val, texture_size_val, video_resolution_val, video_num_frames_val,
    video_fps_val, save_metadata_val,
    progress=gr.Progress(track_tqdm=True)
):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False 

    if not os.path.isdir(batch_input_dir):
        return "Error: Batch input directory not found."

    # Use the provided batch_output_base_name to create a subfolder in the main directory
    batch_output_dir_specific = os.path.join(os.getcwd(), batch_output_base_name)

    batch_video_out_dir = os.path.join(batch_output_dir_specific, "video")
    batch_glb_out_dir = os.path.join(batch_output_dir_specific, "glb")
    batch_gs_out_dir = os.path.join(batch_output_dir_specific, "gaussian")
    batch_meta_out_dir = os.path.join(batch_output_dir_specific, "metadata")
    for d_path in [batch_output_dir_specific, batch_video_out_dir, batch_glb_out_dir, batch_gs_out_dir, batch_meta_out_dir]:
        os.makedirs(d_path, exist_ok=True)

    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]
    all_files = []
    for pattern in image_patterns:
        all_files.extend(sorted_glob(os.path.join(batch_input_dir, pattern)))
    
    all_files = sorted(list(set(all_files)), key=alphanum_key)

    if not all_files:
        return "No images found in the batch input directory."

    try:
        num_gens_per_image = int(float(num_generations_from_slider))
    except ValueError:
        num_gens_per_image = 1
    if num_gens_per_image <=0: num_gens_per_image = 1

    total_images = len(all_files)
    total_iterations = total_images * num_gens_per_image
    log_output = [f"Starting batch processing: {total_images} images, {num_gens_per_image} generation(s) per image. Total iterations: {total_iterations}"]
    
    start_batch_time = time.time()
    processed_iterations_count = 0

    for i, image_path in enumerate(all_files):
        if CANCEL_REQUESTED:
            log_output.append("Batch processing cancelled by user request.")
            break 
        
        input_basename = os.path.splitext(os.path.basename(image_path))[0]
        current_seed_for_file = seed_val # Reset seed for each new file based on UI initial seed

        for iter_k in range(num_gens_per_image):
            if CANCEL_REQUESTED:
                log_output.append(f"Cancellation requested during processing of {input_basename} (iteration {iter_k+1}).")
                break # Breaks inner loop (iterations for current file)

            processed_iterations_count += 1
            progress(processed_iterations_count / total_iterations, desc=f"Img {i+1}/{total_images}, Gen {iter_k+1}/{num_gens_per_image} ({input_basename})")
            
            iter_suffix = f"_{iter_k+1:04d}" if num_gens_per_image > 1 else ""
            current_output_prefix = input_basename + iter_suffix
            
            log_output.append(f"Processing: {current_output_prefix} (Overall iter {processed_iterations_count}/{total_iterations})")
            print(f"Batch Processing: {current_output_prefix} (Overall iter {processed_iterations_count}/{total_iterations})")

            if skip_existing:
                # Check for existence of all requested outputs for *this specific iteration*
                video_target_path = os.path.join(batch_video_out_dir, f"{current_output_prefix}.mp4")
                glb_target_path = os.path.join(batch_glb_out_dir, f"{current_output_prefix}.glb")
                gs_target_path = os.path.join(batch_gs_out_dir, f"{current_output_prefix}.ply")

                video_needed_exists = not gen_video_cb or os.path.exists(video_target_path)
                glb_needed_exists = not extract_glb_cb or os.path.exists(glb_target_path)
                gs_needed_exists = not extract_gs_cb or os.path.exists(gs_target_path)

                if video_needed_exists and glb_needed_exists and gs_needed_exists:
                    log_output.append(f"  Skipping {current_output_prefix}, all requested outputs already exist.")
                    print(f"  Skipping {current_output_prefix}, outputs exist.")
                    continue # Skip this iteration
            
            try:
                img_pil = Image.open(image_path).convert("RGBA") 
                processed_img_pil = preprocess_image(img_pil)

                seed_for_this_iteration = get_seed(randomize_seed_val, current_seed_for_file)
                if not randomize_seed_val:
                    current_seed_for_file += 1

                generated_state, video_out_path = image_to_3d(
                    processed_img_pil, [], False, seed_for_this_iteration,
                    ss_guidance_strength_val, ss_sampling_steps_val,
                    slat_guidance_strength_val, slat_sampling_steps_val,
                    multiimage_algo_val,
                    video_resolution_val, video_num_frames_val, video_fps_val,
                    save_metadata_val,
                    output_filename_prefix=current_output_prefix 
                )
                log_output.append(f"  Generated video: {video_out_path}")

                if extract_glb_cb:
                    if CANCEL_REQUESTED: break
                    glb_path, _ = extract_glb(generated_state, mesh_simplify_val, texture_size_val, save_metadata_val)
                    log_output.append(f"  Extracted GLB: {glb_path}")
                
                if extract_gs_cb:
                    if CANCEL_REQUESTED: break
                    gs_path, _ = extract_gaussian(generated_state, save_metadata_val)
                    log_output.append(f"  Extracted Gaussian Splats: {gs_path}")
                
                elapsed_time = time.time() - start_batch_time
                avg_time_per_iter = elapsed_time / processed_iterations_count if processed_iterations_count > 0 else 0
                remaining_iters = total_iterations - processed_iterations_count
                eta_seconds = remaining_iters * avg_time_per_iter
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if avg_time_per_iter > 0 else "N/A"
                print(f"  Completed: {current_output_prefix}. ETA: {eta_str}")

            except Exception as e:
                error_msg = f"Error processing {current_output_prefix}: {str(e)}"
                log_output.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
        
        if CANCEL_REQUESTED: # Check after inner loop too, to break outer file loop
            break
    
    final_msg = f"Batch processing finished. Processed {processed_iterations_count}/{total_iterations} iterations."
    if CANCEL_REQUESTED:
        final_msg = f"Batch processing cancelled by user. Processed {processed_iterations_count}/{total_iterations} iterations before stopping."
        CANCEL_REQUESTED = False # Reset 
    
    log_output.append(final_msg)
    print(final_msg)
    return "\n".join(log_output)


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with TRELLIS SECourses App (forked from trellis-stable-projectorz) V4 > https://www.patreon.com/posts/117470976
    """.format(code_version))
    
    seed_val = gr.State()
    randomize_seed_val = gr.State()
    ss_guidance_strength_val = gr.State()
    ss_sampling_steps_val = gr.State()
    slat_guidance_strength_val = gr.State()
    slat_sampling_steps_val = gr.State()
    multiimage_algo_val = gr.State()
    num_generations_val = gr.State() # Added
    mesh_simplify_val = gr.State()
    texture_size_val = gr.State()
    video_resolution_val = gr.State()
    video_num_frames_val = gr.State()
    video_fps_val = gr.State()
    save_metadata_val = gr.State()
    batch_input_folder_val = gr.State()
    batch_output_folder_val = gr.State()
    batch_skip_existing_val = gr.State()
    batch_gen_video_cb_val = gr.State()
    batch_extract_glb_cb_val = gr.State()
    batch_extract_gaussian_cb_val = gr.State()

    with gr.Row():
        with gr.Column(scale=1): 
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil", height=512)
                                
                with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=512, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images. 
                        *NOTE: this is an experimental algorithm. It may not produce the best results for all images.*
                    """)
            
            with gr.Row(): # Buttons moved slightly up
                generate_btn = gr.Button("Generate Video(s)", variant="secondary") # Renamed for clarity with N-gens
                generate_and_extract_btn = gr.Button("Generate and Extract All", variant="primary")
                cancel_button = gr.Button("Request Cancel")

            general_status_textbox = gr.Textbox(label="Task Status Log", lines=3, interactive=False, autoscroll=True, max_lines=10)
            
            with gr.Accordion(label="Generation Settings", open=True):
                with gr.Row(): 
                    seed_slider = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1, elem_id="seed_slider_elem")
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True, elem_id="randomize_seed_checkbox_elem")
                
                num_generations_slider = gr.Number(label="Number of Generations", value=1, minimum=1, step=1, precision=0, 
                                                   info="Number of times to generate. For non-random seed, seed increments. For batch, applies to each image.", 
                                                   elem_id="num_generations_slider_elem")

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
                with gr.Row():
                    video_resolution_slider = gr.Slider(256, 2048, label="Video Resolution (pixels)", value=1024, step=64, info="Width of the output video frames.", elem_id="video_resolution_slider_elem")
                    video_num_frames_slider = gr.Slider(30, 480, label="Video Number of Frames", value=240, step=10, info="Total frames in the rotating showcase video.", elem_id="video_num_frames_slider_elem")
                    video_fps_slider = gr.Slider(10, 120, label="Video FPS", value=60, step=1, info="Frames per second for the output video.", elem_id="video_fps_slider_elem")
            
            with gr.Accordion(label="Extraction & Metadata Settings", open=True):
                mesh_simplify_slider = gr.Slider(0.2, 0.99, label="Mesh Simplification Factor", value=0.9, step=0.01, 
                                          info="If you make this lower, it will generate a bigger size Vertices + Triangles mesh file.", elem_id="mesh_simplify_slider_elem")
                texture_size_slider = gr.Slider(512, 2048, label="Texture Size (pixels)", value=1024, step=512, 
                                         info="Resolution of baked texture (e.g., 1024x1024). Higher=sharper but larger GLB.", elem_id="texture_size_slider_elem")
                save_metadata_checkbox = gr.Checkbox(label="Save Generation Metadata (.txt file)", value=True, 
                                                info="Saves parameters and duration to a text file alongside outputs.", elem_id="save_metadata_checkbox_elem")
            
            gr.Markdown("""
                        *NOTE: Gaussian Splat file (.ply) can be very large (~50MB), it may take a while to display and download.*
                        """)
            with gr.Row():
                open_outputs_btn = gr.Button("Open Outputs Folder")
                open_batch_outputs_btn = gr.Button("Open Batch Outputs Folder")

            with gr.Row() as single_image_example_row:
                gr.Examples(
                    examples=[f'assets/example_image/{image_file}' for image_file in os.listdir("assets/example_image") if image_file.lower().endswith(('.png', '.jpg', '.jpeg'))],
                    inputs=[image_prompt], fn=preprocess_image, outputs=[image_prompt], 
                    label="Single Image Examples", examples_per_page=80, run_on_click=True,
                )
            with gr.Row(visible=False) as multiimage_example_row:
                gr.Examples(
                    examples=prepare_multi_example(), inputs=[image_prompt], fn=split_image,
                    outputs=[multiimage_prompt], label="Multi Image Examples", examples_per_page=40, run_on_click=True,
                )
                gr.Markdown("Click an example above to load multiple views. The combined image will be split.")


        with gr.Column(scale=1): 
            video_output = gr.Video(label="Generated 3D Asset (Video - shows last from N-gen)", autoplay=True, loop=True, height=512)
            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB (from last gen)", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian (from last gen)", interactive=False) 
            model_output = LitModel3D(label="Extracted GLB/Gaussian (shows last from N-gen)", exposure=10.0, height=512)
            
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB (last)", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian (last)", interactive=False)  
            
            with gr.Accordion("Configuration Presets", open=True):
                config_status_textbox = gr.Textbox(label="Config Status", interactive=False, lines=1)
                with gr.Row():
                    config_load_dropdown = gr.Dropdown(label="Load Preset", choices=get_config_list(), elem_id="config_load_dropdown_elem")
                with gr.Row():
                    config_save_name_textbox = gr.Textbox(label="Save Preset As", placeholder="Enter preset name")
                    config_save_button = gr.Button("Save Preset")
            
            with gr.Accordion("Batch Processing", open=True):
                batch_input_folder_textbox = gr.Textbox(label="Input Folder (contains images)", placeholder="/path/to/input_images", info="Folder with images to process.", elem_id="batch_input_folder_textbox_elem")
                batch_output_folder_textbox = gr.Textbox(label="Base Output Folder Name", value=BATCH_OUTPUT_DIR_BASE_DEFAULT, info="A folder with this name will be created in the main directory.", elem_id="batch_output_folder_textbox_elem")
                batch_skip_existing_checkbox = gr.Checkbox(label="Skip if all outputs exist for an image iteration", value=True, elem_id="batch_skip_existing_checkbox_elem")
                gr.Markdown("Select operations for batch processing:")
                with gr.Row():
                    batch_gen_video_checkbox = gr.Checkbox(label="Generate Video", value=True, elem_id="batch_gen_video_checkbox_elem")
                    batch_extract_glb_checkbox = gr.Checkbox(label="Extract GLB", value=True, elem_id="batch_extract_glb_checkbox_elem")
                    batch_extract_gs_checkbox = gr.Checkbox(label="Extract Gaussian Splats", value=True, elem_id="batch_extract_gs_checkbox_elem")
                batch_process_button = gr.Button("Start Batch Process", variant="primary")
                batch_status_textbox = gr.Textbox(label="Batch Process Log", lines=10, interactive=False, autoscroll=True)

    is_multiimage = gr.State(False)
    output_buf = gr.State() 


    preset_ui_components = [
        seed_slider, randomize_seed_checkbox, ss_guidance_strength_slider, ss_sampling_steps_slider,
        slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
        num_generations_slider, # Added
        mesh_simplify_slider, texture_size_slider, video_resolution_slider, video_num_frames_slider,
        video_fps_slider, save_metadata_checkbox,
        batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
        batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox
    ]
    # preset_state_components are implicitly defined by UI_COMPONENT_KEYS matching gr.State() names.

    
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

    # Inputs for perform_generations_and_optional_extractions
    gen_inputs = [
        num_generations_slider, # extract_models_after_each is passed by lambda
        image_prompt, multiimage_prompt, is_multiimage, seed_slider, randomize_seed_checkbox,
        ss_guidance_strength_slider, ss_sampling_steps_slider, slat_guidance_strength_slider, slat_sampling_steps_slider,
        multiimage_algo_radio, video_resolution_slider, video_num_frames_slider, video_fps_slider,
        save_metadata_checkbox, mesh_simplify_slider, texture_size_slider
    ]
    # Outputs for perform_generations_and_optional_extractions
    gen_outputs = [
        output_buf, video_output, model_output, download_glb, download_gs,
        extract_glb_btn, extract_gs_btn, general_status_textbox
    ]

    generate_btn.click(
        get_seed, inputs=[randomize_seed_checkbox, seed_slider], outputs=[seed_slider]
    ).then(
        lambda num_gens, img, m_imgs, is_m, seed, rand_s, ssg, sss, slg, sls, malgo, vres, vnf, vfps, smeta, msimp, tsize: \
            perform_generations_and_optional_extractions(
                num_gens, False, # extract_models_after_each = False
                img, m_imgs, is_m, seed, rand_s, ssg, sss, slg, sls, malgo, vres, vnf, vfps, smeta, msimp, tsize
            ),
        inputs=gen_inputs, outputs=gen_outputs
    )

    generate_and_extract_btn.click(
        get_seed, inputs=[randomize_seed_checkbox, seed_slider], outputs=[seed_slider]
    ).then(
        lambda num_gens, img, m_imgs, is_m, seed, rand_s, ssg, sss, slg, sls, malgo, vres, vnf, vfps, smeta, msimp, tsize: \
            perform_generations_and_optional_extractions(
                num_gens, True, # extract_models_after_each = True
                img, m_imgs, is_m, seed, rand_s, ssg, sss, slg, sls, malgo, vres, vnf, vfps, smeta, msimp, tsize
            ),
        inputs=gen_inputs, outputs=gen_outputs
    )
    
    def request_cancellation_action():
        global CANCEL_REQUESTED
        CANCEL_REQUESTED = True
        msg = "CANCEL REQUESTED: Will attempt to stop after the current generation/file. Check console for progress."
        print(msg)
        return msg

    cancel_button.click(request_cancellation_action, outputs=[general_status_textbox])


    video_output.clear(
        lambda: (gr.update(interactive=False), gr.update(interactive=False), None, None, None), # Clear model, downloads too
        outputs=[extract_glb_btn, extract_gs_btn, model_output, download_glb, download_gs],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify_slider, texture_size_slider, save_metadata_checkbox],
        outputs=[model_output, download_glb],
    ).then(
        lambda p: gr.update(interactive=bool(p)), inputs=[download_glb], outputs=[download_glb],
    )
    
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf, save_metadata_checkbox],
        outputs=[model_output, download_gs],
    ).then(
        lambda p: gr.update(interactive=bool(p)), inputs=[download_gs], outputs=[download_gs],
    )

    model_output.clear( 
        lambda: (gr.update(value=None, interactive=False), gr.update(value=None, interactive=False)), 
        outputs=[download_glb, download_gs], 
    )
    
    open_outputs_btn.click(lambda: open_folder(OUTPUT_DIR_BASE), inputs=None, outputs=None) 
    open_batch_outputs_btn.click(lambda current_batch_dir_name: open_folder(os.path.join(os.getcwd(), current_batch_dir_name if current_batch_dir_name else BATCH_OUTPUT_DIR_BASE_DEFAULT)), 
                                 inputs=[batch_output_folder_textbox], outputs=None)


    config_save_button.click(
        save_config,
        inputs=[config_save_name_textbox] + preset_ui_components,
        outputs=[config_status_textbox, config_load_dropdown]
    )
    config_load_dropdown.change( 
        load_config,
        inputs=[config_load_dropdown],
        outputs=[config_status_textbox] + preset_ui_components
    )
    demo.load(initial_load_config, inputs=None, outputs=[config_load_dropdown] + preset_ui_components)


    batch_process_button.click(
        run_batch_processing,
        inputs=[
            num_generations_slider, # Added
            batch_input_folder_textbox, batch_output_folder_textbox, batch_skip_existing_checkbox,
            batch_gen_video_checkbox, batch_extract_glb_checkbox, batch_extract_gs_checkbox,
            seed_slider, randomize_seed_checkbox, ss_guidance_strength_slider, ss_sampling_steps_slider,
            slat_guidance_strength_slider, slat_sampling_steps_slider, multiimage_algo_radio,
            mesh_simplify_slider, texture_size_slider, video_resolution_slider, video_num_frames_slider,
            video_fps_slider, save_metadata_checkbox
        ],
        outputs=[batch_status_textbox]
    )


def initialize_pipeline(precision_arg="fp32"): 
    global pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("models")
    
    if precision_arg == "fp16":
        effective_precision = "half"
    elif precision_arg == "fp32":
        effective_precision = "full"
    else: 
        effective_precision = "full" 

    print('')
    print(f"Using precision: '{effective_precision}' (requested: '{precision_arg}'). Loading...")
    if effective_precision == "half": 
        pipeline.to(torch.float16)
        if "image_cond_model" in pipeline.models:
             if hasattr(pipeline.models['image_cond_model'], 'half'):
                pipeline.models['image_cond_model'].half()


if __name__ == "__main__":
    initialize_pipeline(cmd_args.precision)
    demo.launch(inbrowser=True, share=cmd_args.share)