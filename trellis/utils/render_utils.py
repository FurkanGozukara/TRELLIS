import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence

from api_spz.core.exceptions import CancelledException

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs, dtype=torch.float32):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ],  dtype=dtype, device='cuda') * r
        extr = utils3d.torch.extrinsics_look_at(orig,
                                                torch.tensor([0, 0, 0], dtype=dtype, device='cuda'),
                                                torch.tensor([0, 0, 1], dtype=dtype, device='cuda'))
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        if intr.dtype != dtype:
            intr = intr.to(dtype)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


# Renderer instances are cached because MeshRenderer allocates an nvdiffrast
# RasterizeCudaContext in its constructor; re-creating one for every video/multiview
# render costs time and leaks GPU contexts. All rendering options are (re)assigned on
# every call, so a cached instance behaves exactly like a fresh one.
_RENDERER_CACHE = {}


def _get_renderer(kind):
    renderer = _RENDERER_CACHE.get(kind)
    if renderer is None:
        renderer = {'octree': OctreeRenderer, 'gaussian': GaussianRenderer, 'mesh': MeshRenderer}[kind]()
        _RENDERER_CACHE[kind] = renderer
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, cancel_event=None,
                  frame_callback=None, **kwargs):
    """
    Render a list of camera poses.

    Args:
        frame_callback: optional ``fn(index, total)`` invoked after every rendered frame.
        options['return_depth']: gaussian/octree only - also copy the depth buffer back to
            the host. Off by default because nothing in TRELLIS consumes it and the
            device->host copy is a large part of the per-frame cost.
        options['mesh_return_types']: which buffers the mesh renderer should produce.
            Defaults to ``['normal']`` - the only buffer used by the video/preview code.
            Producing "mask"/"depth" as well triples the antialias work at ssaa=4.
    """
    if isinstance(sample, Octree):
        renderer = _get_renderer('octree')
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = _get_renderer('gaussian')
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = _get_renderer('mesh')
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')

    return_depth = options.get('return_depth', False)
    mesh_return_types = options.get('mesh_return_types', ['normal'])
    total = len(extrinsics) if hasattr(extrinsics, '__len__') else None

    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering',
                                total=total, disable=not verbose):

        if cancel_event and cancel_event.is_set():
            raise CancelledException(f"User Cancelled")

        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if return_depth:
                if 'percent_depth' in res:
                    rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
                elif 'depth' in res:
                    rets['depth'].append(res['depth'].detach().cpu().numpy())
                else:
                    rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr, return_types=mesh_return_types)
            if 'normal' not in rets: rets['normal'] = []

            normal = res['normal']
            if torch.isnan(normal).any() or torch.isinf(normal).any():
                normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)

            rets['normal'].append(np.clip(normal.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))

        if frame_callback is not None:
            frame_callback(j + 1, total)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=30, r=2, fov=40, cancel_event=None,
                 options=None, **kwargs):
    # Dynamically check for the dtype,  so that float16, float32 work:
    if hasattr(sample, 'vertices') and hasattr(sample.vertices, 'dtype'):
        dtype = sample.vertices.dtype
    else:
        dtype = torch.float32 #for Gaussian etc - use default dtype, since float16 isn't supported by those
    # proceed:
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov, dtype=dtype)
    render_options = {'resolution': resolution, 'bg_color': bg_color}
    if options:
        render_options.update(options)
    return render_frames(sample, extrinsics, intrinsics, render_options, cancel_event=cancel_event, **kwargs)


def render_multiview(sample, resolution=512, nviews=30, cancel_event=None, **kwargs):
    # Dynamically check for the dtype,  so that float16, float32 work:
    if hasattr(sample, 'vertices') and hasattr(sample.vertices, 'dtype'):
        dtype = sample.vertices.dtype
    else:
        dtype = torch.float32 #for Gaussian etc - use default dtype, since float16 isn't supported by those
    # proceed:
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov, dtype=dtype)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)},
                        cancel_event=cancel_event, **kwargs)
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi),
                     r=10, fov=8, cancel_event=None, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, cancel_event=cancel_event, **kwargs)