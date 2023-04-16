from __future__ import annotations

import math
import os
from threading import Lock

import numpy as np
import torch
import vapoursynth as vs

from .grl import GRL

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def grlir(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 16,
) -> vs.VideoNode:
    """Efficient and Explicit Modelling of Image Hierarchies for Image Restoration

    :param clip:            Clip to process. Only RGBS format is supported.
    :param device_index:    Device ordinal of the GPU.
    :param num_streams:     Number of CUDA streams to enqueue the kernels.
    :param tile_w:          Tile width. As too large images result in the out of GPU memory issue, so this tile option
                            will first crop input images into tiles, and then process each of them. Finally, they will
                            be merged into one image. 0 denotes for do not use tile.
    :param tile_h:          Tile height.
    :param tile_pad:        Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("grlir: this is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("grlir: only RGBS format is supported")

    if not torch.cuda.is_available():
        raise vs.Error("grlir: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("grlir: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("grlir: setting num_streams greater than `core.num_threads` is useless")

    if os.path.getsize(os.path.join(model_dir, "bsr_grl_base.ckpt")) == 0:
        raise vs.Error("grlir: model files have not been downloaded. run 'python -m vsgrlir' first")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    module = GRL(
        upscale=4,
        embed_dim=180,
        img_size=128,
        upsampler="nearest+conv",
        depths=[4, 4, 8, 8, 8, 4, 4],
        num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        window_size=16,
        stripe_size=[32, 64],
        stripe_shift=True,
        mlp_ratio=2,
        anchor_window_down_factor=4,
        local_connection=True,
    )
    scale = 4

    model_path = os.path.join(model_dir, "bsr_grl_base.ckpt")

    state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    state_dict = {k.replace("model_g.", ""): v for k, v in state_dict.items() if "model_g." in k}

    module.load_state_dict(state_dict)
    module.eval().to(device, memory_format=torch.channels_last)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, module)
            else:
                output = module(img)

            return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor, scale: int, tile_w: int, tile_h: int, tile_pad: int, module: torch.nn.Module
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # process tile
            output_tile = module(input_tile)

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
