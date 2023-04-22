from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs
from vsutil import fallback

from .grl import GRL

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class CUDAGraphs:
        graph: list[torch.cuda.CUDAGraph]
        static_input: list[torch.Tensor]
        static_output: list[torch.Tensor]


@torch.inference_mode()
def grlir(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    cuda_graphs: bool = False,
    model: int = 0,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int | None = None,
) -> vs.VideoNode:
    """Efficient and Explicit Modelling of Image Hierarchies for Image Restoration

    :param clip:            Clip to process. Only RGBH and RGBS formats are supported.
                            RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:    Device ordinal of the GPU.
    :param num_streams:     Number of CUDA streams to enqueue the kernels.
    :param cuda_graphs:     Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels sequentially.
    :param model:           Model to use.
                             0 = Blind Image SR
                             1 = Defocus Deblurring
                             2 = Motion Deblurring (GoPro)
                             3 = Motion Deblurring (RealBlur-J)
                             4 = Motion Deblurring (RealBlur-R)
                             5 = Demosaicking
                             6 = Denoising (sigma 15)
                             7 = Denoising (sigma 25)
                             8 = Denoising (sigma 50)
                             9 = JPEG compression artifact removal (quality 10)
                            10 = JPEG compression artifact removal (quality 20)
                            11 = JPEG compression artifact removal (quality 30)
                            12 = JPEG compression artifact removal (quality 40)
                            13 = Classical Image SR (scale 2)
                            14 = Classical Image SR (scale 3)
                            15 = Classical Image SR (scale 4)
    :param tile_w:          Tile width. As too large images result in the out of GPU memory issue, so this tile option
                            will first crop input images into tiles, and then process each of them. Finally, they will
                            be merged into one image. 0 denotes for do not use tile.
    :param tile_h:          Tile height.
    :param tile_pad:        Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("grlir: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("grlir: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("grlir: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("grlir: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("grlir: setting num_streams greater than `core.num_threads` is useless")

    if model not in range(16):
        raise vs.Error("grlir: model must be between 0 and 15 (inclusive)")

    if os.path.getsize(os.path.join(model_dir, "bsr_grl_base.ckpt")) == 0:
        raise vs.Error("grlir: model files have not been downloaded. run 'python -m vsgrlir' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    scale = 1

    match model:
        case 0:
            model_name = "bsr_grl_base.ckpt"
            module = GRL(
                img_size=128,
                embed_dim=180,
                upscale=4,
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
            tile_pad = fallback(tile_pad, 16)
            pad_size = 64
        case 1:
            model_name = "db_defocus_single_pixel_grl_base.ckpt"
            module = GRL(
                img_size=480,
                embed_dim=180,
                upscale=1,
                upsampler="",
                depths=[4, 4, 8, 8, 8, 4, 4],
                num_heads_window=[3, 3, 3, 3, 3, 3, 3],
                num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
                window_size=16,
                stripe_size=[48, 96],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=True,
            )
            tile_pad = fallback(tile_pad, 16)
            pad_size = 96
        case 2:
            model_name = "db_motion_grl_base_gopro.ckpt"
            module = GRL(
                img_size=480,
                embed_dim=180,
                upscale=1,
                upsampler="",
                depths=[4, 4, 8, 8, 8, 4, 4],
                num_heads_window=[3, 3, 3, 3, 3, 3, 3],
                num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
                window_size=12,
                stripe_size=[48, 96],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=True,
            )
            tile_pad = fallback(tile_pad, 12)
            pad_size = 96
        case 3:
            model_name = "db_motion_grl_base_realblur_j.ckpt"
            module = GRL(
                img_size=480,
                embed_dim=180,
                upscale=1,
                upsampler="",
                depths=[4, 4, 8, 8, 8, 4, 4],
                num_heads_window=[3, 3, 3, 3, 3, 3, 3],
                num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
                window_size=12,
                stripe_size=[48, 96],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=True,
            )
            tile_pad = fallback(tile_pad, 12)
            pad_size = 96
        case 4:
            model_name = "db_motion_grl_base_realblur_r.ckpt"
            module = GRL(
                img_size=480,
                embed_dim=180,
                upscale=1,
                upsampler="",
                depths=[4, 4, 8, 8, 8, 4, 4],
                num_heads_window=[3, 3, 3, 3, 3, 3, 3],
                num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
                window_size=12,
                stripe_size=[48, 96],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=True,
            )
            tile_pad = fallback(tile_pad, 12)
            pad_size = 96
        case 5:
            model_name = "dm_grl_small.ckpt"
            module = GRL(
                img_size=64,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=8,
                stripe_size=[32, 32],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 8)
            pad_size = 32
        case 6:
            model_name = "dn_grl_small_c3s15.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=16,
                stripe_size=[64, 128],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 16)
            pad_size = 128
        case 7:
            model_name = "dn_grl_small_c3s25.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=16,
                stripe_size=[64, 128],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 16)
            pad_size = 128
        case 8:
            model_name = "dn_grl_small_c3s50.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=16,
                stripe_size=[64, 128],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 16)
            pad_size = 128
        case 9:
            model_name = "jpeg_grl_small_c3q10.ckpt"
            module = GRL(
                img_size=288,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=36,
                stripe_size=[72, 144],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 36)
            pad_size = 144
        case 10:
            model_name = "jpeg_grl_small_c3q20.ckpt"
            module = GRL(
                img_size=288,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=36,
                stripe_size=[72, 144],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 36)
            pad_size = 144
        case 11:
            model_name = "jpeg_grl_small_c3q30.ckpt"
            module = GRL(
                img_size=288,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=36,
                stripe_size=[72, 144],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 36)
            pad_size = 144
        case 12:
            model_name = "jpeg_grl_small_c3q40.ckpt"
            module = GRL(
                img_size=288,
                embed_dim=128,
                upscale=1,
                upsampler="",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=36,
                stripe_size=[72, 144],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            tile_pad = fallback(tile_pad, 36)
            pad_size = 144
        case 13:
            model_name = "sr_grl_small_c3x2.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=2,
                upsampler="pixelshuffle",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=32,
                stripe_size=[64, 64],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            scale = 2
            tile_pad = fallback(tile_pad, 32)
            pad_size = 64
        case 14:
            model_name = "sr_grl_small_c3x3.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=3,
                upsampler="pixelshuffle",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=32,
                stripe_size=[64, 64],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            scale = 3
            tile_pad = fallback(tile_pad, 32)
            pad_size = 64
        case 15:
            model_name = "sr_grl_small_c3x4.ckpt"
            module = GRL(
                img_size=256,
                embed_dim=128,
                upscale=4,
                upsampler="pixelshuffle",
                depths=[4, 4, 4, 4],
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                window_size=32,
                stripe_size=[64, 64],
                stripe_shift=True,
                mlp_ratio=2,
                anchor_window_down_factor=4,
                local_connection=False,
            )
            scale = 4
            tile_pad = fallback(tile_pad, 32)
            pad_size = 64

    model_path = os.path.join(model_dir, model_name)

    state_dict = torch.load(model_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model_g.", ""): v for k, v in state_dict.items() if "model_g." in k}
    else:
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}

    module.load_state_dict(state_dict, strict=False)
    module.eval().to(device, memory_format=torch.channels_last)
    if fp16:
        module.half()

    if tile_w > 0 and tile_h > 0:
        pad_w = math.ceil(min(tile_w + 2 * tile_pad, clip.width) / pad_size) * pad_size
        pad_h = math.ceil(min(tile_h + 2 * tile_pad, clip.height) / pad_size) * pad_size
    else:
        pad_w = math.ceil(clip.width / pad_size) * pad_size
        pad_h = math.ceil(clip.height / pad_size) * pad_size

    if cuda_graphs:
        graph: list[torch.cuda.CUDAGraph] = []
        static_input: list[torch.Tensor] = []
        static_output: list[torch.Tensor] = []

        for i in range(num_streams):
            static_input.append(
                torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device).to(memory_format=torch.channels_last)
            )

            torch.cuda.synchronize(device=device)
            stream[i].wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream[i]):
                module(static_input[i])
            torch.cuda.current_stream(device=device).wait_stream(stream[i])
            torch.cuda.synchronize(device=device)

            graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graph[i], stream=stream[i]):
                static_output.append(module(static_input[i]))

        backend = Backend.CUDAGraphs(graph, static_input, static_output)
    else:
        backend = Backend.Eager(module)

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
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "reflect")

                if cuda_graphs:
                    static_input[local_index].copy_(img)
                    graph[local_index].replay()
                    output = static_output[local_index]
                else:
                    output = module(img)

                output = output[:, :, : h * scale, : w * scale]

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
    img: torch.Tensor,
    scale: int,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.CUDAGraphs,
    index: int,
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

            h, w = input_tile.shape[2:]
            mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), mode)

            # process tile
            if isinstance(backend, Backend.CUDAGraphs):
                backend.static_input[index].copy_(input_tile)
                backend.graph[index].replay()
                output_tile = backend.static_output[index]
            else:
                output_tile = backend.module(input_tile)

            output_tile = output_tile[:, :, : h * scale, : w * scale]

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
