import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", filename), "wb") as f:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    url = "https://github.com/HolyWu/vs-grlir/releases/download/model/"
    models = [
        "bsr_grl_base",
        "db_defocus_single_pixel_grl_base",
        "db_motion_grl_base_gopro",
        "db_motion_grl_base_realblur_j",
        "db_motion_grl_base_realblur_r",
        "dm_grl_small",
        "dn_grl_small_c3s15",
        "dn_grl_small_c3s25",
        "dn_grl_small_c3s50",
        "jpeg_grl_small_c3q10",
        "jpeg_grl_small_c3q20",
        "jpeg_grl_small_c3q30",
        "jpeg_grl_small_c3q40",
        "sr_grl_small_c3x2",
        "sr_grl_small_c3x3",
        "sr_grl_small_c3x4",
    ]
    for model in models:
        download_model(url + model + ".ckpt")
