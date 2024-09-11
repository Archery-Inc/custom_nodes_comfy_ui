from typing import List, Dict
from functools import lru_cache
import numpy as np, json
from skimage.color import rgb2lab, deltaE_ciede2000
import os

HIERARCHY_JSON_FILE = (
    f"{os.getcwd()}/ComfyUI/custom_nodes/custom_nodes_comfy_ui/color_hierarchy.json"
)


@lru_cache(maxsize=1)
def _get_color_data():
    with open(HIERARCHY_JSON_FILE) as f:
        color_json = json.load(f)
        rgb_values = np.array(
            [
                [c["xkcd_r"] / 255.0, c["xkcd_g"] / 255.0, c["xkcd_b"] / 255.0]
                for c in color_json
            ]
        )
        return {
            "lab_values": rgb2lab(rgb_values),
            "xkcd_names": [c["xkcd_color"] for c in color_json],
            "color_hierarchy": {c["xkcd_color"]: c for c in color_json},
        }


def get_color_from_rgb(rgb_color: List[float]) -> Dict:
    assert (
        hasattr(rgb_color, "__len__") and len(rgb_color) == 3
    ), "rgb_color must be a list of 3 floats."
    assert all(
        [rgb_color[i] >= 0 and rgb_color[i] <= 255.0 for i in range(3)]
    ), "R, G, and B values must be between 0 and 255."

    rgb_color = [x / 255.0 for x in rgb_color]
    # scikit-image's rgb2lab wants m*n*3 arrays.
    lab = rgb2lab([[rgb_color]])[0][0]
    return get_color_from_lab(lab)


def get_color_from_lab(lab_color: List[float]) -> Dict:
    color_data = _get_color_data()
    assert (
        hasattr(lab_color, "__len__") and len(lab_color) == 3
    ), "lab_color must be a list of 3 floats."
    assert lab_color[0] >= 0 and lab_color[0] <= 100, "L should be between 0 and 100."
    assert (
        lab_color[1] >= -128 and lab_color[1] <= 127
    ), "A should be between -128 and 127."
    assert (
        lab_color[2] >= -128 and lab_color[2] <= 127
    ), "B should be between -128 and 127."

    # lab_values: (n, 3). lab_color: (3,). wrap lab_color so it's (1, 3) and
    # vectorized comparison can work.
    dists = deltaE_ciede2000(
        color_data["lab_values"], np.array([lab_color]), channel_axis=1
    )
    xkcd_name = color_data["xkcd_names"][dists.argmin()]
    return color_data["color_hierarchy"][xkcd_name]
