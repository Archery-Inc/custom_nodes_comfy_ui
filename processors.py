from collections import Counter
import re
import numpy as np
from sklearn.cluster import KMeans
import sys, os, torch
from numpy import ndarray
from .color_utils import *
from .colornamer import get_color_from_rgb
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ArcheryPrint:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("FLOAT", {"default": 0}),
            },
            "optional": {
                "title": ("STRING", {"default": "Result"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(self, input_text: float, title: str):
        print(title, ":", input_text)
        return ()


class ArcheryColorDetection:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "k_means_algorithm": (
                    ["lloyd", "elkan", "auto", "full"],
                    {
                        "default": "lloyd",
                    },
                ),
                "accuracy": (
                    "INT",
                    {
                        "default": 60,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                    },
                ),
                "threshold": (
                    "INT",
                    {
                        "default": 60,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("luminance",)
    FUNCTION = "main"

    def main(
        self,
        input_image: torch.Tensor,
        k_means_algorithm: str = "lloyd",
        accuracy: int = 60,
        threshold: int = 60,
        unique_id: str = "",
        extra_pnginfo: str = "",
        output_text: str = "",
    ) -> str:
        self.exclude = []
        self.num_iterations = int(512 * (accuracy / 100))
        self.algorithm = k_means_algorithm

        rgb = self.get_main_color(input_image)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return ("#FFFFFF" if luminance < threshold / 100 else "#353536",)

    def get_main_color(self, image: torch.Tensor) -> List[ndarray]:
        pixels = image.view(-1, image.shape[-1]).numpy()
        # remove transparent pixels
        mask = pixels[:, 3] > 0.5
        filtered_pixels = pixels[mask]
        colors = (
            KMeans(
                n_clusters=1,
                algorithm=self.algorithm,
                max_iter=self.num_iterations,
            )
            .fit(filtered_pixels)
            .cluster_centers_
        )
        return float(colors[0][0]), float(colors[0][1]), float(colors[0][2])


class ArcheryBackgroundColorDetection:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "k_means_algorithm": (
                    ["lloyd", "elkan", "auto", "full"],
                    {
                        "default": "lloyd",
                    },
                ),
                "accuracy": (
                    "INT",
                    {
                        "default": 60,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 4,
                        "display": "slider",
                        "min": 1,
                        "max": 10,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "output_text": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "color0",
        "color1",
        "color2",
        "color3",
        "proportion0",
        "proportion1",
        "proportion2",
        "proportion3",
        "output_color",
    )
    FUNCTION = "main"

    def main(
        self,
        input_image: torch.Tensor,
        k_means_algorithm: str = "lloyd",
        accuracy: int = 60,
        count: int = 4,
        unique_id: str = "",
        extra_pnginfo: str = "",
        output_text: str = "",
    ) -> str:
        self.exclude = []
        self.num_iterations = int(512 * (accuracy / 100))
        self.algorithm = k_means_algorithm

        colors = self.get_main_colors(input_image, count)
        output_color = self.get_background_color(colors)

        hex_colors = [rgb_to_hex(rgb) for (rgb, _) in colors]
        proportions = [proportion for (_, proportion) in colors]
        return tuple([*hex_colors, *proportions, output_color])

    def get_background_color(self, colors: list[tuple[ndarray, float]]):
        first_rgb = colors[0][0]
        distances = [cie94_distance_rgb(first_rgb, color) for color, _ in colors]
        max_distance = max(distances)

        color_similarity_threshold = 0.25 * 100
        color_contrast_threshold = 0.15 * 100

        if max_distance < color_similarity_threshold:
            hsl = rgb_to_hsl(first_rgb)
            luminance = hsl[2]
            if luminance < 0.2:
                return "#FFFFFF"
            if luminance > 0.8:
                return "#000000"
            hsl[0] = sum([k * rgb_to_hsl(color)[0] for color, k in colors])
            hsl[2] = 0.95
            return rgb_to_hex(hsl_to_rgb(hsl))

        min_distance_to_black = min(
            [cie94_distance_rgb(np.array([0, 0, 0]), color) for color, _ in colors]
        )

        min_distance_to_white = min(
            [
                cie94_distance_rgb(np.array([255, 255, 255]), color)
                for color, _ in colors
            ]
        )

        if max(min_distance_to_black, min_distance_to_white) > color_contrast_threshold:
            if min_distance_to_black > min_distance_to_white:
                return "#000000"
            return "#FFFFFF"

        if min_distance_to_black > min_distance_to_white:
            return rgb_to_hex(
                hsl_to_rgb(np.array([0, 0, 2 * color_contrast_threshold / 100]))
            )
        return rgb_to_hex(
            hsl_to_rgb(np.array([0, 0, 1 - 2 * color_contrast_threshold / 100]))
        )

    def get_main_colors(self, image: torch.Tensor, count: int):
        pixels = image.view(-1, image.shape[-1]).numpy()
        # remove transparent pixels
        mask = pixels[:, 3] > 0.5
        filtered_pixels = pixels[mask]

        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=count,
            algorithm=self.algorithm,
            max_iter=self.num_iterations,
        ).fit(filtered_pixels)

        colors = kmeans.cluster_centers_
        labels = kmeans.labels_

        cluster_sizes = Counter(labels)
        total = cluster_sizes.total()
        rgbs_with_sizes = [
            (
                to_rgb(255 * c),
                cluster_sizes[i] / total,
            )
            for i, c in enumerate(colors)
        ]

        return sorted(rgbs_with_sizes, key=lambda x: x[1], reverse=True)


class ArcheryPromptParser:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
            },
            "optional": {
                "color0": ("STRING", {"default": ""}),
                "color1": ("STRING", {"default": ""}),
                "color2": ("STRING", {"default": ""}),
                "color3": ("STRING", {"default": ""}),
                "proportion0": ("FLOAT", {"default": 0}),
                "proportion1": ("FLOAT", {"default": 0}),
                "proportion2": ("FLOAT", {"default": 0}),
                "proportion3": ("FLOAT", {"default": 0}),
                "background_color": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "main"

    def main(
        self,
        prompt: str,
        color0: str,
        color1: str,
        color2: str,
        color3: str,
        proportion0: float,
        proportion1: float,
        proportion2: float,
        proportion3: float,
        background_color: str,
    ) -> str:
        min_color_proportion_threshold = 0.05
        proportions = [proportion0, proportion1, proportion2, proportion3]
        colors = [color0, color1, color2, color3]
        major_colors = [
            colors[i]
            for i in range(len(colors))
            if proportions[i] > min_color_proportion_threshold
        ]

        readable_colors = []
        for color in major_colors:
            readable_color = self.color_to_human_readable(color)
            if readable_color not in readable_colors:  # Deduplicate colors
                readable_colors.append(readable_color)

        def replace_colors(match):
            fallbacks = match.group().split("|")
            fallback = self.get_first_working_fallback(
                fallbacks, len(readable_colors) - 1
            )
            replaced_str = (
                fallback.replace("{", "")
                .replace("}", "")
                .replace("background", self.color_to_human_readable(background_color))
            )
            for i in range(len(readable_colors)):
                replaced_str = replaced_str.replace(f"color{i}", readable_colors[i])
            return replaced_str

        result = re.sub(r"\{[^}]*\}", replace_colors, prompt)
        print("Prompt: ", result)
        return (result,)

    def get_first_working_fallback(
        self, fallbacks: list[str], last_available_color_index: int
    ):
        for fallback in fallbacks:
            is_working = True
            for i in range(last_available_color_index + 1, 4):
                if f"color{i}" in fallback:
                    is_working = False
                    break
            if is_working:
                return fallback
        return fallbacks[-1]

    def color_to_human_readable(self, color: str):
        rgb = hex_to_rgb(color)
        luminance = get_luminance(rgb)
        if luminance < 0.2:
            return "black"
        if luminance > 0.8:
            return "white"
        mapping = get_color_from_rgb(rgb)
        family = mapping["color_family"]
        type = mapping["color_type"].removesuffix(" color")
        return type + " " + family
