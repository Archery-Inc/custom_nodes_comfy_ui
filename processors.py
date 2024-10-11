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
                "default_background": ("STRING", {"default": ""}),
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
        default_background: str = "",
        unique_id: str = "",
        extra_pnginfo: str = "",
        output_text: str = "",
    ) -> str:
        self.exclude = []
        self.num_iterations = int(512 * (accuracy / 100))
        self.algorithm = k_means_algorithm

        colors = self.get_main_colors(input_image, count)
        if default_background:
            output_color = default_background
        else:
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


# These values are established by empiricism with tests (tradeoff: performance VS precision)
NEWTON_ITERATIONS = 4
NEWTON_MIN_SLOPE = 0.001
SUBDIVISION_PRECISION = 0.0000001
SUBDIVISION_MAX_ITERATIONS = 10

kSplineTableSize = 11
kSampleStepSize = 1.0 / (kSplineTableSize - 1.0)


def A(aA1: float, aA2: float):
    return 1.0 - 3.0 * aA2 + 3.0 * aA1


def B(aA1: float, aA2: float):
    return 3.0 * aA2 - 6.0 * aA1


def C(aA1: float):
    return 3.0 * aA1


# Returns x(t) given t, x1, and x2, or y(t) given t, y1, and y2.
def calc_bezier(aT: float, aA1: float, aA2: float):
    return ((A(aA1, aA2) * aT + B(aA1, aA2)) * aT + C(aA1)) * aT


# Returns dx/dt given t, x1, and x2, or dy/dt given t, y1, and y2.
def get_slope(aT: float, aA1: float, aA2: float):
    return 3.0 * A(aA1, aA2) * aT * aT + 2.0 * B(aA1, aA2) * aT + C(aA1)


def binary_subdivide(aX: float, aA: float, aB: float, mX1: float, mX2: float):
    i = 0
    currentX, currentT = 0.0, 0.0
    while i < SUBDIVISION_MAX_ITERATIONS:
        currentT = aA + (aB - aA) / 2.0
        currentX = calc_bezier(currentT, mX1, mX2) - aX
        if abs(currentX) <= SUBDIVISION_PRECISION:
            break
        if currentX > 0.0:
            aB = currentT
        else:
            aA = currentT
        i += 1
    return currentT


def newton_raphson_iterate(aX: float, aGuessT: float, mX1: float, mX2: float):
    for i in range(NEWTON_ITERATIONS):
        current_slope = get_slope(aGuessT, mX1, mX2)
        if current_slope == 0.0:
            return aGuessT
        currentX = calc_bezier(aGuessT, mX1, mX2) - aX
        aGuessT -= currentX / current_slope
    return aGuessT


def linear_easing(x: float):
    return x


def bezier(mX1: float, mY1: float, mX2: float, mY2: float):
    if not (0 <= mX1 <= 1 and 0 <= mX2 <= 1):
        raise ValueError("bezier x values must be in [0, 1] range")

    if mX1 == mY1 and mX2 == mY2:
        return linear_easing

    # Precompute samples table
    sample_values = np.zeros(kSplineTableSize)
    for i in range(kSplineTableSize):
        sample_values[i] = calc_bezier(i * kSampleStepSize, mX1, mX2)

    def get_t_for_x(aX):
        interval_start = 0.0
        current_sample = 1
        last_sample = kSplineTableSize - 1

        while current_sample != last_sample and sample_values[current_sample] <= aX:
            current_sample += 1
            interval_start += kSampleStepSize
        current_sample -= 1

        dist = (aX - sample_values[current_sample]) / (
            sample_values[current_sample + 1] - sample_values[current_sample]
        )
        guess_for_t = interval_start + dist * kSampleStepSize

        initial_slope = get_slope(guess_for_t, mX1, mX2)
        if initial_slope >= NEWTON_MIN_SLOPE:
            return newton_raphson_iterate(aX, guess_for_t, mX1, mX2)
        elif initial_slope == 0.0:
            return guess_for_t
        else:
            return binary_subdivide(
                aX, interval_start, interval_start + kSampleStepSize, mX1, mX2
            )

    def bezier_easing(x):
        if x == 0 or x == 1:
            return x
        return calc_bezier(get_t_for_x(x), mY1, mY2)

    return bezier_easing


class ArcheryLatentKeyframe:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_start": ("INT", {"default": 0}),
                "frame_end": ("INT", {"default": 120}),
                "start_strength": ("FLOAT", {"default": 0}),
                "end_strength": ("FLOAT", {"default": 1}),
                "ease": ("STRING", {"default": "1,0,0,1"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)
    FUNCTION = "main"

    def main(
        self,
        frame_start: int,
        frame_end: int,
        start_strength: float,
        end_strength: float,
        ease: str,
    ):
        bezier_xs = [float(x) for x in ease.split(",")]
        easing_f = bezier(bezier_xs[0], bezier_xs[1], bezier_xs[2], bezier_xs[3])

        values = [0] * 120
        for i in range(120):
            if i < frame_start:
                values[i] = start_strength
            elif i > frame_end:
                values[i] = end_strength
            else:
                t = easing_f((i - frame_start) / (frame_end - frame_start))
                values[i] = t * (end_strength - start_strength) + start_strength
        return (values,)
