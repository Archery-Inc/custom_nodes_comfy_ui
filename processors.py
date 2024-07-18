import sys, os, torch
from numpy import ndarray
from sklearn.cluster import KMeans
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ArcheryPrint:
    CATEGORY = "archery"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(self, input_text: str):
        print("Result:", input_text)
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
                n_clusters=2,
                algorithm=self.algorithm,
                max_iter=self.num_iterations,
            )
            .fit(filtered_pixels)
            .cluster_centers_
        )
        return float(colors[0][0]), float(colors[0][1]), float(colors[0][2])
