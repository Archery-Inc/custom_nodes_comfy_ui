import torch
import math
from .glsl_renderer import GLSL

# pip install moderngl


class ArcheryGLSL:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animated_image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "out_width": ("INT", {"default": 1024, "min": 1}),
                "out_height": ("INT", {"default": 576, "min": 1}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 2147483647}),
                "frame_count": ("INT", {"default": 72, "min": 1, "max": 2147483647}),
                "shader": ("STRING", {"multiline": True}),
            },
            "optional": {
                "skip_frame": ("INT", {"default": 1, "min": 1}),
                "background": ("STRING", {"default": "#FFFFFF"}),
                "foreground": ("STRING", {"default": "#000000"}),
                "position": (
                    [
                        "center",
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "top-left",
                        "top-right",
                        "bottom-left",
                        "bottom-right",
                    ],
                    {"default": "center"},
                ),
            },
        }

    def run(
        self,
        image,
        out_width,
        out_height,
        frame_rate,
        frame_count,
        shader,
        skip_frame,
        background,
        foreground,
        position,
    ):
        glsl = GLSL(
            image,
            out_width,
            out_height,
            shader,
            frame_rate,
            frame_count,
            background,
            foreground,
            position
        )
        images = [
            glsl.render(frame_index, skip_frame)
            for frame_index in range(math.ceil(frame_count / skip_frame))
        ]
        return (torch.cat(images, dim=0),)
