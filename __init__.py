from .logic import ArcheryIfElse
from .inputs import *
from .glsl import ArcheryGLSL
import shutil

shutil.copyfile(
    "/src/ComfyUI/custom_nodes/ComfyUI-archery/js/index.js",
    "/src/ComfyUI/web/extensions/archery/index.js",
)

NODE_CLASS_MAPPINGS = {
    "ArcheryInputBool": ArcheryInputBool,
    "ArcheryInputFloat": ArcheryInputFloat,
    "ArcheryInputInt": ArcheryInputInt,
    "ArcheryInputAnimateDiffModelSelector": ArcheryInputAnimateDiffModelSelector,
    "ArcheryInputCheckpointSelector": ArcheryInputCheckpointSelector,
    "ArcheryInputIpAdapterSelector": ArcheryInputIpAdapterSelector,
    "ArcheryInputIpAdapterWeightSelector": ArcheryInputIpAdapterWeightSelector,
    "ArcheryInputLoraSelector": ArcheryInputLoraSelector,
    "ArcheryInputUpscalerSelector": ArcheryInputUpscalerSelector,
    "ArcheryInputAnimateDiffModelSelector": ArcheryInputAnimateDiffModelSelector,
    "ArcheryInputNoiseTypeSelector": ArcheryInputNoiseTypeSelector,
    "ArcheryInputAnimateDiffMotionLoraSelector": ArcheryInputAnimateDiffMotionLoraSelector,
    "ArcheryInputString": ArcheryInputString,
    "ArcheryInputStringMultiline": ArcheryInputStringMultiline,
    "ArcheryIfElse": ArcheryIfElse,
    "ArcheryGLSL": ArcheryGLSL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArcheryInputBool": "Archery Input Bool",
    "ArcheryInputFloat": "Archery Input Float",
    "ArcheryInputInt": "Archery Input Int",
    "ArcheryInputAnimateDiffModelSelector": "Archery Input AnimateDiffModel Selector",
    "ArcheryInputCheckpointSelector": "Archery Input Checkpoint Selector",
    "ArcheryInputIpAdapterSelector": "Archery Input IpAdapter Selector",
    "ArcheryInputIpAdapterWeightSelector": "Archery Input IpAdapterWeight Selector",
    "ArcheryInputLoraSelector": "Archery Input Lora Selector",
    "ArcheryInputUpscalerSelector": "Archery Input Upscaler Selector",
    "ArcheryInputAnimateDiffModelSelector": "Archery Input AnimateDiffModel Selector",
    "ArcheryInputNoiseTypeSelector": "Archery Input Noise Type Selector",
    "ArcheryInputAnimateDiffMotionLoraSelector": "Archery Input Animate Diff Motion Lora Selector",
    "ArcheryInputString": "Archery Input String",
    "ArcheryInputStringMultiline": "Archery Input String Multiline",
    "ArcheryIfElse": "Archery If Else",
    "ArcheryGLSL": "Archery GLSL",
}
