from .logic import ArcheryIfElse
from .inputs import *
from .glsl import ArcheryGLSL
import shutil, os, __main__, filecmp

def install_js_files():
    extensions_folder = os.path.join(
        os.path.dirname(os.path.realpath(__main__.__file__)),
        "web" + os.sep + "extensions" + os.sep + "archery",
    )
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
    if not os.path.exists(extensions_folder):
        print('Making the "web\\extensions\\archery" folder')
        os.makedirs(extensions_folder)

    result = filecmp.dircmp(javascript_folder, extensions_folder)
    if result.left_only or result.diff_files:
        print("Update to javascripts files detected")
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            print(f"Copying {file} to extensions folder")
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extensions_folder, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)


install_js_files()

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
