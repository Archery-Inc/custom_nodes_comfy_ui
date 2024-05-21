import folder_paths

class ArcheryInputBool:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("BOOLEAN", {"default": False})}}

    def run(self, value):
        return (value,)


class ArcheryInputFloat:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.001 }) }}

    def run(self, value):
        return (value,)


class ArcheryInputInt:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647}) }}

    def run(self, value):
        return (value,)


class ArcheryInputAnimateDiffModelSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (folder_paths.get_filename_list("animatediff_models"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (folder_paths.get_filename_list("animatediff_models"), ),}}

    def run(self, value):
        return (value,)


class ArcheryInputAnimateDiffMotionLoraSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (folder_paths.get_filename_list("animatediff_motion_lora"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (folder_paths.get_filename_list("animatediff_motion_lora"), ),}}

    def run(self, value):
        return (value,)


class ArcheryInputCheckpointSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (folder_paths.get_filename_list("checkpoints"), ),}}

    def run(self, value):
        return (value,)


ip_adapter_options = ['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)']

class ArcheryInputIpAdapterSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (ip_adapter_options,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ip_adapter_options, ),}}

    def run(self, value):
        return (value,)

WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer']

class ArcheryInputIpAdapterWeightSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (WEIGHT_TYPES,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (WEIGHT_TYPES, ),}}

    def run(self, value):
        return (value,)


class ArcheryInputLoraSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (folder_paths.get_filename_list("loras"), ),}}

    def run(self, value):
        return (value,)
    
class ArcheryInputUpscalerSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (folder_paths.get_filename_list("upscale_models"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (folder_paths.get_filename_list("upscale_models"), ),}}

    def run(self, value):
        return (value,)


NOISE_TYPES = ["default", "constant", "empty", "repeated_context", "FreeNoise"]

class ArcheryInputNoiseTypeSelector:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (NOISE_TYPES,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (NOISE_TYPES, ),}}

    def run(self, value):
        return (value,)
    

class ArcheryInputString:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", { "default": "" }) }}

    def run(self, value):
        return (value,)


class ArcheryInputStringMultiline:
    CATEGORY = "archery-inc"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("STRING", { "default": "", "multiline": True }) }}

    def run(self, value):
        return (value,)
