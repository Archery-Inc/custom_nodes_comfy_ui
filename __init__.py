class Bool:
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    CATEGORY = "archery-inc"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"value": ("BOOLEAN", {"default": False})}}

    def run(self, value):
        return (value,)


class Int:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    CATEGORY = "archery-inc"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647})
            }
        }

    def run(self, value):
        return (value,)

NODE_CLASS_MAPPINGS = {"Bool": Bool, "Int": Int}
NODE_DISPLAY_NAME_MAPPINGS = {"Bool Constant": "Bool", "Int": "Int"}
