
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_typ = AnyType("*")

class ArcheryIfElse:
    CATEGORY = 'archery-inc'
    RETURN_TYPES = (any_typ,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN",),
            },
            "optional": {
                "true": (any_typ,),
                "false": (any_typ,),
            }
        }
    
    def run(self, condition, true=None, false=None):
        return (true if condition else false,)
