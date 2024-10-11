import torch, math, moderngl, numpy as np
from PIL import Image


VERTEX_SHADER = """
    #version 330
    in vec2 iPosition;
    out vec2 fragCoord;

    void main() {
        gl_Position = vec4(iPosition, 0.0, 1.0);
        fragCoord = iPosition / 2.0 + 0.5;
        fragCoord.y = 1. - fragCoord.y;
    }
"""

FRAGMENT_SHADER_HEADER = """
    #version 330

    in vec2 fragCoord;
    uniform vec2 iResolution;
    uniform float iTime;
    uniform float iDuration;
    uniform float iTimeDelta;
    uniform int iFrame;
    uniform sampler2D iChannel0;
    uniform vec3 backgroundColor;
    uniform vec3 foregroundColor;
    uniform float textureTop;
    uniform float textureBottom;
    uniform float textureLeft;
    uniform float textureRight;
    uniform float int0;
    uniform float int1;

    out vec4 fragColor;
"""


class GLSL:
    def __init__(
        self,
        images,
        out_width: int,
        out_height: int,
        shader: str,
        frame_rate: int,
        frame_count: int,
        background: str,
        foreground: str,
        position: str,
        margin: float,
        x: float,
        y: float,
        int0: int,
        int1: int,
    ):
        ctx = moderngl.create_context(
            standalone=True,
        )
        program = ctx.program(VERTEX_SHADER, FRAGMENT_SHADER_HEADER + shader)

        vbo = ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1], dtype="f4").tobytes()
        )
        self.vao = ctx.simple_vertex_array(program, vbo, "iPosition")
        self.fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((out_width, out_height), 4)]
        )

        # Send image to shader
        image = self._transform(
            images[0], out_width, out_height, position, margin, x, y
        )

        iChannel0 = ctx.texture(image.size, components=4, data=image.tobytes())
        iChannel0.repeat_x = False
        iChannel0.repeat_y = False
        iChannel0.use(location=0)

        # Uniform locations
        self.iTime = program.get("iTime", None)
        self.iFrame = program.get("iFrame", None)
        iResolution = program.get("iResolution", None)
        iTimeDelta = program.get("iTimeDelta", None)
        iDuration = program.get("iDuration", None)
        backgroundColor = program.get("backgroundColor", None)
        foregroundColor = program.get("foregroundColor", None)
        int0_loc = program.get("int0", None)
        int1_loc = program.get("int1", None)

        # Uniform initialization
        self.runtime = 0
        self.delta = 1.0 / frame_rate
        if iResolution:
            iResolution.value = (out_width, out_height)
        if iTimeDelta:
            iTimeDelta.value = self.delta
        if iDuration:
            iDuration.value = self.delta * frame_count
        if backgroundColor:
            backgroundColor.value = self._hex_to_vec3(background)
        if foregroundColor:
            foregroundColor.value = self._hex_to_vec3(foreground)
        if int0_loc:
            int0_loc.value = int0
        if int1_loc:
            int1_loc.value = int1

    def render(self, frame_index: int, skip_frame: int):
        self.fbo.use()
        self.fbo.clear(1.0, 1.0, 1.0, 1.0)

        # Render the image
        self.vao.render()
        pixels = self.fbo.color_attachments[0].read()
        image = Image.frombytes("RGBA", self.fbo.size, pixels, "raw", "RGBA", 0, -1)
        image = image.convert("RGB")

        # Update uniforms
        if self.iTime:
            self.iTime.value = self.runtime
        if self.iFrame:
            self.iFrame.value = skip_frame * frame_index
        self.runtime += skip_frame * self.delta

        # Convert image to torch tensor
        pixels = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(pixels)[None,]

    def _hex_to_vec3(self, hex: str):
        h = hex.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def _transform(
        self,
        img,
        out_width: int,
        out_height: int,
        position: str,
        margin: float,
        x: float,
        y: float,
    ):
        numpy_img = 255.0 * img.cpu().numpy()
        pil_img = Image.fromarray(np.clip(numpy_img, 0, 255).astype(np.uint8))
        final_img = Image.new("RGBA", (out_width, out_height))

        # Put the image at the center while keeping aspect ratio
        w = int((1 - 2 * margin) * out_width)
        h = math.ceil(w * pil_img.height / pil_img.width)
        if out_width * pil_img.height / pil_img.width > out_height:
            h = int((1 - 2 * margin) * out_height)
            w = math.ceil(h * pil_img.width / pil_img.height)
        pil_img = pil_img.resize((w, h))

        p = 0
        match position:
            case "center":
                left, top = (out_width - w) // 2, (out_height - h) // 2
            case "left":
                left, top = p, (out_height - h) // 2
            case "right":
                left, top = out_width - w - p, (out_height - h) // 2
            case "top":
                left, top = (out_width - w) // 2, p
            case "bottom":
                left, top = (out_width - w) // 2, out_height - h - p
            case "top-left":
                left, top = p, p
            case "top-right":
                left, top = out_width - w - p, p
            case "bottom-left":
                left, top = p, out_height - h - p
            case "bottom-right":
                left, top = out_width - w - p, out_height - h - p
            case "custom":
                left, top = int(x * out_width), int(y * out_height)
            case _:
                left, top = (out_width - w) // 2, (out_height - h) // 2

        final_img.paste(
            pil_img,
            (left, top),
        )
        return final_img

    def _transform_image(self, img, out_width: int, out_height: int, mat: list[float]):
        numpy_img = 255.0 * img.cpu().numpy()
        src_img = Image.fromarray(np.clip(numpy_img, 0, 255).astype(np.uint8))
        dst_img = Image.new("RGBA", (out_width, out_height))

        determinant = mat[0] * mat[4] - mat[3] * mat[1]
        if determinant == 0:
            return dst_img
        if determinant < 0:
            src_img = src_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        scale_x = math.sqrt(mat[0] ** 2 + mat[3] ** 2)
        scale_y = math.sqrt(mat[1] ** 2 + mat[4] ** 2)
        dimension = (int(scale_x * src_img.width), int(scale_y * src_img.height))

        angle = math.atan2(mat[3] / scale_y, mat[0] / scale_x)
        translation = (mat[2], mat[5])

        # Applies the transformation
        dst_img.paste(
            src_img.resize(dimension).rotate(angle * 180 / math.pi, expand=True),
            translation,
        )

        return dst_img
