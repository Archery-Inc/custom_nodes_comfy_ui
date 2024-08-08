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

    out vec4 fragColor;
"""


class GLSL:
    def __init__(
        self,
        images,
        out_width,
        out_height,
        shader,
        frame_rate,
        frame_count,
        background,
        foreground,
        position,
    ):
        ctx = moderngl.create_context(
            standalone=True, backend="egl", libgl="libGL.so.1", libegl="libEGL.so.1"
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
        (image, l, r, b, t) = self._resize_and_center_image(
            images[0], out_width, out_height, position
        )
        iChannel0 = ctx.texture(image.size, components=4, data=image.tobytes())
        iChannel0.repeat_x = False
        iChannel0.repeat_y = False
        iChannel0.use(location=0)

        # Uniform locations
        self.iTime = program.get("iTime", None)
        self.iFrame = program.get("iFrame", None)
        iResolution = program.get("iResolution", None)
        left = program.get("textureLeft", None)
        right = program.get("textureRight", None)
        bottom = program.get("textureBottom", None)
        top = program.get("textureTop", None)
        iTimeDelta = program.get("iTimeDelta", None)
        iDuration = program.get("iDuration", None)
        backgroundColor = program.get("backgroundColor", None)
        foregroundColor = program.get("foregroundColor", None)

        # Uniform initialization
        self.runtime = 0
        self.delta = 1.0 / frame_rate
        if iResolution:
            iResolution.value = (out_width, out_height)
        if iTimeDelta:
            iTimeDelta.value = self.delta
        if iDuration:
            iDuration.value = self.delta * frame_count
        if left:
            left.value = l
        if right:
            right.value = r
        if bottom:
            bottom.value = b
        if top:
            top.value = t
        if backgroundColor:
            backgroundColor.value = self._hex_to_vec3(background)
        if foregroundColor:
            foregroundColor.value = self._hex_to_vec3(foreground)

    def render(self, frame_index, skip_frame):
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

    def _hex_to_vec3(self, hex):
        h = hex.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def _resize_and_center_image(self, img, out_width, out_height, position):
        numpy_img = 255.0 * img.cpu().numpy()
        pil_img = Image.fromarray(np.clip(numpy_img, 0, 255).astype(np.uint8))
        final_img = Image.new("RGBA", (out_width, out_height))

        # Put the image at the center while keeping aspect ratio
        margin_percent = 0.1
        w = int((1 - 2 * margin_percent) * out_width)
        h = math.ceil(w * pil_img.height / pil_img.width)
        if h > out_height:
            h = int((1 - 2 * margin_percent) * out_height)
            w = math.ceil(h * pil_img.width / pil_img.height)
        pil_img = pil_img.resize((w, h))

        padding_percent = 0.05
        p = int(padding_percent * out_width)
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
            case _:
                left, top = (out_width - w) // 2, (out_height - h) // 2

        right = left + pil_img.width
        bottom = top + pil_img.height
        final_img.paste(
            pil_img,
            (left, top),
        )
        return (
            final_img,
            left / out_width,
            right / out_width,
            top / out_height,
            bottom / out_height,
        )
