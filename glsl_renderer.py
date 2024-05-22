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

    out vec4 fragColor;
"""


class GLSL:
    def __init__(self, images, out_width, out_height, shader, frame_rate, frame_count):
        ctx = moderngl.create_context(standalone=True, backend="egl", libgl="libGL.so.1", libegl="libEGL.so.1")
        program = ctx.program(VERTEX_SHADER, FRAGMENT_SHADER_HEADER + shader)

        vbo = ctx.buffer(
            np.array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1], dtype="f4").tobytes()
        )
        self.vao = ctx.simple_vertex_array(program, vbo, "iPosition")
        self.fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((out_width, out_height), 4)]
        )

        # Send image to shader
        image = self._resize_and_center_image(images[0], out_width, out_height)
        iChannel0 = ctx.texture(image.size, components=4, data=image.tobytes())
        iChannel0.use(location=0)

        # Uniform locations
        self.iTime = program.get("iTime", None)
        self.iFrame = program.get("iFrame", None)
        iResolution = program.get("iResolution", None)
        iTimeDelta = program.get("iTimeDelta", None)
        iDuration = program.get("iDuration", None)

        # Uniform initialization
        self.runtime = 0
        self.delta = 1.0 / frame_rate
        if iResolution:
            iResolution.value = (out_width, out_height)
        if iTimeDelta:
            iTimeDelta.value = self.delta
        if iDuration:
            iDuration.value = self.delta * frame_count

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

    def _resize_and_center_image(self, img, out_width, out_height):
        numpy_img = 255.0 * img.cpu().numpy()
        pil_img = Image.fromarray(np.clip(numpy_img, 0, 255).astype(np.uint8))
        final_img = Image.new("RGBA", (out_width, out_height))

        margin = 50
        # Put the image at the center while keeping aspect ratio
        if out_width / out_height < pil_img.width / pil_img.height:
            w = out_width - 2 * margin
            pil_img = pil_img.resize((w, math.ceil(w * pil_img.height / pil_img.width)))
        else:
            h = out_height - 2 * margin
            pil_img = pil_img.resize((math.ceil(h * pil_img.width / pil_img.height), h))

        final_img.paste(
            pil_img,
            ((out_width - pil_img.width) // 2, (out_height - pil_img.height) // 2),
        )
        return final_img
