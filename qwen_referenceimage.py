import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

device = "mps"

print("Loading Qwen-Image-2.1...")

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to(device)

# ---------------------------------------------------------
# Load reference images
# ---------------------------------------------------------

# Image 1 = speaker/person
speaker = Image.open("./reference/Karthik.png").convert("RGB")

# Image 2 = Qwen logo
qwen_logo = Image.open("./reference/qwen.jpeg").convert("RGB")

reference_images = [
    speaker,
    qwen_logo
]

print("Generating using reference images...")

prompt = """
Create a professional 16:9 YouTube thumbnail for a video about Qwen-Image-2.1.

Use the PERSON from reference image 1 as the main speaker.
Preserve his facial identity and appearance.

Use the Qwen logo from reference image 2 prominently in the thumbnail.

Place the speaker on the right side of the thumbnail.
Place the Qwen branding and visual elements on the left side.

Add large, bold, highly readable text:

"QWEN IMAGE 2.1"

Use dramatic professional studio lighting and strong visual separation.
Modern AI/technology aesthetic.
High-quality YouTube thumbnail composition.
"""

with torch.inference_mode():

    image = pipe(
        prompt=prompt,

        # Multiple reference images
        image=reference_images,

        # 16:9 YouTube format
        width=1536,
        height=864,

        num_inference_steps=40,

        # Qwen 2.1 uses KV cache by default
        use_kv_cache=True,

    ).images[0]

image.save("qwen21-thumbnail.png")

print("Saved qwen21-thumbnail.png")