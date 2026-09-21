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

# Load the previously generated image
original = Image.open("qwen21-thumbnail.png").convert("RGB")

prompt = """
Edit this image.

Keep the person exactly the same.
Preserve the person's face, expression, hairstyle and clothing.

Keep the existing composition and layout.

ONLY change the background to a futuristic dark AI laboratory
with subtle purple and blue lighting.

Keep the text "QWEN IMAGE 2.1" unchanged.

Do not add any additional text.
"""

print("Editing image...")

with torch.inference_mode():
    edited = pipe(
        prompt=prompt,
        image=original,
        width=1536,
        height=864,
        num_inference_steps=40,
        use_kv_cache=True,
    ).images[0]

edited.save("qwen21-thumbnail-edited.png")

print("Saved qwen21-thumbnail-edited.png")