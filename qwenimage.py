import torch
from diffusers import QwenImage21Pipeline

device = "mps"

print("Loading Qwen-Image-2.1...")

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to(device)

print("Generating...")

prompt = """
Create an youtube thumbnail for a video titled "QWEN2.1 Image"
The thumnail should feature a man talking about it
Make sure to include QWEN Logo in the thumbnail
Add some text that says "QWEN2.1 Image" in the thumbnail

"""

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=40,
    ).images[0]

image.save("qwen21-thumnail.png")

print("Saved qwen21-thumnail.png")