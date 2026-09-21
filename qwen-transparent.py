import torch
from diffusers import QwenImage21Pipeline

device = "mps"

print("Loading Qwen-Image-2.1...")

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to(device)

print("Generating transparent sticker...")

prompt = """
Create a premium transparent sticker for an AI developer.

The sticker should feature a cute futuristic AI robot holding a small
laptop and coding.

The robot should have:
- a modern friendly design
- subtle purple and violet lighting
- small glowing AI elements
- a laptop with code visible on the screen
- clean professional details

Add the text:

"LOCAL AI"

underneath the robot using bold modern typography.

Make the entire design look like a high-quality die-cut technology sticker.

IMPORTANT:

Generate the image natively with a transparent background.

The output must contain a real alpha channel (RGBA).

There must be NO background behind the sticker.

Do not generate a white background.
Do not generate a black background.
Do not generate a colored background.
Do not generate a checkerboard pattern.
Do not simulate transparency visually.

Everything outside the sticker must be genuinely transparent.

Keep clean, crisp edges around the robot and typography.

Include a subtle white sticker border around the complete design.

The final output should look like a professional transparent PNG sticker
that could be placed directly over another image, presentation, website,
or YouTube thumbnail.
"""

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=40,
    ).images[0]

print("Output mode:", image.mode)

image.save("qwen21-transparent-sticker.png")

print("Saved qwen21-transparent-sticker.png")