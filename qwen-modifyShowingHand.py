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
Edit the provided image while preserving the existing thumbnail design.

Keep everything in the image unchanged, including:

- The person's facial identity and appearance

- Facial expression

- Hairstyle

- Clothing

- Background

- Qwen logo

- "QWEN IMAGE 2.1" text

- Typography and colors

- Overall composition and visual style

ONLY modify the person's pose.

Change the person's arm and hand so that one hand is extended toward the Qwen logo with an open palm, as if he is presenting or introducing Qwen-Image-2.1.

The open palm should naturally point toward and visually lead to the Qwen logo.

Make the arm position anatomically natural and consistent with the person's body.

The hand must look photorealistic with five correctly formed fingers and natural proportions.

Do not change anything else in the thumbnail.

IMPORTANT:

Preserve the person's identity exactly.

Preserve the existing Qwen logo exactly.

Preserve all existing text exactly.

Do not redesign the background.

Do not add new objects.

Do not add additional text.

Do not create additional arms or hands.

Only change the arm and hand gesture.
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