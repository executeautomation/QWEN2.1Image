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
- Existing Qwen logo design
- "QWEN IMAGE 2.1" text
- Typography and colors
- Overall composition and visual style

Make ONLY the following two changes:

CHANGE 1 — HAND GESTURE

Change one arm and hand so that the person extends an open palm
toward the Qwen logo, as if he is presenting or introducing
Qwen-Image-2.1.

The open palm should naturally point toward and visually lead
the viewer's attention toward the Qwen logo.

Make the arm position anatomically natural and consistent
with the person's body.

The hand must be photorealistic with exactly five naturally
formed fingers and realistic proportions.


CHANGE 2 — LIGHT FLARE ON QWEN LOGO

Add a cinematic light flare and subtle luminous glow around
the existing Qwen logo.

The light flare must originate from the Qwen logo itself.

Create a bright but controlled highlight around the logo,
with a soft radial bloom and subtle light rays extending
slightly outward.

The flare should make the Qwen logo feel illuminated and
visually important without obscuring the logo.

Keep the original Qwen logo completely recognizable.

Do NOT redesign, distort, replace, recolor or modify the
geometry of the Qwen logo.

The lighting effect should remain localized around the Qwen
logo.

Do NOT add light flares anywhere else in the image.

Do NOT add glow around the person.

Do NOT add glow around the text.

Do NOT change the overall background lighting.


IMPORTANT:

Preserve the person's identity exactly.

Preserve the existing Qwen logo exactly — only add lighting
around it.

Preserve all existing text exactly.

Do not redesign the background.

Do not add new objects.

Do not add additional text.

Do not create additional arms, hands or fingers.

Only change:
1. The arm/hand gesture.
2. The localized light flare around the Qwen logo.
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

edited.save("qwen21-thumbnail-ShowingHandFlareInLogo.png")

print("Saved qwen21-thumbnail-ShowingHandFlareInLogo.png")