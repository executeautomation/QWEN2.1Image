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

# ============================================================
# Load 4 reference images
# ============================================================

# Reference 1:
original = Image.open("./reference/karthik.png").convert("RGB")

# Reference 2:
# Official Qwen logo
qwen_logo = Image.open("./reference/qwen.jpeg").convert("RGB")

# Reference 3:
# Official Gemma logo
gemma_logo = Image.open("./reference/gemma.png").convert("RGB")

# Reference 4:
# Official MiniMax logo
minimax_logo = Image.open("./reference/minimax.png").convert("RGB")

reference_images = [
    original,
    qwen_logo,
    gemma_logo,
    minimax_logo,
]

# ============================================================
# Prompt
# ============================================================

prompt = """
You are given exactly four reference images.

REFERENCE IMAGE 1:
This is the SPEAKER.
Use ONLY this image as the identity reference for the person.
Preserve his face, hairstyle, skin tone and appearance.

REFERENCE IMAGE 2:
This is the official QWEN LOGO.
Use this exact visual design whenever displaying Qwen.

REFERENCE IMAGE 3:
This is the official GEMMA LOGO.
Use this exact visual design whenever displaying Gemma.

REFERENCE IMAGE 4:
This is the official MINIMAX LOGO.
Use this exact visual design whenever displaying MiniMax.

Create a completely new 16:9 YouTube thumbnail.

Place the speaker from REFERENCE IMAGE 1 on the RIGHT.

On the LEFT add the headline:

BEST
REALISTIC
IMAGE?

Use bold Vox-style typography.
Make REALISTIC large and blue.
Make BEST and IMAGE? white.

Each logo must appear exactly ONCE.

Qwen = Reference Image 2
Gemma = Reference Image 3
MiniMax = Reference Image 4

Do not confuse the reference images.
Do not use the Qwen logo for Gemma.
Do not use the Qwen logo for MiniMax.
Do not duplicate any logo.
Do not redesign any logo.
Do not merge the logos.
Do not invent logos.

Use a premium dark background with subtle blue and purple lighting.

Keep the composition clean and suitable for a professional YouTube thumbnail.
Do not declare a winner.
Do not add benchmark numbers.
Do not add additional text.
"""

print("Generating multi-model comparison thumbnail...")

with torch.inference_mode():
    edited = pipe(
        prompt=prompt,
        image=reference_images,
        width=1536,
        height=864,
        num_inference_steps=50,
        use_kv_cache=True,
    ).images[0]

edited.save("multimodel-realistic-image-comparison.png")

print("Saved multimodel-realistic-image-comparison.png")