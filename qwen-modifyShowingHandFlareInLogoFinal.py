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
Enhance the provided YouTube thumbnail while preserving its existing
composition, person, branding, text and overall design.

Do NOT redesign the thumbnail.

Keep the person's facial identity exactly the same.
Keep the person's pose and open hand exactly as they are.
Keep the Qwen logo design exactly the same.
Keep all existing text exactly spelled as:

QWEN
IMAGE 2.1

Make ONLY the following visual enhancements:


1. EMPHASIZE "2.1"

Make "2.1" the strongest purple visual accent in the headline.

Keep "QWEN" bright white.

Keep "IMAGE" predominantly white or very light lavender.

Make "2.1" brighter and more distinctly purple with a subtle luminous
highlight.

The viewer should immediately understand that the important message is:

QWEN IMAGE → 2.1


2. IMPROVE THE QWEN LOGO LIGHTING

Keep the existing Qwen logo completely unchanged.

Refine the existing light flare around the Qwen logo.

Create a concentrated premium purple-white glow originating directly
from the logo.

Reduce excessive starburst rays.

Use a controlled soft bloom around the logo with subtle cinematic
light rays.

The logo itself must remain sharp, readable and clearly visible.


3. CONNECT THE LOGO TO THE OPEN HAND

Add a very subtle purple light spill from the illuminated Qwen logo
onto the person's open palm.

The palm should catch a small amount of realistic purple-white light,
as though the glowing Qwen logo is illuminating the hand.

Do NOT create a glowing hand.

Do NOT add an energy ball.

Do NOT add lightning.

The effect should be subtle and photorealistic.


4. ADD SUBTLE RIM LIGHT TO THE PERSON

Add a thin, realistic purple-blue rim light along the left-facing edge
of the person's body where he faces the Qwen graphics.

Apply the rim light subtly along portions of the:

shoulder,
arm,
hair edge,
and side of the face.

The lighting should visually integrate the person with the purple
Qwen environment.

Do not alter skin tone.


5. ENHANCE THE PERSON

Slightly improve facial clarity, local contrast and sharpness.

Make the eyes and facial details slightly more visually prominent for
a YouTube thumbnail.

Preserve the person's facial identity EXACTLY.

Do not beautify or reshape the face.
Do not change facial features.
Do not change expression.
Do not change hairstyle.
Do not change clothing.


6. IMPROVE DEPTH

Keep the existing dark blue/purple background.

Slightly increase separation between:

foreground person,
headline,
Qwen logo,
background Qwen graphic.

Use subtle lighting and depth rather than adding new objects.

Keep the large background Qwen symbol understated so it does not
compete with the person or headline.


IMPORTANT:

Do not change the composition.
Do not move the person.
Do not change the hand pose.
Do not add additional hands or fingers.
Do not replace or redesign the Qwen logo.
Do not change any words.
Do not add additional text.
Do not add icons.
Do not add objects.
Do not add borders.
Do not add unnecessary effects.

The final result should look like a polished, high-CTR technology
YouTube thumbnail while remaining clean and professional.

The visual hierarchy should be:

1. QWEN IMAGE 2.1
2. Person's face
3. Illuminated Qwen logo
4. Open hand presenting the logo
5. Background graphics
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

edited.save("qwen21-thumbnail-ShowingHandFlareInLogoFinal.png")

print("Saved qwen21-thumbnail-ShowingHandFlareInLogoFinal.png")