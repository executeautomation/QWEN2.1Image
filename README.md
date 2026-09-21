# Qwen-Image-2.1 Thumbnail & Sticker Pipeline

A small collection of scripts that exercise **Qwen-Image-2.1** via
[HuggingFace `diffusers`](https://github.com/huggingface/diffusers) on Apple
Silicon (MPS). They generate and iteratively edit YouTube thumbnails, multi-model
comparison shots, and transparent PNG stickers.

---

## What's in here

| File | What it does |
|---|---|
| [qwenimage.py](qwenimage.py) | Pure text-to-image: a 1024×1024 YouTube thumbnail for "QWEN2.1 Image" with a speaker and the Qwen logo baked into the prompt. |
| [qwen_referenceimage.py](qwen_referenceimage.py) | 16:9 thumbnail using **two** reference images (`Karthik.png` speaker + `qwen.jpeg` logo). Speaker on the right, branding on the left, large "QWEN IMAGE 2.1" text. |
| [qwen-multreference.py](qwen-multreference.py) | 16:9 multi-model comparison thumbnail using **four** reference images: the speaker plus the official **Qwen**, **Gemma**, and **MiniMax** logos. |
| [qwen-modify.py](qwen-modify.py) | Image-edit pass on the basic thumbnail — swaps the background for a futuristic AI lab while preserving person + text + logo. |
| [qwen-modifyShowingHand.py](qwen-modifyShowingHand.py) | Edit pass: changes the speaker's arm/hand so an open palm presents the Qwen logo. Anatomy-preserving prompt. |
| [qwen-modifyShowingHandFlareInLogo.py](qwen-modifyShowingHandFlareInLogo.py) | Edit pass: adds the presenting-hand pose **and** a cinematic light flare around the existing Qwen logo. |
| [qwen-modifyShowingHandFlareInLogoFinal.py](qwen-modifyShowingHandFlareInLogoFinal.py) | Final polish pass — emphasizes "2.1", refines the logo glow, connects logo to the open palm via subtle purple light spill, adds rim-light to the person, and tightens depth/contrast. |
| [qwen-transparent.py](qwen-transparent.py) | Text-to-image with a natively transparent background — produces `qwen21-transparent-sticker.png`, a die-cut "LOCAL AI" robot sticker with a real alpha channel. |
| `reference/` | Local reference images used by the multi-reference scripts (`Karthik.png`, `qwen.jpeg`, `Gemma.png`, `MiniMax.png`). |

---

## Pipeline in one diagram

```
┌──────────────────────────────────────────────────────────────┐
│                QwenImage21Pipeline.from_pretrained()         │
│                       Qwen/Qwen-Image-2.1                    │
│                       torch_dtype = bfloat16                 │
└──────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   text-only prompt     reference images     single image
   (qwenimage.py,       (qwen_referenceimage, (qwen-modify*.py,
    qwen-transparent)    qwen-multreference)   editing passes)
            │                 │                 │
            └─────────────────┴─────────────────┘
                              ▼
            pipe(prompt, image=…, width, height,
                 num_inference_steps, use_kv_cache=True)
                              ▼
                  images[0].save("…-thumbnail.png")
```

---

## Hardware target

All scripts use `device = "mps"`, so they are designed for **Apple Silicon
(M1/M2/M3/M4) Macs**. The model is loaded in `torch.bfloat16` to fit comfortably
in unified memory.

> Other backends work too — change `device` to `"cuda"` for an NVIDIA GPU or
> `"cpu"` for a CPU-only run (very slow). On CUDA you may also want to enable
> `torch.compile` for a speed-up.

---

## Setup

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run will download `Qwen/Qwen-Image-2.1` from the HuggingFace Hub into
your local cache (~ several GB).

---

## Running the scripts

Each script is self-contained — run them individually in order to build up
the final thumbnail:

```bash
# 1) Base thumbnail with references
python qwen_referenceimage.py          # → qwen21-thumbnail.png

# 2) Iterative edits on the base thumbnail
python qwen-modify.py                                       # → qwen21-thumbnail-edited.png
python qwen-modifyShowingHand.py                            # → qwen21-thumbnail-edited.png
python qwen-modifyShowingHandFlareInLogo.py                 # → qwen21-thumbnail-ShowingHandFlareInLogo.png
python qwen-modifyShowingHandFlareInLogoFinal.py            # → qwen21-thumbnail-ShowingHandFlareInLogoFinal.png

# 3) Multi-model comparison shot
python qwen-multreference.py            # → multimodel-realistic-image-comparison.png

# 4) Pure text-to-image and transparent sticker
python qwenimage.py                     # → qwen21-thumnail.png  (sic, original filename)
python qwen-transparent.py              # → qwen21-transparent-sticker.png
```

> **Note** — most `*-modify*` scripts overwrite the same output filename
> (`qwen21-thumbnail-edited.png`) so run them in the order above if you want
> to preserve each intermediate result manually.

---

## Reference images

| File | Source | Used in |
|---|---|---|
| `Karthik.png` | speaker portrait | `qwen_referenceimage.py`, `qwen-multreference.py` |
| `qwen.jpeg` | official Qwen logo | `qwen_referenceimage.py`, `qwen-multreference.py` |
| `Gemma.png` | official Gemma logo | `qwen-multreference.py` |
| `MiniMax.png` | official MiniMax logo | `qwen-multreference.py` |

The reference images themselves are not bundled — drop your own copies into
`reference/` (filenames are case-sensitive in `qwen-multreference.py`).

---

## Why a `dev` diffusers?

`QwenImage21Pipeline` is brand-new, so the pipeline class only exists in the
latest development branch of `diffusers`. `requirements.txt` pins to a
`git+https://` install of `main`. Once the class ships in a tagged release,
swap the line for a normal version pin like `diffusers==0.41.0`.

---

## License

Scripts in this repo are provided as-is for experimentation with Qwen-Image-2.1.
The model weights are governed by the upstream
[Qwen-Image-2.1 license](https://huggingface.co/Qwen/Qwen-Image-2.1).