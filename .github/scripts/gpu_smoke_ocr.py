"""Tier-2 OCR smoke test: prove the bundled vLLM actually runs the production
parser VLM (PaddleOCR-VL-1.5) on the RTX 5090 (sm_120).

Unlike gpu_smoke.py (a text-only chat model), this loads the real fast-mode
parser, renders an image with known text, and runs a multimodal generate via
vLLM's `LLM.chat` API (which applies the model's registered chat template, so
we don't hand-build the image-placeholder prompt). Validates: VLM loads with
trust_remote_code on sm_120, multimodal pipeline runs, and OCR reads the text.

PASS gate is "engine initialised + non-empty generation" (proves the model
runs on this wheel/GPU). Exact-text recognition is reported as a soft signal —
OCR accuracy/prompt-tuning is an apollo-integration concern, not a wheel test.
"""

import base64
import io
import os
import sys

MODEL = os.environ.get("OCR_SMOKE_MODEL") or "PaddlePaddle/PaddleOCR-VL-1.5"
GPU_MEM = float(os.environ.get("OCR_SMOKE_GPU_MEM") or "0.45")
EXPECT = "HELLO 12345"


def make_text_image(text: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 120), "white")
    draw = ImageDraw.Draw(img)
    # Default bitmap font — no external font file needed on the runner.
    draw.text((20, 45), text, fill="black")
    return img


def main() -> int:
    import torch

    print("torch:", torch.__version__)
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print("CUDA device:", torch.cuda.get_device_name(0), f"(sm_{cap[0]}{cap[1]})")

    from vllm import LLM, SamplingParams

    img = make_text_image(EXPECT)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # trust_remote_code: PaddleOCR-VL ships custom modeling code. enforce_eager
    # keeps startup quick + VRAM low for a smoke test.
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=4096,
        enforce_eager=True,
    )

    # LLM.chat applies the model's chat template (handles the image placeholder
    # correctly) — more robust than hand-building the prompt.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "OCR the text in this image."},
            ],
        }
    ]
    out = llm.chat(messages, SamplingParams(max_tokens=64, temperature=0.0))
    text = out[0].outputs[0].text
    print("OCR output:", repr(text))

    assert text.strip(), "engine ran but produced no output"

    norm = "".join(text.upper().split())
    hit = "HELLO" in norm or "12345" in norm
    print(f"recognized expected text: {hit}  (expected ~{EXPECT!r})")

    print("OCR SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
