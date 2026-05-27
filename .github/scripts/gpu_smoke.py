"""Tier-2 GPU smoke test: prove the built vLLM wheel runs real inference on
the RTX 5090 (Blackwell sm_120).

Unlike Tier-1 (`smoke_import.py`, import-only on a GPU-less hosted runner),
this loads a small model and generates tokens — exercising the sm_120 SASS
inside `vllm._C` end to end on actual hardware: weight load, CUDA kernel
launch, sampling. It needs vLLM's FULL runtime dependency closure installed
(not `--no-deps`), so the workflow installs the wheels with deps resolved.

A deliberately tiny model is used (fast to pull, small VRAM) — the point is
to validate the engine executes on this GPU, not to benchmark a production
model.
"""

import os
import sys

# Small, broadly-supported model. Override via env to test the real
# production models (Qwen3 AWQ-4bit chat, PaddleOCR-VL parser) on demand.
MODEL = os.environ.get("GPU_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GPU_MEM = float(os.environ.get("GPU_SMOKE_GPU_MEM", "0.5"))


def main() -> int:
    import torch

    print("torch:", torch.__version__)
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False — no GPU visible to this runner")
        return 1
    print("CUDA device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cap[0]}{cap[1]}")

    from vllm import LLM, SamplingParams

    # enforce_eager=True skips CUDA-graph capture: faster startup, lower VRAM,
    # and it's the forward-pass + kernel execution we care about for a smoke
    # test, not graph-mode perf.
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=2048,
        enforce_eager=True,
    )
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    text = out[0].outputs[0].text
    print("generated:", repr(text))
    assert text.strip(), "empty generation — engine ran but produced no tokens"

    print("GPU SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
