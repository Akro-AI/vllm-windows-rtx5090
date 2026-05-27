"""Release smoke test: validate the published wheel set imports cleanly.

Runs on a GPU-less GitHub-hosted runner, so it asserts only that the
native extensions LOAD (DLL/ABI resolution) — not that CUDA is usable.
This catches the two failure modes the wheel build can produce:

  1. torch / torchvision / torchaudio ABI mismatch — torchvision and
     torchaudio link against torch's C++ ABI; if the triple is not
     mutually consistent, importing them raises.
  2. vllm._C DLL load failure — the compiled vLLM extension failing to
     load against the bundled torch was the original "ImportError: DLL
     load failed while importing _C" that motivated building the wheel
     set in-house.

It deliberately does NOT `import vllm` (which would run vllm/__init__.py
and pull vLLM's full Windows dependency closure — out of scope here and
flaky to resolve on a hosted runner). Full-stack import + inference is
validated downstream in Socrates / on a GPU runner.
"""

import glob
import importlib.util
import os
import sys
import sysconfig
import types

print("Python:", sys.version)

import torch  # noqa: E402

print("torch:", torch.__version__)
import torchvision  # noqa: E402

print("torchvision:", torchvision.__version__)
import torchaudio  # noqa: E402

print("torchaudio:", torchaudio.__version__)

site = sysconfig.get_paths()["purelib"]
vllm_dir = os.path.join(site, "vllm")
if not os.path.isdir(vllm_dir):
    raise SystemExit(f"FAIL: vllm package dir not found at {vllm_dir}")

# Direct-load the compiled extension without running vllm/__init__.py.
# Stub the parent package so Python lets us load the dotted submodule;
# torch is already imported so its runtime DLLs are on the search path.
pkg = types.ModuleType("vllm")
pkg.__path__ = [vllm_dir]
sys.modules["vllm"] = pkg

cands = glob.glob(os.path.join(vllm_dir, "_C*.pyd"))
if not cands:
    raise SystemExit(f"FAIL: vllm._C extension (.pyd) not found in {vllm_dir}")

pyd = cands[0]
print("loading:", pyd)
spec = importlib.util.spec_from_file_location("vllm._C", pyd)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("vllm._C loaded OK")

print("SMOKE TEST PASSED")
