"""Step2: same-weight logprob recomputation must match (π_θ ≈ π_rollout).

Run on head:
  CUDA_VISIBLE_DEVICES=0 ${SAO_WS}/tmp/phase1-venv/bin/python \\
    tests/test_logprob_align.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ROOT = os.environ.get("MODEL_ROOT")
MODEL = os.environ.get(
    "SAO_MODEL",
    str(Path(MODEL_ROOT) / "Qwen3-4B-Instruct-2507") if MODEL_ROOT else "",
)
MAX_NEW = int(os.environ.get("SAO_MAX_NEW", "32"))
ATOL = float(os.environ.get("SAO_LOGPROB_ATOL", "1e-4"))


def token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token logprob of input_ids[t] given prefix [:t], for t=1..L-1; pad t=0 with 0."""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # [1, L, V]
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = input_ids[:, 1:].unsqueeze(-1)
    tok = logp.gather(-1, tgt).squeeze(-1)  # [1, L-1]
    pad = torch.zeros(input_ids.size(0), 1, device=tok.device, dtype=tok.dtype)
    return torch.cat([pad, tok], dim=-1)  # [1, L]


def main() -> None:
    assert torch.cuda.is_available(), "need GPU"
    if not MODEL:
        raise SystemExit("set SAO_MODEL or MODEL_ROOT")
    device = torch.device("cuda:0")
    print("model", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt = "Compute 17+25. Answer with the number only.\n"
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        return_dict_in_generate=True,
    )
    seq = out.sequences  # [1, L]
    lp1 = token_logprobs(model, seq)
    lp2 = token_logprobs(model, seq)
    gen_slice = slice(inputs["input_ids"].shape[1], seq.shape[1])
    diff = (lp1[:, gen_slice] - lp2[:, gen_slice]).abs().max().item()
    print("gen_tokens", gen_slice.stop - gen_slice.start)
    print("max_abs_diff_recompute", diff)
    text = tok.decode(seq[0, gen_slice], skip_special_tokens=True)
    print("sample:", repr(text[:200]))
    if diff > ATOL:
        raise SystemExit(f"FAIL: recompute diff {diff} > {ATOL}")
    print("OK: logprob align (same-weight recompute)")


if __name__ == "__main__":
    # allow importing nothing from sao package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
