import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.modern import build_modern_transformer
from research.profile_sweep import sweep, find_max_tokens, synthetic_batch, run_once
from research.accum import default_forward_hidden

def model_fn(**kwargs):
    return build_modern_transformer(
        src_vocab_size=16000,
        tgt_vocab_size=40000,
        d_model=512,
        encoder_layers=6,
        decoder_layers=4,
        heads=8,
        d_ff=2048,
        dropout=0.1,
        tie_embeddings=True,
        **kwargs
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running profile sweep on device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    tokens_list = [2048, 3072, 4096]
    results = []
    for max_tok in tokens_list:
        print(f"\n--- Testing max_tokens={max_tok} ---")
        rows = sweep(model_fn, [{"kv_heads": None}, {"kv_heads": 4}], max_tokens=max_tok, device=device)
        results.extend(rows)

    print("\n--- Summary Table ---")
    print(f"{'kv_heads':<10} | {'max_tokens':<10} | {'params_M':<10} | {'step_ms':<10} | {'tok/s':<10} | {'peak_alloc_gb':<14} | {'peak_res_gb':<12}")
    print("-" * 85)
    for r in results:
        kv = "null" if r["setting"]["kv_heads"] is None else str(r["setting"]["kv_heads"])
        print(f"{kv:<10} | {r['max_tokens']:<10} | {r.get('params_M', 0):<10.2f} | {r.get('step_ms', 0):<10.1f} | {r.get('tok_per_s', 0):<10.1f} | {r.get('peak_alloc_gb', 0):<14.2f} | {r.get('peak_reserved_gb', 0):<12.2f}")

    if device.type == "cuda":
        print("\n--- Finding Max Tokens (Full MHA, kv_heads=None) ---")
        def fits_mha(toks: int) -> bool:
            try:
                m = model_fn(kv_heads=None).to(device)
                b = synthetic_batch(toks)
                run_once(m, b, device, steps=2, warmup=1)
                del m, b
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                torch.cuda.empty_cache()
                return False

        max_toks_mha = find_max_tokens(fits_mha, lo=2048, hi=16384, granularity=512)
        print(f"find_max_tokens (kv_heads=null): {max_toks_mha}")
