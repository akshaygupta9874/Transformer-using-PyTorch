"""Toy at-init measurement: tied W, Pre-RMSNorm decoder, d=512, V=40000, 4 layers (spec's init recipe)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from research.tie_diagnostics import tied_grad_split

class RMS(nn.Module):
    def __init__(s, d): super().__init__(); s.weight = nn.Parameter(torch.ones(d))
    def forward(s, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x) * s.weight

class Block(nn.Module):
    def __init__(s, d, ff, h, std, nres):
        super().__init__()
        s.n1, s.n2, s.h = RMS(d), RMS(d), h
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.wo = nn.Linear(d, d, bias=False)
        s.g = nn.Linear(d, ff, bias=False); s.u = nn.Linear(d, ff, bias=False); s.dn = nn.Linear(ff, d, bias=False)
        for m in (s.qkv, s.g, s.u): nn.init.normal_(m.weight, 0, 0.02)
        for m in (s.wo, s.dn): nn.init.normal_(m.weight, 0, 0.02 / math.sqrt(2 * nres))
    def forward(s, x, rec):
        B, T, d = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        a = s.wo(a); rec.append(a.float().pow(2).mean().sqrt().item()); x = x + a
        m = s.dn(F.silu(s.g(s.n2(x))) * s.u(s.n2(x))); rec.append(m.float().pow(2).mean().sqrt().item())
        return x + m

class Toy(nn.Module):
    def __init__(s, V, d, L, std, scale):
        super().__init__()
        s.tgt_embed = nn.Embedding(V, d); nn.init.normal_(s.tgt_embed.weight, 0, std)
        s.proj = nn.Linear(d, V, bias=False); s.proj.weight = s.tgt_embed.weight
        s.blocks = nn.ModuleList(Block(d, 1024, 8, std, L) for _ in range(L)); s.fn = RMS(d); s.scale = scale; s.rec = []
    def decode(s, tgt):
        x = s.tgt_embed(tgt) * s.scale; s.emb_rms = x.float().pow(2).mean().sqrt().item(); s.rec = []
        for b in s.blocks: x = b(x, s.rec)
        s.pre_rms = x.float().pow(2).mean().sqrt().item(); return s.fn(x)

V, d = 40000, 512
g = torch.Generator().manual_seed(0)
ranks = torch.arange(1, V).float(); pz = 1 / ranks; pz /= pz.sum()
tgt_in = torch.multinomial(pz, 16 * 64, replacement=True, generator=g).view(16, 64) + 1
tgt_out = torch.multinomial(pz, 16 * 64, replacement=True, generator=g).view(16, 64) + 1
batch = {"tgt_in": tgt_in, "tgt_out": tgt_out}
fh = lambda m, b: (m.decode(b["tgt_in"]), b["tgt_out"], None)
print(f"{'std':>7} {'scale':>6} | emb_rms sublayer_rms pre_norm_rms | ||g_emb|| ||g_proj|| ratio cos | init logit std")
for std in (0.02, 0.5 ** 0.5 / 16):
    for scale in (1.0, math.sqrt(512)):
        torch.manual_seed(0)
        m = Toy(V, d, 4, std, scale)
        r = tied_grad_split(m, batch, fh, torch.device("cpu"))
        with torch.no_grad(): h = m.decode(tgt_in); lg = (h @ m.proj.weight.t()).std().item()
        sub = sum(m.rec) / len(m.rec)
        print(f"{std:7.4f} {scale:6.2f} | {m.emb_rms:7.3f} {sub:11.4f} {m.pre_rms:11.3f} | {r['norm_emb_path']:.3e} {r['norm_proj_path']:.3e} {r['ratio_emb_over_proj']:6.2f} {r['cosine']:+.3f} | {lg:.3f}")
