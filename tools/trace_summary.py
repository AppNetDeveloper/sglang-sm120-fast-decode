"""Suma el tiempo de GPU por kernel de una traza de torch.profiler (SGLang /start_profile). Uso: analiza_traza.py traza.json.gz pasos"""
import gzip, json, sys, re, collections
f, pasos = sys.argv[1], int(sys.argv[2])
d = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
tot = collections.Counter(); cnt = collections.Counter()
for e in ev:
    n = e["name"]; tot[n] += e["dur"]; cnt[n] += 1
T = sum(tot.values())
def cat(n):
    l = n.lower()
    for k, pats in [("GEMM fp4", ["fp4", "nvfp4", "blockscaled", "e2m1"]), ("GEMM fp8", ["fp8", "e4m3", "sm89_xmma", "f8"]),
                    ("GDN/mamba", ["gdn", "chunk_", "fused_recurrent", "gated_delta", "causal_conv", "mamba", "ssm", "replay"]),
                    ("atención", ["flashinfer", "attention", "batchprefill", "batchdecode", "xqa", "fmha", "paged"]),
                    ("GEMM bf16/cublas", ["gemm", "gemv", "cutlass", "sm80", "sm90", "wmma", "splitk", "cublas", "matmul"]),
                    ("muestreo/topk", ["topk", "sample", "softmax", "argmax", "sort"]),
                    ("copias/memset", ["memcpy", "memset", "copy"]), ("norm/act/elementwise", ["norm", "elementwise", "silu", "act", "rotary", "rope", "vectorized", "reduce"])]:
        if any(p in l for p in pats): return k
    return "otros"
C = collections.Counter(); 
for n, t in tot.items(): C[cat(n)] += t
print(f"kernels: {len(ev)}  ({len(ev)/pasos:.0f}/paso)  GPU total {T/1000:.1f} ms  ({T/1000/pasos:.2f} ms/paso)")
for k, t in C.most_common(): print(f"  {k:24s} {t/1000/pasos:7.3f} ms/paso  {100*t/T:5.1f}%")
print("top 25 kernels:")
for n, t in tot.most_common(25): print(f"  {t/1000/pasos:7.3f} ms/paso  x{cnt[n]/pasos:6.1f}  {100*t/T:5.1f}%  {n[:150]}")
if ev:
    t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev)
    print(f"ventana {(t1-t0)/1000:.1f} ms, GPU ocupada {100*T/(t1-t0):.1f}%")
