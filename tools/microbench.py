"""Microbench de GEMMs de decode (Qwen3.8-27B ThinkingCap A4) en CUDA graph, SM120. SOLO en la ventana.
Pesos sintéticos con las formas reales; L2 frío rotando NCOPIAS copias (>256 MB por forma).
Uso: PYTHONPATH=/workspace/kdev/sglang_src /workspace/sgl0521/bin/python microbench.py [--solo ba|fp8|fp4] [--m 1,4,8,16]
  --solo fp8 : barrido de tiles triton_scaled_mm en CADA forma FP8 real; escribe los JSON de config afinada en
               /workspace/kdev/fp8_configs/ y los INSTALA en la copia (sglang_src/.../gemm/configs) solo donde
               Triton gana a CUTLASS; al final comprueba que apply_fp8_linear despacha a Triton.
  --solo fp4 : KDA NVFP4 por la puerta pública (allowlist parcheada) en M=1..16 vs flashinfer cutlass.
Imprime µs/llamada y GB/s efectivos (bytes de peso / tiempo) frente a 1790 GB/s.
Toda la salida va también a /workspace/kdev/mb_<fecha>.txt.
"""
import argparse, os, shutil, traceback
import torch
import torch.nn.functional as F

BW = 1790.0
DEV = "cuda"


def graph_us(fn, reps=20, iters=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (reps * iters)


class Rot:
    """Rota entre copias para que cada llamada lea el peso desde DRAM (como en el servidor)."""
    def __init__(self, make, nbytes):
        self.n = max(2, int(384e6 // max(nbytes, 1)) + 1)
        self.n = min(self.n, 64)
        self.items = [make() for _ in range(self.n)]
        self.i = 0
    def next(self):
        self.i = (self.i + 1) % self.n
        return self.items[self.i]


def report(tag, M, us, nbytes, ref=None):
    gbs = nbytes / us / 1e3
    extra = f"  vs actual x{ref/us:4.2f}" if ref else ""
    print(f"  {tag:34s} M={M:2d} {us:8.1f} us  {gbs:6.0f} GB/s ({100*gbs/BW:4.0f}%){extra}", flush=True)
    return us


def try_bench(tag, M, fn, nbytes, ref=None, check=None):
    try:
        if check is not None:
            check()
        return report(tag, M, graph_us(fn), nbytes, ref)
    except Exception as e:
        print(f"  {tag:34s} M={M:2d} NO DISPONIBLE: {type(e).__name__}: {str(e).splitlines()[0][:140]}")
        return None


# ---------------- bf16 in_proj_ba [96, 5120] ----------------
import triton, triton.language as tl

@triton.jit
def _splitk_bf16(x_ptr, w_ptr, out_ptr, M, N, K, sxm, swn, som,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    kper = K // SPLIT; k0 = pk * kper
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k0, k0 + kper, BK):
        rk = k + tl.arange(0, BK)
        xb = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=rm[:, None] < M, other=0.0)
        wb = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=rn[:, None] < N, other=0.0)
        acc += tl.dot(xb, tl.trans(wb))
    tl.atomic_add(out_ptr + rm[:, None] * som + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N), sem="relaxed")


def splitk_bf16(x, w, split=40, bn=32, bk=128):
    M, K = x.shape; N = w.shape[0]
    acc = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    _splitk_bf16[(triton.cdiv(N, bn), split)](x, w, acc, M, N, K, x.stride(0), w.stride(0), acc.stride(0),
                                             BM=16, BN=bn, BK=bk, SPLIT=split, num_warps=4)
    return acc.to(torch.bfloat16)


def bench_ba(Ms):
    print("\n== in_proj_ba bf16 [96 x 5120] (48 llamadas/paso) ==")
    N, K = 96, 5120; nb = N * K * 2
    rot = Rot(lambda: torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02, nb)
    # qkvz FP8 [5120 -> 16384] para la prueba en paralelo (alt_stream como en Qwen3_5GatedDeltaNet)
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
    Nq = 16384
    rq = Rot(lambda: ((torch.randn(Nq, K, device=DEV) * 0.05).to(torch.float8_e4m3fn).t(),
                      torch.rand(Nq, 1, device=DEV) * 0.01), Nq * K)
    alt = torch.cuda.Stream()
    for M in Ms:
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        ref = try_bench("cuBLAS F.linear (actual)", M, lambda: F.linear(x, rot.next()), nb)
        w0 = rot.items[0]
        out_ref = F.linear(x, w0).float()
        def chk(f):
            def c():
                d = (f(x, w0).float() - out_ref).abs().max().item()
                if d > 2e-2: raise RuntimeError(f"no coincide: {d}")
            return c
        try:
            from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16, can_use_tiny_gemm
            if can_use_tiny_gemm(N, K, 16):
                try_bench("tiny_gemm_bf16 (JIT sglang)", M, lambda: tiny_gemm_bf16(x, rot.next(), max_m=16), nb, ref,
                          chk(lambda a, b: tiny_gemm_bf16(a, b, max_m=16)))
            else:
                print("  tiny_gemm_bf16: forma no soportada")
        except Exception as e:
            print("  tiny_gemm_bf16 NO DISPONIBLE:", e)
        if M == 1:
            try:
                from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
                try_bench("hopper_bf16_gemv (= --bf16 gemv)", M, lambda: hopper_bf16_gemv(x, rot.next()), nb, ref,
                          chk(hopper_bf16_gemv))
            except Exception as e:
                print("  hopper_bf16_gemv NO DISPONIBLE:", e)
        for split in (20, 40):
            try_bench(f"triton split-K {split}", M, lambda s=split: splitk_bf16(x, rot.next(), s), nb, ref,
                      chk(lambda a, b, s=split: splitk_bf16(a, b, s)))
        # en paralelo con qkvz FP8 como en el servidor
        xq = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        def qkvz():
            w, s = rq.next(); return apply_fp8_linear(xq, w, s, None, use_per_token_if_dynamic=True, compressed_tensor_quant=True)
        tq = try_bench("qkvz FP8 solo [5120->16384]", M, qkvz, Nq * K)
        def par(ba_fn):
            def f():
                cur = torch.cuda.current_stream(); alt.wait_stream(cur)
                qkvz()
                with torch.cuda.stream(alt): ba_fn()
                cur.wait_stream(alt)
            return f
        tp = try_bench("qkvz || cuBLAS ba (actual)", M, par(lambda: F.linear(x, rot.next())), Nq * K + nb)
        tps = try_bench("qkvz || triton split-K 40 ba", M, par(lambda: splitk_bf16(x, rot.next(), 40)), Nq * K + nb)
        if tq and tp:
            print(f"  -> coste real de ba en paralelo: {tp - tq:6.1f} us/capa (x48 = {48*(tp-tq)/1000:.2f} ms/paso)"
                  + (f"; con split-K: {tps - tq:6.1f} us/capa" if tps else ""))


# ---------------- FP8 per-token / per-channel (compressed-tensors) ----------------
# (nombre, K, N, llamadas/paso de verify). Formas leídas de las cabeceras del checkpoint A4.
FP8_SHAPES = [
    ("GDN in_proj_qkvz (qkv 10240 + z 6144)", 5120, 16384, 48),
    ("GDN out_proj / attn o_proj", 6144, 5120, 64),
    ("attn qkv (q+gate 12288, k 1024, v 1024)", 5120, 14336, 16),
    ("mlp gate_up L56-63", 5120, 34816, 8),
    ("mlp down L56-63", 17408, 5120, 8),
    ("lm_head verify", 5120, 248320, 1),
    ("lm_head borrador FR-Spec 40960", 5120, 40960, 3),
    ("lm_head borrador FR-Spec mix64", 5120, 45376, 3),
]
FP8_EXTRA_M = (2, 8, 12, 16, 24, 32, 48, 64)  # se miden con los 6 mejores tiles de la fase 1
FP8_NULL_M = (96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384)  # centinelas -> CUTLASS
CFG_OUT = os.environ.get("FP8_CFG_OUT", "./fp8_configs_out")
CFG_INSTALL = os.environ.get("FP8_CFG_INSTALL", "")  # path to <sglang>/kernels/ops/gemm/configs to auto-install winners; empty = do not install
MIN_WIN = 0.97  # Triton entra solo si su GEMM tarda <= 97 % de la de CUTLASS


def _fp8_cfg_name(N, K):
    from sglang.srt.utils.common import get_device_name
    return f"N={N},K={K},device_name={get_device_name().replace(' ', '_')},dtype=fp8_w8a8_channelwise.json"


def bench_fp8(Ms, rapido=False):
    import json, os, shutil
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
    from sglang.kernels.ops.gemm import fp8_scaled_mm
    from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
    print("\n== FP8 W8A8 per-token x per-channel: barrido triton_scaled_mm vs CUTLASS (solo GEMM; la cuantización es común) ==")
    os.makedirs(CFG_OUT, exist_ok=True)
    tiles = [(bn, bk, nw, st) for bn in (16, 32, 64) for bk in (128, 256, 512) for nw in (4, 8) for st in (3, 4)] + [(128, 256, 8, 4), (32, 256, 4, 5)]
    resumen = []
    for name, K, N, calls in FP8_SHAPES:
        nb = N * K
        rot = Rot(lambda: ((torch.randn(N, K, device=DEV) * 0.05).to(torch.float8_e4m3fn).t(),
                           (torch.rand(N, 1, device=DEV) * 0.01 + 1e-3)), nb)
        print(f"\n {name}  K={K} N={N}  ({calls}/paso)  copias L2-frío={rot.n}", flush=True)
        w0, s0 = rot.items[0]
        cfg = {}
        def measure(M, tile_list):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            qx, xs = sglang_per_token_quant_fp8(x)
            ref_o = fp8_scaled_mm(qx, w0, xs, s0, torch.bfloat16).float()
            tol = 2e-3 * ref_o.abs().max().item() + 1e-3
            def cut():
                w, s = rot.next(); return fp8_scaled_mm(qx, w, xs, s, torch.bfloat16)
            tc = graph_us(cut, reps=10, iters=10)
            bm = max(16, triton.next_power_of_2(M))
            res = []
            for (bn, bk, nw, st) in tile_list:
                kw = dict(block_size_m=bm, block_size_n=bn, block_size_k=bk, use_heuristic=False, num_warps=nw, num_stages=st)
                try:
                    o = triton_scaled_mm(qx, w0, xs, s0, torch.bfloat16, None, **kw).float()
                    d = (o - ref_o).abs().max().item()
                    if d > tol:
                        continue
                    def f(kw=kw):
                        w, s = rot.next(); return triton_scaled_mm(qx, w, xs, s, torch.bfloat16, None, **kw)
                    res.append((graph_us(f, reps=10, iters=10), (bn, bk, nw, st), d))
                except Exception:
                    pass
            res.sort()
            return tc, bm, res
        # fase 1: barrido completo a las M pedidas
        top = {}
        for M in Ms:
            tc, bm, res = measure(M, tiles)
            for t, tl_, d in res[:6]: top[tl_] = min(top.get(tl_, 1e9), t)
            _apunta(cfg, name, M, tc, bm, res, nb, resumen)
        # fase 2: el resto de M con los 6 mejores tiles
        if not rapido:
            best6 = sorted(top, key=top.get)[:6]
            for M in FP8_EXTRA_M:
                if M in Ms: continue
                tc, bm, res = measure(M, best6)
                _apunta(cfg, name, M, tc, bm, res, nb, resumen)
        for M in FP8_NULL_M:
            cfg.setdefault(str(M), None)
        fname = _fp8_cfg_name(N, K)
        out = {k: cfg[k] for k in sorted(cfg, key=int)}
        json.dump(out, open(os.path.join(CFG_OUT, fname), "w"), indent=1)
        gana = [k for k, v in out.items() if v]
        if gana and CFG_INSTALL:
            shutil.copy(os.path.join(CFG_OUT, fname), os.path.join(CFG_INSTALL, fname))
            print(f"  -> JSON instalado ({len(gana)} M con Triton: {','.join(gana)}): {CFG_INSTALL}/{fname}")
        else:
            p = os.path.join(CFG_INSTALL, fname) if CFG_INSTALL else ''
            if p and os.path.exists(p): os.remove(p)
            print(f"  -> CUTLASS gana en todas las M: sin JSON instalado ({CFG_OUT}/{fname} solo de registro)")
        del rot; torch.cuda.empty_cache()
    _verifica_despacho_fp8()
    print("\n RESUMEN FP8 (ahorro por paso a 1 petición = M=4 verify):")
    for r in resumen: print("  ", r)


def _apunta(cfg, name, M, tc, bm, res, nb, resumen):
    if not res:
        print(f"  M={M:2d} cutlass {tc:7.1f} us ({nb/tc/1e3:5.0f} GB/s)  triton: ningún tile válido -> null"); cfg[str(M)] = None; return
    t, (bn, bk, nw, st), d = res[0]
    win = t <= MIN_WIN * tc
    print(f"  M={M:2d} cutlass {tc:7.1f} us ({nb/tc/1e3:5.0f} GB/s, {100*nb/tc/1e3/BW:3.0f}%)   triton bm{bm} bn{bn} bk{bk} w{nw} s{st} "
          f"{t:7.1f} us ({nb/t/1e3:5.0f} GB/s, {100*nb/t/1e3/BW:3.0f}%)  x{tc/t:4.2f}  dif {d:.1e}  {'TRITON' if win else 'cutlass'}", flush=True)
    cfg[str(M)] = ({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "num_warps": nw, "num_stages": st} if win else None)
    if M == 4: resumen.append(f"{name}: cutlass {tc:.1f} -> {'triton ' + format(t, '.1f') if win else 'sin cambio'} us")


def _verifica_despacho_fp8():
    """Prueba de que, con el JSON presente, apply_fp8_linear va a triton_scaled_mm y no a cutlass
    (fp8_utils.apply_fp8_linear: use_tuned_triton_channelwise -> get_w8a8_channelwise_fp8_config -> triton_scaled_mm)."""
    import sglang.srt.layers.quantization.fp8_utils as fu
    from sglang.kernels.ops.gemm.fp8_kernel import get_w8a8_channelwise_fp8_configs
    get_w8a8_channelwise_fp8_configs.cache_clear()  # lru_cache: el proceso pudo leer "sin fichero" antes
    calls = []
    orig = fu.triton_scaled_mm
    fu.triton_scaled_mm = lambda *a, **k: (calls.append(k.get("block_size_n")), orig(*a, **k))[1]
    print("\n Verificación de despacho (apply_fp8_linear real, M=4 y M=512):")
    try:
        for name, K, N, _ in FP8_SHAPES:
            w = torch.zeros(N, K, device=DEV, dtype=torch.float8_e4m3fn).t(); s = torch.rand(N, 1, device=DEV) * 0.01
            for M in (4, 512):
                calls.clear()
                fu.apply_fp8_linear(torch.randn(M, K, device=DEV, dtype=torch.bfloat16), w, s, None,
                                    use_per_token_if_dynamic=True, compressed_tensor_quant=True)
                print(f"   {name:42s} M={M:3d} -> {'TRITON bn=' + str(calls[0]) if calls else 'cutlass'}")
            del w
    finally:
        fu.triton_scaled_mm = orig


# ---------------- NVFP4 W4A4 (MLP capas 0-55) ----------------
FP4_SHAPES = [("mlp gate_up", 5120, 34816, 56), ("mlp down", 17408, 5120, 56)]


def bench_fp4(Ms):
    print("\n== NVFP4 W4A4: KDA qwen3x SM120 por la PUERTA PÚBLICA (allowlist kdev) vs flashinfer cutlass ==")
    from flashinfer import mm_fp4
    from sglang.srt.layers.quantization.modelopt_quant import fp4_quantize
    from sglang.kernels.ops.gemm import try_qwen3x_nvfp4_gemm
    for name, K, N, calls in FP4_SHAPES:
        nb = N * K // 2 + N * K // 16
        gs = torch.tensor(1.0, device=DEV)
        def make():
            w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
            wq, wsf = fp4_quantize(w, gs)  # igual que el scheme: [N, K/2] u8 + escala swizzle
            return wq, wsf.view(torch.float8_e4m3fn).reshape(N, -1)
        rot = Rot(make, nb)
        alpha = torch.tensor(1.0, device=DEV)
        print(f" {name} K={K} N={N} ({calls}/paso)")
        ok_all = True
        for M in range(1, 17):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            xq, xsf = fp4_quantize(x, gs)
            w0, s0 = rot.items[0]
            # mismos tensores que compressed_tensors_w4a4_nvfp4.apply_weights: weight_packed.T, weight_scale.T
            o_ref = mm_fp4(xq, w0.T, xsf, s0.T, alpha, torch.bfloat16, backend="cutlass").float()
            o_kda = try_qwen3x_nvfp4_gemm(xq, w0.T, xsf, s0.T, alpha, torch.bfloat16, N)
            if o_kda is None:
                print(f"  M={M:2d} la puerta KDA devuelve None (forma no admitida o layout distinto) -> sigue cutlass"); ok_all = False; continue
            d = (o_kda.float() - o_ref).abs(); rel = d.max().item() / (o_ref.abs().max().item() + 1e-6)
            mean_rel = (d.mean() / (o_ref.abs().mean() + 1e-6)).item()
            ok = rel < 1e-2
            ok_all &= ok
            if M not in Ms and M not in (2, 3, 12):
                print(f"  M={M:2d} numérico {'OK' if ok else 'FALLA'} (rel máx {rel:.1e}, medio {mean_rel:.1e})"); continue
            tc = graph_us(lambda: (lambda w, s: mm_fp4(xq, w.T, xsf, s.T, alpha, torch.bfloat16, backend="cutlass"))(*rot.next()))
            tk = graph_us(lambda: (lambda w, s: try_qwen3x_nvfp4_gemm(xq, w.T, xsf, s.T, alpha, torch.bfloat16, N))(*rot.next()))
            print(f"  M={M:2d} cutlass {tc:6.1f} us ({100*nb/tc/1e3/BW:3.0f}%)  KDA {tk:6.1f} us ({100*nb/tk/1e3/BW:3.0f}%)  x{tc/tk:4.2f}  "
                  f"numérico {'OK' if ok else 'FALLA'} (rel máx {rel:.1e}, medio {mean_rel:.1e})", flush=True)
        print(f"  => {name}: {'APTO M=1..16' if ok_all else 'NO APTO en alguna M: bajar SGLANG_KDEV_KDA_NVFP4_MAX_M o quitar la forma'}")
        del rot; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["ba", "fp8", "fp4"])
    ap.add_argument("--m", default="1,4,8,16")
    ap.add_argument("--rapido", action="store_true", help="fp8: solo las M de --m (sin fase 2)")
    a = ap.parse_args()
    Ms = [int(v) for v in a.m.split(",")]
    import datetime, sys as _sys
    _log = f"./mb_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    _f = open(_log, "w", buffering=1)
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d)
        def flush(self):
            for x in self.s: x.flush()
    _sys.stdout = _Tee(_sys.__stdout__, _f); _sys.stderr = _Tee(_sys.__stderr__, _f)
    print("salida también en", _log)
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU {torch.cuda.get_device_name()}  libre {free:.1f} GB")
    assert free > 20, "producción parece estar viva (poca memoria libre): no corro"
    for sec, fn in (("ba", bench_ba), ("fp8", lambda m: bench_fp8(m, a.rapido)), ("fp4", bench_fp4)):
        if a.solo in (None, sec):
            try:
                fn(Ms)
            except Exception:
                traceback.print_exc()
