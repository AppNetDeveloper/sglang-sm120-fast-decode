"""Test de equivalencia FR-Spec con lm_head FP8 (compressed-tensors, per-channel, peso guardado [K, V]).

  --cpu : lógica de recorte con tensores sintéticos (sin GPU; CUDA_VISIBLE_DEVICES= ).
  --gpu : lm_head REAL del checkpoint con el kernel real (apply_fp8_linear) y medición en CUDA graph
          a M=1,4,8,16: completo [248320] vs recortado [40960]. ~1,8 GB de GPU. SOLO en la ventana.
Uso: PYTHONPATH=/workspace/kdev/sglang_src /workspace/sgl0521/bin/python test_frspec_fp8.py --cpu|--gpu
"""
import argparse, json, os, sys
import torch

MODEL = os.environ.get("MODEL_DIR", "./model")
MAP = os.environ.get("TOKEN_MAP", "../tokens/hot_tokens_qwen38_es_en_code_40000.pt")


def fake_head(w_vk: torch.Tensor, scale_v1: torch.Tensor):
    """Módulo con la misma forma que deja CompressedTensorsW8A8Fp8.process_weights_after_loading."""
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(w_vk.t(), requires_grad=False)  # [K, V], stride (1, K)
    m.weight_scale = torch.nn.Parameter(scale_v1, requires_grad=False)  # [V, 1] fp32
    m.input_scale = None
    v = w_vk.shape[0]
    m.num_embeddings = m.num_embeddings_padded = m.num_embeddings_per_partition = v
    m.org_vocab_size = m.org_vocab_size_padded = v
    m.logical_widths = [v]
    return m


def ref_logits(x, head):
    w = head.weight.float().t() * head.weight_scale.float()  # [V, K] descuantizado
    return x.float() @ w.t()


def test_cpu():
    from sglang.srt.speculative.spec_utils import prune_quantized_lm_head

    torch.manual_seed(0)
    V, K = 4096, 256
    w = (torch.randn(V, K) * 0.05).to(torch.float8_e4m3fn)
    s = torch.rand(V, 1) * 0.01 + 1e-3
    head = fake_head(w, s)
    hot = torch.randperm(V)[:640].sort().values
    p = prune_quantized_lm_head(head, hot)
    assert p is not head and head.weight.shape == (K, V), "el head original no debe cambiar"
    assert p.weight.shape == (K, 640) and p.weight.stride() == (1, K), (p.weight.shape, p.weight.stride())
    assert p.weight_scale.shape == (640, 1)
    assert p.num_embeddings == 640 and p.logical_widths == [640]
    x = torch.randn(4, K)
    full = ref_logits(x, head)[:, hot]
    pr = ref_logits(x, p)
    assert torch.equal(full, pr), (full - pr).abs().max()
    # bf16 sin cuantizar -> None (camino viejo)
    hb = torch.nn.Module(); hb.weight = torch.nn.Parameter(torch.randn(V, K, dtype=torch.bfloat16))
    assert prune_quantized_lm_head(hb, hot) is None
    # escala por bloques (no recortable por filas) -> error claro, no basura silenciosa
    head.weight_scale = torch.nn.Parameter(torch.rand(V // 128, K // 128), requires_grad=False)
    try:
        prune_quantized_lm_head(head, hot); raise SystemExit("FALLO: aceptó escala por bloques")
    except NotImplementedError:
        pass
    print("CPU OK: peso y escala recortados por vocab, layout [K,H] stride (1,K), original intacto")


def load_real_head():
    from safetensors import safe_open
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name in ("lm_head.weight", "lm_head.weight_scale"):
        with safe_open(os.path.join(MODEL, idx[name]), "pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    return out["lm_head.weight"], out["lm_head.weight_scale"].float().reshape(-1, 1)


def graph_time_us(fn, iters=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): fn()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (iters * 10)


def test_gpu():
    from sglang.srt.speculative.spec_utils import prune_quantized_lm_head
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    free = torch.cuda.mem_get_info()[0] / 2**30
    assert free > 3.0, f"solo {free:.1f} GB libres: no corro"
    w, s = load_real_head()
    print("lm_head real:", tuple(w.shape), w.dtype, "escala", tuple(s.shape))
    head = fake_head(w.cuda(), s.cuda()); del w
    hot = torch.tensor(torch.load(MAP), dtype=torch.long, device="cuda")
    p = prune_quantized_lm_head(head, hot)
    V = head.weight.shape[1]; H = p.weight.shape[1]
    run = lambda h, x: apply_fp8_linear(x, h.weight, h.weight_scale, None, use_per_token_if_dynamic=True, compressed_tensor_quant=True)
    torch.manual_seed(0)
    worst = 0.0
    for M in (1, 4, 8, 16):
        x = torch.randn(M, 5120, device="cuda", dtype=torch.bfloat16)
        lf = run(head, x)[:, hot].float(); lp = run(p, x).float()
        d = (lf - lp).abs().max().item(); worst = max(worst, d)
        same_top1 = torch.equal(lf.argmax(-1), lp.argmax(-1))
        top4 = (lf.topk(4).indices == lp.topk(4).indices).float().mean().item()
        tf = graph_time_us(lambda: run(head, x)); tp = graph_time_us(lambda: run(p, x))
        bf = V * 5120 / tf / 1e3; bp = H * 5120 / tp / 1e3
        print(f"M={M:2d}  max|dif|={d:.3e}  top1 igual={same_top1}  top4={top4:.2f}  "
              f"completo {tf:7.1f} us ({bf:5.0f} GB/s)  recortado {tp:6.1f} us ({bp:5.0f} GB/s)  ahorro {tf-tp:6.1f} us/llamada")
    print("GPU OK" if worst < 1e-2 else f"AVISO: diferencia máx {worst} (mirar si cambia el kernel por N)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu", action="store_true"); g.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    test_cpu() if a.cpu else test_gpu()
