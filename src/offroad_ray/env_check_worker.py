"""Runtime environment check. Runs under the ISOLATED Cosmos interpreter on a GPU.

Exercises the code paths that have broken before rather than checking imports:
cuDNN Conv3D, flash-attn kernels on this GPU architecture, TransformerEngine's
own cuDNN loading, natten, decord decoding of a real prepared clip, and parity
between our cached edge control and the pinned Cosmos on-the-fly edge path.

Usage: python env_check_worker.py <clip.mp4> <control_edge.mp4> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


def check(name, fn, results):
    t0 = time.time()
    try:
        out = fn()
        results[name] = {"ok": True, "seconds": round(time.time() - t0, 3), **(out or {})}
    except Exception as e:
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-3000:]}


def main(clip: str, control: str, out_path: str) -> int:
    results: dict = {"python": sys.version, "executable": sys.executable, "base_prefix": sys.base_prefix,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    import torch
    results["torch"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cudnn"] = torch.backends.cudnn.version()

    def gpu():
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_gb": round(p.total_memory / 1e9, 2),
                "uuid": str(getattr(p, "uuid", "")), "arch_list": torch.cuda.get_arch_list()}
    check("gpu", gpu, results)

    def conv3d():
        torch.backends.cudnn.enabled = True
        x = torch.randn(1, 16, 9, 64, 64, device="cuda", dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(16, 32, 3, padding=1).cuda().to(torch.bfloat16)
        y = conv(x)
        torch.cuda.synchronize()
        ref = torch.nn.functional.conv3d(x.float(), conv.weight.float(), conv.bias.float(), padding=1)
        return {"out_shape": list(y.shape), "max_abs_err_vs_fp32": float((y.float() - ref).abs().max())}
    check("cudnn_conv3d", conv3d, results)

    def flash():
        import flash_attn
        from flash_attn import flash_attn_func
        q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
        o = flash_attn_func(q, q, q)
        torch.cuda.synchronize()
        ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)).transpose(1, 2)
        return {"version": flash_attn.__version__, "max_abs_err_vs_sdpa": float((o - ref).abs().max())}
    check("flash_attn", flash, results)

    def te():
        import transformer_engine
        import transformer_engine.pytorch as tep
        lin = tep.Linear(128, 128).cuda()
        y = lin(torch.randn(16, 128, device="cuda"))
        torch.cuda.synchronize()
        return {"version": transformer_engine.__version__, "out_shape": list(y.shape)}
    check("transformer_engine", te, results)

    def natten():
        import natten as nt
        return {"version": nt.__version__}
    check("natten", natten, results)

    def decord_decode():
        import decord
        vr = decord.VideoReader(clip)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return {"version": decord.__version__, "frames": int(frames.shape[0]), "shape": list(frames.shape),
                "fps": float(vr.get_avg_fps())}
    check("decord_real_clip", decord_decode, results)

    def edge_parity():
        """Compare our cached control with the pinned Cosmos path on the same clip."""
        import numpy as np
        from cosmos_transfer2._src.transfer2.inference.utils import read_and_resize_input
        from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputEdge
        import cv2
        frames, fps, ar, hw = read_and_resize_input(clip, num_total_frames=93, interpolation=cv2.INTER_AREA,
                                                    resolution="720")
        aug = AddControlInputEdge(input_keys=["video"], output_keys=["control_input_edge"], use_random=False,
                                  preset_strength="medium")
        upstream = aug({"video": frames})["control_input_edge"].numpy()[0]  # (T,H,W)
        ours, _, _, _ = read_and_resize_input(control, num_total_frames=93, interpolation=cv2.INTER_LINEAR,
                                              resolution="720")
        ours = ours.numpy()[0]
        agree = float((ours == upstream).mean())
        binary = float(np.isin(np.unique(ours), [0, 255]).all())
        return {"model_input_thw": list(frames.shape[1:]), "aspect": ar, "fps": fps,
                "pixel_agreement": agree, "ours_binary": bool(binary),
                "upstream_edge_density": float((upstream > 0).mean()), "ours_edge_density": float((ours > 0).mean())}
    check("edge_control_parity", edge_parity, results)

    results["all_ok"] = all(v.get("ok") for v in results.values() if isinstance(v, dict) and "ok" in v)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in results.items()}))
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
