"""VLM judge worker: runs under the Cosmos interpreter on one GPU (stdlib imports at top).

    python -I vlm_judge_worker.py <model_dir> <items.json> <out.jsonl>

items.json: [{"judge_id": str, "path": mp4}, ...]. Appends one JSON line per item to out.jsonl and
skips ids already present (resumable). The judge is BLIND: it is never told which condition was
requested, nor whether a clip is real or synthetic.
"""
import json
import os
import re
import socket
import sys
import time

SYSTEM = ("You are a careful video quality analyst. Answer the question in the following format: "
          "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.")

QUESTION = """This clip is from a forward-facing camera on an off-road vehicle. Examine it closely.

Score it and answer with ONE JSON object inside <answer>, with exactly these keys:
- "condition": the dominant weather or lighting you observe, one of "clear_day", "overcast", "fog", "rain", "night", "snow", "other"
- "condition_intensity": integer 1-5, how strongly that condition is present (1 = barely, 5 = extreme)
- "photorealism": integer 1-5, how much it looks like real camera footage rather than computer-generated imagery (5 = indistinguishable from real)
- "temporal_stability": integer 1-5, how stable textures and objects are over time (1 = heavy flicker, morphing or popping, 5 = fully stable)
- "physical_plausibility": integer 1-5, whether lighting, shadows, water, snow and terrain behave physically correctly
- "artifacts": list of short strings naming visible defects (empty list if none)
- "summary": one sentence

Keep the reasoning under 120 words."""

SCORE_KEYS = ("condition_intensity", "photorealism", "temporal_stability", "physical_plausibility")
CONDITIONS = {"clear_day", "overcast", "fog", "rain", "night", "snow", "other"}


def load_frames(path, fps=4.0, max_side=560):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = src_fps / fps
    want = {int(round(i * step)) for i in range(int(n / step))}
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            h, w = f.shape[:2]
            s = max_side / max(h, w)
            f = cv2.resize(f, (int(w * s) // 28 * 28, int(h * s) // 28 * 28), interpolation=cv2.INTER_AREA)
            frames.append(f[:, :, ::-1])
        i += 1
    if len(frames) % 2:  # Qwen2.5-VL groups frames in temporal pairs
        frames = frames[:-1]
    return np.stack(frames)


def _fields(body):
    """Lenient JSON extraction: tolerates ```json fences and a missing closing brace."""
    body = body.replace("```json", "").replace("```", "")
    i = body.find("{")
    if i < 0:
        return None
    frag = body[i:]
    j = frag.rfind("}")
    for cand in ([frag[: j + 1]] if j >= 0 else []) + [frag.rstrip().rstrip(",") + "\n}", frag.rstrip().rstrip(",") + '"\n}']:
        try:
            return json.loads(cand)
        except Exception:
            pass
    d = {}  # last resort: key-by-key regex
    for k in ("condition", "summary"):
        m = re.search(r'"%s"\s*:\s*"([^"]*)"' % k, frag)
        if m:
            d[k] = m.group(1)
    for k in ("condition_intensity", "photorealism", "temporal_stability", "physical_plausibility"):
        m = re.search(r'"%s"\s*:\s*(\d)' % k, frag)
        if m:
            d[k] = int(m.group(1))
    m = re.search(r'"artifacts"\s*:\s*\[(.*?)\]', frag, re.S)
    if m:
        d["artifacts"] = re.findall(r'"([^"]*)"', m.group(1))
    return d if "condition" in d and "photorealism" in d else None


def parse(text):
    m = re.search(r"<answer>(.*?)(</answer>|$)", text, re.S)
    if not m:
        return None, "no answer (reasoning truncated)" if "</think>" not in text else "no answer tag"
    d = _fields(m.group(1))
    if d is None:
        return None, "no json"
    out = {"condition": str(d.get("condition", "")).strip().lower().replace(" ", "_")}
    if out["condition"] not in CONDITIONS:
        out["condition"] = "other"
    for k in SCORE_KEYS:
        try:
            v = int(d.get(k))
            out[k] = v if 1 <= v <= 5 else None
        except Exception:
            out[k] = None
    a = d.get("artifacts") or []
    out["artifacts"] = [str(x)[:80] for x in a][:8] if isinstance(a, list) else [str(a)[:80]]
    out["summary"] = str(d.get("summary", ""))[:300]
    return out, None


def main(model_dir, items_path, out_path):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    items = json.load(open(items_path))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["judge_id"])
            except Exception:
                pass
    todo = [it for it in items if it["judge_id"] not in done]
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    proc = AutoProcessor.from_pretrained(model_dir)
    load_s = time.time() - t0
    host = socket.gethostname()
    with open(out_path, "a") as f:
        for it in todo:
            t1 = time.time()
            rec = {"judge_id": it["judge_id"], "path": it["path"], "host": host, "load_s": round(load_s, 1)}
            try:
                frames = load_frames(it["path"])
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], videos=[frames], fps=[4.0], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                parsed, err = parse(raw)
                rec.update(ok=parsed is not None, parse_error=err, raw=raw[-3000:], frames=int(len(frames)),
                           **(parsed or {}))
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:500]}")
            rec["seconds"] = round(time.time() - t1, 2)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print(json.dumps({"host": host, "items": len(items), "ran": len(todo), "load_s": round(load_s, 1),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
