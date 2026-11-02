"""Value-model study of Limite-1B-Violetto, GPU side (design: docs/preregistration.md).

    python scripts/value_study.py gen    --taskset hmmt26 --start 0 --end 17 --samples 24 --out gen [--deadline T]
    python scripts/value_study.py branch --plan plan.jsonl.gz --out branch [--deadline T]
    python scripts/value_study.py judge  --rollouts rollouts.jsonl.gz --out judge [--deadline T]
    python scripts/value_study.py parity --rollouts rollouts.jsonl.gz --out parity

Runs inside the upstream `limite-evals` environment (see gpu/setup.sh); T is a unix time after which a run stops
cleanly and labels itself PARTIAL.

gen     plain Violetto samples on a harness taskset (harness rendering, chat profile, seed = problem * 32 + rollout,
        native budget), graded with the harness protocols; every finished sample is written at once with its token ids.
branch  continuations of stored rollouts from a prefix (a fraction f of the rollout's tokens): the controlled
        intervention behind the commitment curve. Same sampling and grading; budget = native budget - prefix.
judge   one forward pass per rollout through (a) Paradigma's value model (V before every response token) and (b) Limite
        itself (log-prob of the sampled token, entropy, top-20 confidence, hidden states of layers 16/32/final).
        Writes features on a grid of prefix lengths, so the analysis can judge any prefix offline.
parity  value model and Limite: chunked SDPA drop-in vs the reference eager attention. Gate for `judge`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

REPO_ID = "paradigma-inc/limite-1b-violetto"
REVISION = "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"  # the checkpoint pinned by limite-evals @7ddad34
VALUE_ID = "paradigma-inc/limite-1b-value-model"
VALUE_REVISION = "ce83740e0e853a7d4e5b80fca29a65fd5b8db7fb"
MAX_TOKENS = 131072 - 4096
EOS = 151643
GROUP = 32
AIME_INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
GRID = 256  # judge features every GRID response tokens
WINDOW = 512  # sliding window of the confidence features
ABS_POS = (512, 1024, 2048, 4096, 8192, 16384)
FRACS = (0.05, 0.15, 0.35, 0.6)
LAYERS = (16, 32)  # plus the final normed hidden state
FEATURES = ("v", "vmean", "lp_w", "ent_w", "conf_w", "conf_min", "lp_mean")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def open_any(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def frac_cut(f: float, n: int) -> int:
    return min(n - 1, max(1, round(f * n)))


def answer_key(text: str) -> str | None:
    from limite_experiments.answers import answer_key as key

    return key(text)


def violetto_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_ID, revision=REVISION))


def harness(taskset: str, tokenizer):
    """Problems of a harness taskset with the harness's rendering, and its scorer."""
    from limite_evals import profiles, suites, tasksets
    from limite_evals_core import protocols

    spec = suites.taskset("math-extended", taskset)
    bindings = tasksets.taskset_protocols(spec)
    profile = profiles.resolve("chat")
    kwargs = dict(profile.chat_template_kwargs or {})
    problems = []
    for i, row in enumerate(tasksets.load_raw_rows(spec)):
        prompt, gold = tasksets.render_row(spec, row, profile)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kwargs
        )
        problems.append({"index": i, "gold": gold, "ids": tokenizer(text, add_special_tokens=False)["input_ids"]})

    def score(text: str, gold) -> dict:
        return {k: (None if v.correct is None else bool(v.correct)) for k, v in protocols.score(text, gold, bindings).items()}

    return problems, score, profile


def dev_prompt_ids(tokenizer, problem: str) -> list[int]:
    """The dev-pool rendering of scripts/generate_dev_rollouts.py (harness AIME instruction + chat template)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": AIME_INSTRUCTION + problem}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# --- generation (gen, branch) ------------------------------------------------------------------------------------------


def make_llm(src: Path, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(model=str(src), dtype="bfloat16", max_model_len=131072, seed=0,
               gpu_memory_utilization=gpu_memory_utilization)


def stream_generate(llm, prompts, params, on_finish, deadline: float) -> bool:
    """Enqueue everything, step the engine, hand each finished request to on_finish(i, output). Returns False if the
    deadline aborted unfinished requests (the caller labels the run PARTIAL)."""
    try:
        ids = [str(x) for x in llm.enqueue(prompts, params, use_tqdm=False)]
    except AttributeError:  # no streaming API: one blocking call
        for i, out in enumerate(llm.generate(prompts, params, use_tqdm=False)):
            on_finish(i, out)
        return True
    internal = {rid.rsplit("-", 1)[0]: rid for rid in ids}
    slot = {ext: i for i, ext in enumerate(internal)}
    open_ids = set(slot)
    engine = llm.llm_engine
    while open_ids and engine.has_unfinished_requests():
        if time.time() > deadline:
            engine.abort_request([internal[x] for x in sorted(open_ids)], internal=True)
            return False
        for out in engine.step():
            if not out.finished:
                continue
            ext = out.request_id if out.request_id in slot else out.request_id.rsplit("-", 1)[0]
            if ext in open_ids:
                open_ids.discard(ext)
                on_finish(slot[ext], out)
    if open_ids:
        raise RuntimeError(f"{len(open_ids)} requests never finished")
    return True


def sampling(profile, seed: int, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=profile.temperature, top_p=profile.top_p, max_tokens=max_tokens, seed=seed,
                          stop_token_ids=[EOS], skip_special_tokens=True)


def cmd_gen(args) -> int:
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt

    src = violetto_dir()
    tok = AutoTokenizer.from_pretrained(src)
    problems, score, profile = harness(args.taskset, tok)
    problems = problems[args.start : args.end]
    budget = min(args.max_tokens, MAX_TOKENS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "problems.json").write_text(json.dumps(
        [{"index": p["index"], "gold": str(p["gold"]), "prompt_ids": p["ids"]} for p in problems]))
    llm = make_llm(src, args.gpu_memory_utilization)
    items = [(p, r) for r in range(args.samples) for p in problems]
    prompts = [TokensPrompt(prompt_token_ids=p["ids"]) for p, _ in items]
    params = [sampling(profile, p["index"] * GROUP + r, budget) for p, r in items]
    stream = (args.out / "samples.jsonl").open("w")
    started, n = time.time(), [0, 0]

    def done(i, out):
        p, r = items[i]
        o = out.outputs[0]
        rec = {"taskset": args.taskset, "problem": p["index"], "rollout": r, "tokens": len(o.token_ids),
               "finish_reason": o.finish_reason, "answer": answer_key(o.text), **score(o.text, p["gold"]),
               "token_ids": list(o.token_ids)}
        stream.write(json.dumps(rec) + "\n")
        stream.flush()
        n[0] += 1
        n[1] += rec["tokens"]
        if n[0] % 32 == 0:
            log(f"gen {n[0]}/{len(items)} samples, {n[1] / max(1, time.time() - started):.0f} tok/s")

    complete = stream_generate(llm, prompts, params, done, args.deadline or math.inf)
    stream.close()
    label = "PARTIAL" if not complete else ("PILOT" if budget < MAX_TOKENS else "FULL")
    (args.out / "summary.json").write_text(json.dumps({
        "taskset": args.taskset, "start": args.start, "end": args.end, "samples_requested": args.samples,
        "samples": n[0], "generated_tokens": n[1], "minutes": round((time.time() - started) / 60, 2),
        "max_tokens": budget, "label": label, "revision": REVISION}, indent=1))
    log(f"gen finished: {n[0]} samples, label {label}")
    return 0


def cmd_branch(args) -> int:
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt

    src = violetto_dir()
    tok = AutoTokenizer.from_pretrained(src)
    plan = [json.loads(line) for line in open_any(args.plan)]
    sets = {}
    for item in plan:
        if item["taskset"] not in sets:
            sets[item["taskset"]] = harness(item["taskset"], tok)
    args.out.mkdir(parents=True, exist_ok=True)
    llm = make_llm(src, args.gpu_memory_utilization)
    reqs = []  # (item, j, prefix ids, prompt ids)
    for item in plan:
        problems, _, _ = sets[item["taskset"]]
        ids = tok(item["completion"], add_special_tokens=False)["input_ids"]
        cut = frac_cut(item["f"], len(ids))
        for j in range(item["m"]):
            reqs.append((item, j, ids[:cut], problems[item["problem"]]["ids"]))
    profile = sets[plan[0]["taskset"]][2]
    prompts = [TokensPrompt(prompt_token_ids=pids + pre) for _, _, pre, pids in reqs]
    params = [sampling(profile, args.seed_base + 64 * k + j, min(args.max_new, MAX_TOKENS - len(pre)))
              for k, (_, j, pre, _) in enumerate(reqs)]
    stream = (args.out / "continuations.jsonl").open("w")
    started, n = time.time(), [0, 0]

    def done(i, out):
        item, j, pre, _ = reqs[i]
        o = out.outputs[0]
        problems, score, _ = sets[item["taskset"]]
        text = tok.decode(pre + list(o.token_ids), skip_special_tokens=True)
        rec = {"key": item["key"], "f": item["f"], "j": j, "prefix_len": len(pre), "tokens": len(o.token_ids),
               "finish_reason": o.finish_reason, "answer": answer_key(text),
               **score(text, problems[item["problem"]]["gold"])}
        stream.write(json.dumps(rec) + "\n")
        stream.flush()
        n[0] += 1
        n[1] += rec["tokens"]
        if n[0] % 32 == 0:
            log(f"branch {n[0]}/{len(reqs)}, {n[1] / max(1, time.time() - started):.0f} tok/s")

    complete = stream_generate(llm, prompts, params, done, args.deadline or math.inf)
    stream.close()
    (args.out / "summary.json").write_text(json.dumps({
        "requests": len(reqs), "finished": n[0], "generated_tokens": n[1], "complete": complete,
        "label": "FULL" if complete and args.max_new >= MAX_TOKENS else ("PARTIAL" if not complete else "PILOT"),
        "minutes": round((time.time() - started) / 60, 2), "revision": REVISION}, indent=1))
    log(f"branch finished: {n[0]}/{len(reqs)} continuations, complete={complete}")
    return 0


# --- judges ------------------------------------------------------------------------------------------------------------


def chunked_attention(qt, kt, vt, left: int, scale: float):
    """Causal attention for one sequence without an L x L mask: q [H, T, D], k/v [Hkv, T, D] -> [H, T, D]. `left` >= 0
    restricts keys to [q - left, q] (flash-attn's window convention); -1 = no window.
    Global layers: 4096-query chunks against all earlier keys with a lower-right causal bias (flash kernel, no mask).
    Local layers: 1024-query chunks against [s - left, e) with a small band mask."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right

    H, T, _ = qt.shape
    rep = H // kt.shape[0]
    kt, vt = kt.repeat_interleave(rep, 0), vt.repeat_interleave(rep, 0)
    out = torch.empty_like(qt)
    # global layers may fall back to SDPA's math kernel (Blackwell): keep its fp32 score block near 1.5 GB
    chunk = max(256, min(4096, int(1.5e9 / (H * T * 4)))) if left < 0 else 1024
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        k0 = 0 if left < 0 else max(0, s - left)
        if left < 0:
            mask = causal_lower_right(e - s, e - k0)
        else:
            qi = torch.arange(s, e, device=qt.device)[:, None]
            ki = torch.arange(k0, e, device=qt.device)[None, :]
            mask = ((ki <= qi) & (ki >= qi - left))[None, None]
        out[:, s:e] = F.scaled_dot_product_attention(qt[None, :, s:e], kt[None, :, k0:e], vt[None, :, k0:e],
                                                     attn_mask=mask, scale=scale)[0]
    return out


def sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal=True, softmax_scale=None, window_size=(-1, -1), **_):
    """Drop-in for flash_attn_varlen_func (one unpadded sequence, q/k/v [T, H, D]), used for the value model where
    FlashAttention is not installed (Blackwell G4)."""
    if cu_q.numel() != 2 or not causal:
        raise ValueError("sdpa_varlen supports one causal sequence")
    out = chunked_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), window_size[0], softmax_scale)
    return out.transpose(0, 1).contiguous()


def limite_chunked(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **_):
    """HF attention interface for Limite ([B, H, T, D] in, [B, T, H, D] out). Limite's `sliding_window` is the span
    including the query token (k >= q - window), so the flash-style left extent is span - 1."""
    left = -1 if sliding_window is None else int(sliding_window) - 1
    out = chunked_attention(query[0], key[0], value[0], left, scaling)
    return out.transpose(0, 1)[None].contiguous(), None


def load_value_model(device: str = "cuda"):
    from huggingface_hub import snapshot_download

    vdir = Path(snapshot_download(VALUE_ID, revision=VALUE_REVISION))
    sys.path.insert(0, str(vdir))
    import modeling_backbone
    import value_model as vm

    model = vm.load_value_model(vdir, device=device, attention_backend="eager")
    modeling_backbone.flash_attn_varlen_func = sdpa_varlen
    return model


def value_post(model, ids: list[int], impl: str = "flash_attention_2"):
    """Raw post-token values for one sequence; impl 'eager' = the reference attention (no length guard here)."""
    import torch
    import torch.nn.functional as F

    model.config._attn_implementation = impl
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    pos = torch.arange(len(ids), device="cuda")[None]
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        h = model.backbone(input_ids=x, position_ids=pos).last_hidden_state
        v = F.linear(h.float(), model.value_head.weight, model.value_head.bias).squeeze(-1)[0]
    return v


def load_limite(src: Path, impl: str = "chunked"):
    """impl 'eager' = the reference attention; 'sdpa' = HF sdpa (L x L masks); 'chunked' = limite_chunked."""
    import torch
    from transformers import AttentionInterface

    from limite_experiments.third_party.limite.modeling_limite import LimiteForCausalLM

    AttentionInterface.register("chunked_sdpa", limite_chunked)
    model = LimiteForCausalLM.from_pretrained(src, dtype=torch.bfloat16,
                                              attn_implementation="eager" if impl == "eager" else "sdpa")
    # 'sdpa' keeps HF's own masked sdpa path
    if impl == "chunked":
        model.config._attn_implementation = "chunked_sdpa"
    return model.cuda().eval()


def limite_pass(model, ids: list[int], p: int, keep: list[int]):
    """Per response token t: log-prob of the sampled token, entropy and top-20 confidence (DeepConf), all computed at the
    state after the prefix of length t; hidden states (layers 16, 32, final) after the prefixes of length `keep`."""
    import torch

    with torch.inference_mode():  # the logits loop too: lm_head.weight requires grad, so autograd would keep every chunk
        return _limite_pass(model, ids, p, keep)


def _limite_pass(model, ids: list[int], p: int, keep: list[int]):
    import torch

    grabbed = {}

    def hook(name):
        def fn(_m, _i, out):
            grabbed[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    handles = [model.model.layers[i].register_forward_hook(hook(i)) for i in LAYERS]
    handles.append(model.model.norm.register_forward_hook(hook("final")))
    try:
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        no_masks = {"full_attention": None, "sliding_attention": None}  # limite_chunked builds its own per chunk
        with torch.no_grad():
            if model.config._attn_implementation == "chunked_sdpa":
                pos = torch.arange(len(ids), device="cuda")[None]  # explicit: never derived from the mask dict
                model.model(input_ids=x, attention_mask=no_masks, position_ids=pos, use_cache=False)
            else:
                model.model(input_ids=x, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    hfin = grabbed["final"][0]
    n = len(ids) - p
    W = model.lm_head.weight
    target = x[0, p:]
    lp, ent, conf = (torch.empty(n, device="cuda") for _ in range(3))
    for s in range(0, n, 1024):
        e = min(n, s + 1024)
        logits = torch.nn.functional.linear(hfin[p - 1 + s : p - 1 + e].to(W.dtype), W).float()  # bf16 matmul
        logp = torch.log_softmax(logits, -1)
        lp[s:e] = logp.gather(1, target[s:e, None])[:, 0]
        ent[s:e] = -(logp.exp() * logp).sum(-1)
        conf[s:e] = -logp.topk(20, -1).values.mean(-1)
    idx = torch.tensor([p + t - 1 for t in keep], device="cuda")
    hid = torch.stack([grabbed[i][0][idx] for i in LAYERS] + [hfin[idx]], 1)  # [len(keep), 3, hidden]
    return lp, ent, conf, hid


def grid_features(values, lp, ent, conf, n: int):
    """Judge features after prefixes of length t on the grid (and at t = n, the complete response)."""
    import torch

    grid = list(range(GRID, n, GRID)) + [n]
    c_lp, c_v = torch.cumsum(lp, 0), torch.cumsum(values[:n], 0)
    c_ent, c_conf = torch.cumsum(ent, 0), torch.cumsum(conf, 0)

    def wmean(c, t):
        a = max(0, t - WINDOW)
        return float((c[t - 1] - (c[a - 1] if a > 0 else 0)) / (t - a))

    rows, worst = [], math.inf
    for t in grid:
        conf_w = wmean(c_conf, t)
        worst = min(worst, conf_w)
        v_t = float(values[t]) if t < len(values) else float(values[-1])
        rows.append([v_t, float(c_v[t - 1] / t), wmean(c_lp, t), wmean(c_ent, t), conf_w, worst, float(c_lp[t - 1] / t)])
    return grid, rows


def cmd_judge(args) -> int:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    src = violetto_dir()
    tok = AutoTokenizer.from_pretrained(src)
    items = [json.loads(line) for line in open_any(args.rollouts)]
    for it in items:  # gen output has no key: taskset:problem:rollout
        it.setdefault("key", f"{it.get('taskset')}:{it.get('problem')}:{it.get('rollout')}")
    sets = {}
    for it in items:
        if it.get("taskset") and it["taskset"] not in sets:
            sets[it["taskset"]] = harness(it["taskset"], tok)
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in args.out.glob("judge-*.jsonl"):
        done |= {json.loads(line)["key"] for line in f.open()}
    vmodel = load_value_model()
    lmodel = load_limite(src)
    shard = len(list(args.out.glob("judge-*.jsonl")))
    meta = (args.out / f"judge-{shard:03d}.jsonl").open("w")
    hids, started, toks, count = {}, time.time(), 0, 0
    for it in items:
        if it["key"] in done:
            continue
        if args.deadline and time.time() > args.deadline:
            log("judge: deadline reached")
            break
        if it.get("taskset"):
            pids = sets[it["taskset"]][0][it["problem"]]["ids"]
        else:
            pids = dev_prompt_ids(tok, it["problem_text"])
        rids = it.get("token_ids") or tok(it["completion"], add_special_tokens=False)["input_ids"]
        rids = rids[: args.max_response] if args.max_response else rids
        n = len(rids)
        if n < 2:
            continue
        ids = list(pids) + list(rids)
        p = len(pids)
        keep = sorted({t for t in ABS_POS if t < n} | {frac_cut(f, n) for f in FRACS})
        try:
            post = value_post(vmodel, ids)
            values = post[p - 1 :]  # values[t] = V(prompt, response[:t]); values[n] = after the whole response
            lp, ent, conf, hid = limite_pass(lmodel, ids, p, keep)
            grid, rows = grid_features(values, lp, ent, conf, n)
            lp_all, v_final = float(lp.mean()), float(values[n])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"judge: OOM on {it['key']} ({len(ids)} tokens), skipped")
            continue
        del post, values, lp, ent, conf
        torch.cuda.empty_cache()
        meta.write(json.dumps({"key": it["key"], "n": n, "grid": grid, "features": rows, "keep": keep,
                               "lp_mean_all": lp_all, "v_final": v_final}) + "\n")
        meta.flush()
        hids[it["key"]] = hid.half().cpu().numpy()
        toks += len(ids)
        count += 1
        if count % 50 == 0:
            log(f"judge {count} rollouts, {toks / (time.time() - started):.0f} tok/s")
            np.savez_compressed(args.out / f"hidden-{shard:03d}.npz", **{k: v for k, v in hids.items()})
    meta.close()
    np.savez_compressed(args.out / f"hidden-{shard:03d}.npz", **{k: v for k, v in hids.items()})
    (args.out / f"summary-{shard:03d}.json").write_text(json.dumps({
        "rollouts": count, "tokens": toks, "minutes": round((time.time() - started) / 60, 2),
        "tok_per_s": round(toks / max(1e-9, time.time() - started)), "value_revision": VALUE_REVISION,
        "revision": REVISION}, indent=1))
    log(f"judge finished: {count} rollouts, {toks} tokens")
    return 0


def cmd_parity(args) -> int:
    import torch
    from transformers import AutoTokenizer

    src = violetto_dir()
    tok = AutoTokenizer.from_pretrained(src)
    items = [json.loads(line) for line in open_any(args.rollouts)][:3]
    problems, _, _ = harness(items[0]["taskset"], tok)
    vmodel = load_value_model()
    report = {"value": [], "limite": []}
    for it, limit in zip(items, (1500, 4000, 8000), strict=False):
        pids = problems[it["problem"]]["ids"]
        rids = tok(it["completion"], add_special_tokens=False)["input_ids"][: limit - len(pids)]
        ids = pids + rids
        ref = value_post(vmodel, ids, "eager")
        new = value_post(vmodel, ids, "flash_attention_2")
        d = (ref - new).abs()
        report["value"].append({"tokens": len(ids), "max_abs": float(d.max()), "mean_abs": float(d.mean())})
        log(f"value parity {len(ids)} tokens: max {float(d.max()):.2e} mean {float(d.mean()):.2e}")
        torch.cuda.empty_cache()
    del vmodel
    torch.cuda.empty_cache()
    # Limite: three implementations at 3,000 tokens (most positions beyond the 1,024-token window). bf16 kernels differ
    # from the eager reference anyway, so the window logic is tested against HF's own sdpa (materialised masks).
    it = items[1]
    pids = problems[it["problem"]]["ids"]
    ids = pids + tok(it["completion"], add_special_tokens=False)["input_ids"][: 3000 - len(pids)]
    out = {}
    for impl in ("eager", "sdpa", "chunked"):
        m = load_limite(src, impl)
        lp, _, _, _ = limite_pass(m, ids, len(pids), [len(ids) - len(pids) - 1])
        out[impl] = lp
        del m
        torch.cuda.empty_cache()
    for a, b in (("chunked", "sdpa"), ("chunked", "eager"), ("sdpa", "eager")):
        d = (out[a] - out[b]).abs()
        tail = d[1100:] if len(d) > 1100 else d  # positions where the window matters
        report["limite"].append({"pair": f"{a}-{b}", "tokens": len(ids), "max_abs_logprob": float(d.max()),
                                 "mean_abs_logprob": float(d.mean()), "mean_abs_beyond_window": float(tail.mean())})
    lim = {r["pair"]: r for r in report["limite"]}
    ok = all(r["max_abs"] < 0.02 and r["mean_abs"] < 0.003 for r in report["value"]) and         lim["chunked-sdpa"]["mean_abs_logprob"] < 0.02 and         lim["chunked-eager"]["mean_abs_logprob"] < 1.5 * lim["sdpa-eager"]["mean_abs_logprob"] + 0.002
    report["pass"] = ok
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(report, indent=1))
    log(f"parity pass={ok}: {json.dumps(report)}")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--taskset", default="hmmt26")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--end", type=int, default=10**6)
    g.add_argument("--samples", type=int, default=32)
    g.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    b = sub.add_parser("branch")
    b.add_argument("--plan", type=Path, required=True)
    b.add_argument("--max-new", type=int, default=MAX_TOKENS)
    b.add_argument("--seed-base", type=int, default=50_000_000, help="a follow-up run uses a new base (no duplicates)")
    j = sub.add_parser("judge")
    j.add_argument("--rollouts", type=Path, required=True)
    j.add_argument("--max-response", type=int, default=0)
    p = sub.add_parser("parity")
    p.add_argument("--rollouts", type=Path, required=True)
    for s in (g, b, j, p):
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--deadline", type=float, default=0.0)
        s.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()
    return {"gen": cmd_gen, "branch": cmd_branch, "judge": cmd_judge, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
