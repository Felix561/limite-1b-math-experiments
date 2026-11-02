"""Violetto rollouts on the development pool (vLLM, harness rendering and sampling), graded with Math-Verify.

    python scripts/generate_dev_rollouts.py --pool pool.jsonl --out rollouts.jsonl --metrics metrics/devgen.jsonl \
        [--samples 2 --max-tokens 16384 --chunk 400 --deadline UNIX_TIME]

Rendering = the harness's AIME prompt (instruction prefix + checkpoint chat template); sampling = the harness's chat
profile (temperature 0.6, top_p 0.95), per-problem seed. Results are appended per chunk (resumable: ids already in
--out are skipped) and the run stops cleanly --stop-margin-min before --deadline. Every chunk logs throughput,
accuracy, truncation and token statistics to --metrics. The study's dev pool is the first 2,000 problems of the pool
(the run stopped at its time limit after 5 chunks).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO_ID = "paradigma-inc/limite-1b-violetto"
REVISION = "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
AIME_INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


def grade(completion: str, answer: str) -> bool:
    from math_verify import parse, verify

    try:
        gold = parse(f"${answer}$")
        if not gold:
            gold = parse(answer)
        pred = parse(completion)
        return bool(gold and pred and verify(gold, pred))
    except Exception:  # noqa: BLE001 - an unparsable answer is simply not verified
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time to stop before (0 = no limit)")
    ap.add_argument("--max-problems", type=int, default=0, help="only the first N pool problems (0 = all)")
    ap.add_argument("--stop-margin-min", type=float, default=25)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    deadline = args.deadline or None
    src = snapshot_download(REPO_ID, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(src)
    pool = [json.loads(line) for line in args.pool.open()]
    if args.max_problems:
        pool = pool[: args.max_problems]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    todo = [p for p in pool if p["id"] not in done]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=src, dtype="bfloat16", max_model_len=args.max_tokens + 4096,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    started = time.time()
    with args.out.open("a") as out, args.metrics.open("a") as metrics:
        for c in range(0, len(todo), args.chunk):
            if deadline and time.time() > deadline - args.stop_margin_min * 60:
                metrics.write(json.dumps({"event": "stopped_before_deadline", "t": time.time()}) + "\n")
                break
            chunk = todo[c : c + args.chunk]
            prompts, params = [], []
            for p in chunk:
                text = tokenizer.apply_chat_template([{"role": "user", "content": AIME_INSTRUCTION + p["problem"]}],
                                                     tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                for s in range(args.samples):  # one request per sample, seed = problem * samples + sample (as limite-eval)
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    params.append(SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=args.max_tokens,
                                                 seed=int(p["id"][1:]) * args.samples + s))
            t0 = time.time()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            dt = time.time() - t0
            tokens, correct, truncated = [], 0, 0
            for i, p in enumerate(chunk):
                group = outputs[i * args.samples : (i + 1) * args.samples]
                o = group[0]
                samples = []
                for s in (g.outputs[0] for g in group):
                    ok = grade(s.text, p["answer"])
                    correct += ok
                    truncated += s.finish_reason == "length"
                    tokens.append(len(s.token_ids))
                    samples.append({"text": s.text, "tokens": len(s.token_ids), "finish_reason": s.finish_reason,
                                    "correct": ok})
                out.write(json.dumps({**p, "prompt_tokens": len(o.prompt_token_ids), "samples": samples}) + "\n")
            out.flush()
            n = len(tokens)
            metrics.write(json.dumps({
                "event": "chunk", "t": time.time(), "elapsed_min": round((time.time() - started) / 60, 2),
                "problems": len(chunk), "rollouts": n, "gen_tokens": sum(tokens), "tok_per_s": round(sum(tokens) / dt, 1),
                "accuracy": round(correct / n, 4), "truncated_frac": round(truncated / n, 4),
                "tokens_mean": round(statistics.mean(tokens)), "tokens_median": round(statistics.median(tokens)),
                "done_problems": len(done) + c + len(chunk),
            }) + "\n")
            metrics.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
