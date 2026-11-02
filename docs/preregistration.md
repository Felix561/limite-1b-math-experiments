# Value-model study: pre-registration

Written on 2026-10-31, after the AIME 2026 reproduction and before any other run of this study: before any HMMT
Feb 2026 sample was generated and before any judge was evaluated. Names of scripts and judges are updated to match
the code; no criterion or number was changed. Later changes are listed, dated, under Deviations.

## Question

A Limite-1B-Violetto solution to a competition problem is about 30k tokens long on AIME 2026 and 45k on HMMT Feb 2026.
**Where in a solution is its outcome decided, and can a cheap judge see it early enough to spend compute better?**

1. **Commitment curve (a controlled intervention).** Continue stored AIME 2026 rollouts from their first
   f ∈ {5, 15, 35, 60} % of tokens (m = 2 fresh continuations each, harness sampling) and measure P(correct | prefix).
   Plan: 32 failed + 32 successful rollouts that ended with an answer, from the 10 AIME 2026 problems with failures
   (28 failures sampled from the 9 mixed problems, 4 from the never-solved problem); 512 continuations, ~22M tokens.
   f = 0 is the problem's base rate from the 32 stored samples; f = 1 is the original outcome.
2. **Judges.** One forward pass per rollout gives, after every prefix length t:
   - `value`: Paradigma's released value model (revision `ce83740e`), V(prompt, response[:t]), trained to predict
     the final correctness return from the preceding tokens; also its running mean (`value_mean`);
   - confidence: Limite's own token statistics over the last 512 tokens (mean log-prob of the sampled tokens,
     entropy, DeepConf top-20 confidence, lowest window confidence so far, mean log-prob so far);
   - `probe`: a ridge probe on Limite's hidden states (layers 16, 32, final) after the prefix, fitted on the dev pool
     only (layer chosen by grouped 5-fold cross-validation on the dev pool).
3. **Early selection**, simulated exactly on stored rollouts (stopping a rollout at c tokens costs c tokens; the judge
   sees exactly the prefix it would see live). `select(judge, c, B, m)`: start B samples, read the judge at c tokens
   (samples that finished earlier are judged on their whole response), continue only the m best, majority vote over
   the m. Cost = Σ min(L, c) + Σ_kept (L − c). Compared with plain sampling (maj@k for every k, the accuracy–token
   frontier) and with the oracle (the true outcome as the judge) as the upper bound. 200 random sample orders per
   problem.

## Data

| Set | Rollouts | Use |
|---|---|---|
| dev pool: 2,000 problems × 2 samples, 16k-token cap, Math-Verify labels | 4,000 | the only data used to choose judges, fit the probe and set any threshold |
| AIME 2026 reproduction (FULL, avg@32, 2026-10-26) | 960 | evaluation (saturated; 10 problems with failures) and the commitment curve |
| HMMT Feb 2026, harness taskset `hmmt26`, avg@32, native budget, seed = problem × 32 + rollout | up to 33 × 32 (new) | headline evaluation (paper: 83.62 %), and a second reproduction |

The dev pool is drawn from DeepScaleR, NuminaMath-1.5 (competition sources) and MATH (levels 3–5). It is
deduplicated and decontaminated against every evaluation set of the harness: no shared word 10-gram, no containment.
Labels: the harness's `exact` verdict (MathArena grader) for AIME 2026 and HMMT Feb 2026, Math-Verify for the dev
pool. Dev rollouts truncated at 16k tokens are excluded from probe training, because their label reflects the cap,
not the reasoning. The ground truth of the evaluated sets is never used to choose anything.

## Pre-registered choices and criteria

- **Primary judge** = the judge with the highest dev pairwise accuracy at c = 4,096 tokens. This uses dev problems
  with one correct and one wrong sample, both longer than c. Ties go to the cheaper judge
  (confidence < probe < value).
- **Primary policy** = `select(primary judge, c = 4096, B = 8, m = 2)`. The secondary grid is
  c ∈ {1k, 2k, 4k, 8k}, B ∈ {4, 8, 16}, m ∈ {1, 2, 4} for all judges, reported in full and never cherry-picked.
- **Primary metric**: on HMMT Feb 2026, the voted accuracy of the primary policy minus the plain-sampling frontier,
  interpolated at the policy's mean tokens per problem, with a bootstrap over problems (10,000 resamples).
  "Early selection pays" if the 95 % CI lies above 0. Secondary: the per-sample accuracy of the kept samples vs
  random samples (selection lift), and the same analysis on AIME 2026.
- **AUROC curve**: within-problem AUROC of each judge vs prefix length (absolute and as a fraction), on AIME 2026
  and HMMT Feb 2026, with bootstrap CIs over problems.
- **Commitment curve**: mean P(correct | prefix at f) for originally failed vs successful rollouts, with bootstrap
  CIs over rollouts. "The outcome is set by f" where the two curves' CIs separate and stay separated. Also the
  Spearman correlation between judge and P̂(correct) at each f.
- **Judge cost**: the value model is a separate 1B forward pass over the prefix, counted as 0.1 token-equivalents
  per scored token (reported separately). Confidence and probe cost ≈ 0, because they come from the policy's own
  forward pass during generation.

## Runs

The jobs are `gpu/value_study.sh` with the modes `parity`, `judges`, `branch` and `hmmt`. They ran on one GPU per
job, using small L4 sessions for smoke tests and an NVIDIA RTX PRO 6000 Blackwell (96 GB) for everything else. The
judges' attention runs through a chunked SDPA drop-in, because FlashAttention is not available on Blackwell and full
L×L masks do not fit. `parity` must pass first (see Deviations).

## Interpretation fixed in advance

- If the value model judges early, it is a free inference-time pruner.
- If the value model judges only late, it is a verifier of finished solutions, not an early judge.
- If the probe beats both, Limite's own states know the outcome before the text shows it.
- The commitment curve shows where in a solution the outcome is set, and so bounds every judge.

Related methods: DeepConf (confidence-filtered early stopping), Math-Shepherd (Monte Carlo prefix values), Thought
Anchors (resampling reasoning steps).

## Deviations (recorded as they happened)

- **2026-10-31, parity gate for Limite (before any main run).** Smoke test on an L4. Value model, chunked vs eager:
  max |Δ| 3.3e-3 / 5.3e-3 / 7.2e-3 and mean 5–6e-4 at 1.5k / 4k / 8k tokens (pass). Limite at 3,000 tokens, mean
  |Δ log-prob|: chunked–sdpa 0.0090, chunked–eager 0.0110, and **HF's own sdpa–eager 0.0105**. Beyond the
  1,024-token window the three are 0.0088 / 0.0100 / 0.0091. The pre-registered "chunked–sdpa < 0.005" was stricter
  than the disagreement between HF's own bf16 kernels, and the error beyond the window is no larger than inside it
  (so there is no window off-by-one). Accepted with the criterion "chunked–eager ≤ 1.5 × (sdpa–eager) + 0.002"
  (0.0110 ≤ 0.0178). The judge features average over 512-token windows, so ~0.01 nats of per-token kernel noise is
  immaterial.
- **2026-10-31, smoke tests.** The first smoke run failed on a code bug: position ids were derived from the mask
  dict, so they are now passed explicitly. The second failed on the too-strict gate above. The third passed every
  stage, but showed a judge throughput of ~1.2k tok/s and out-of-memory errors from ~13k tokens. Fixed by:
  - a flash-style lower-right causal bias for global layers;
  - 1,024-query band blocks for local layers;
  - bf16 logits;
  - score blocks sized by sequence length.

  On the Blackwell GPU, parity passes with the final kernels (chunked = HF sdpa exactly; both 0.0104 from eager) at
  12.5k tok/s.
- **2026-10-31, compute re-plan before any main data.** Measured throughput and session limits forced these changes:
  - HMMT Feb 2026 **avg@24** instead of 32, in two jobs (problems 0–16 and 17–32);
  - commitment curve on **24 failed + 24 successful** rollouts (20 sampled from the mixed problems + 4 from the
    never-solved problem), m = 2, so 384 continuations (~17M tokens);
  - the dev pool judged on its first 8,200 response tokens, since only checkpoints ≤ 8,192 use the dev pool.
- **2026-11-01, branch job split.** The branch job reached its time limit after 262 of 384 continuations. The
  remaining 122 (the plan items without continuations) ran in a second job with a new seed base (60,000,000
  instead of 50,000,000), so that no seed was reused. A single job with the default seed base is statistically
  equivalent.
- **2026-11-01, probe layers.** Layer-32 hidden states overflowed fp16 in 283 of 5,794 judged rollouts. Its probe
  therefore has invalid features and never wins the cross-validated layer choice. Layers 16 and final are clean, and
  the probe uses one of them at every checkpoint.
