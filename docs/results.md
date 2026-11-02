# Full results

Every number here is computed from the stored runs by `scripts/analyze_value_study.py` (→
`results/value_model/analysis.json`), `scripts/controls.py` (→ `results/value_model/controls.json`) and
`scripts/analyze_reproduction.py` (→ `results/reproduction/aime26.json`). The design and its dated deviations are
in [preregistration.md](preregistration.md). All evaluated runs are FULL: complete sampling at the native budget.

## Commitment curve (AIME 2026)

Setup: 24 failed + 24 successful stored rollouts. The failed ones are 20 from the 9 problems with mixed outcomes and
4 from the never-solved problem; the successful ones are spread round-robin over the mixed problems. Each rollout is
cut at a fraction f of its tokens, and 2 fresh continuations are sampled from the cut. That gives 384 continuations,
with harness sampling and a budget of 126,976 tokens minus the prefix. Intervals are a bootstrap over rollouts.

P(correct | prefix), mean and 95 % CI:

| Prefix kept | 0 % (problem base rate) | 5 % | 15 % | 35 % | 60 % | 100 % |
|---|---|---|---|---|---|---|
| failed rollouts | 0.64 (0.50–0.75) | 0.56 (0.38–0.73) | 0.56 (0.42–0.71) | 0.56 (0.40–0.71) | 0.38 (0.23–0.54) | 0 |
| successful rollouts | 0.87 (0.82–0.92) | 0.88 (0.77–0.96) | 0.83 (0.71–0.94) | 0.92 (0.81–1.00) | 1.00 (48 / 48) | 1 |

Paired: each prefix minus its own problem's base rate, in percentage points (this is what `figures/commitment.svg`
shows):

| Prefix kept | 5 % | 15 % | 35 % | 60 % |
|---|---|---|---|---|
| failed rollouts | −7 (−23, +8) | −7 (−20, +5) | −7 (−20, +4) | **−26 (−42, −10)** |
| successful rollouts | +1 (−9, +8) | −4 (−13, +4) | +5 (−5, +13) | **+13 (+9, +18)** |

Spearman correlation between a judge's score at the prefix and the measured P(correct | prefix): value model
0.20 / 0.28 / 0.24 / 0.27 at 5 / 15 / 35 / 60 %, which is weak. The probe and the confidence judges are ≈ 0.

## How early can a judge tell? (within-problem AUROC)

The mean over problems with mixed outcomes (HMMT Feb 2026: 14, AIME 2026: 9) of the AUROC among that problem's
rollouts. The value model's 95 % CIs on HMMT are 0.41–0.71 at 5 %, 0.58–0.82 at 35 %, 0.65–0.87 at 60 % and
0.73–0.94 at 100 %; on AIME they are 0.50–0.75, 0.64–0.89, 0.77–0.93 and 0.85–0.96.

**HMMT Feb 2026**

| Judge | 512 | 1k | 2k | 4k | 8k | 16k tokens | 5 % | 15 % | 35 % | 60 % | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value model | 0.60 | 0.42 | 0.46 | 0.56 | 0.53 | 0.55 | 0.56 | 0.63 | 0.70 | 0.77 |
| value model (running mean) | 0.59 | 0.52 | 0.50 | 0.53 | 0.58 | 0.56 | 0.57 | 0.60 | 0.71 | 0.76 |
| confidence (window) | 0.50 | 0.41 | 0.43 | 0.53 | 0.53 | 0.43 | 0.45 | 0.46 | 0.40 | 0.54 |
| lowest window confidence | 0.49 | 0.46 | 0.43 | 0.47 | 0.51 | 0.52 | 0.54 | 0.55 | 0.58 | 0.60 |
| log-prob (window) | 0.48 | 0.46 | 0.50 | 0.53 | 0.56 | 0.46 | 0.51 | 0.44 | 0.45 | 0.47 |
| neg. entropy (window) | 0.47 | 0.45 | 0.43 | 0.50 | 0.57 | 0.44 | 0.48 | 0.49 | 0.43 | 0.51 |
| mean log-prob so far | 0.48 | 0.44 | 0.43 | 0.48 | 0.44 | 0.47 | 0.47 | 0.34 | 0.34 | 0.39 |
| hidden-state probe | 0.50 | 0.50 | 0.52 | 0.45 | 0.47 | 0.50 | 0.52 | 0.55 | 0.60 | 0.63 |

**AIME 2026**

| Judge | 512 | 1k | 2k | 4k | 8k | 16k tokens | 5 % | 15 % | 35 % | 60 % | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value model | 0.61 | 0.45 | 0.50 | 0.47 | 0.62 | 0.76 | 0.62 | 0.55 | 0.77 | 0.85 |
| value model (running mean) | 0.59 | 0.56 | 0.46 | 0.48 | 0.50 | 0.62 | 0.58 | 0.56 | 0.66 | 0.72 |
| confidence (window) | 0.53 | 0.50 | 0.47 | 0.63 | 0.54 | 0.43 | 0.55 | 0.38 | 0.57 | 0.58 |
| lowest window confidence | 0.50 | 0.52 | 0.55 | 0.48 | 0.51 | 0.60 | 0.54 | 0.54 | 0.62 | 0.68 |
| log-prob (window) | 0.60 | 0.44 | 0.52 | 0.59 | 0.50 | 0.43 | 0.49 | 0.41 | 0.45 | 0.57 |
| neg. entropy (window) | 0.54 | 0.50 | 0.45 | 0.58 | 0.49 | 0.45 | 0.52 | 0.40 | 0.49 | 0.58 |
| mean log-prob so far | 0.60 | 0.52 | 0.52 | 0.49 | 0.54 | 0.52 | 0.52 | 0.37 | 0.40 | 0.49 |
| hidden-state probe | 0.52 | 0.40 | 0.37 | 0.52 | 0.43 | 0.40 | 0.56 | 0.59 | 0.44 | 0.45 |

On HMMT the mean log-prob so far is *below* chance (0.34–0.48): wrong solutions are written with higher token
confidence. This is a post-hoc observation and was not tested.

**Development pool (judge choice).** These are global AUROCs across all dev rollouts longer than the checkpoint, so
they mix problems. The value model scores 0.58 / 0.60 / 0.63 / 0.62 / 0.68 after 512 / 1k / 2k / 4k / 8k tokens,
while its within-problem AUROC on the evaluated sets is ~0.5 at the same points: early values mainly encode problem
difficulty. Pairwise accuracy at 4,096 tokens, on the 23 dev problems with one correct and one wrong sample both
longer than 4k tokens: probe 0.57, value 0.52, others 0.26–0.48. So the pre-registered rule picked the probe, by a
small margin on few pairs. The probe's cross-validated layer is layer 16 or the final layer at every checkpoint.

## Early selection vs plain sampling at equal compute

The policy `select(judge, c, B, m)` starts B samples, reads the judge after c tokens, finishes the best m and takes
a majority vote. It is compared with plain majority voting (maj@k) interpolated at the policy's mean tokens per
problem. The cost counts all generated tokens including stopped prefixes, plus 0.1 token-equivalents per token
scored by the separate value model. Each policy is simulated exactly on the stored rollouts over 200 random sample
orders per problem. Intervals are a bootstrap over problems.

Plain sampling, voted accuracy: HMMT maj@1 / 2 / 4 / 8 / 16 / 24 = 81.0 / 82.1 / 85.1 / 86.8 / 88.6 / 89.5 %;
AIME maj@1 / 2 / 4 / 8 / 16 = 92.3 / 93.2 / 95.6 / 96.5 / 96.7 %.

**Pre-registered setting c = 4,096, B = 8, m = 2**, Δ vs plain sampling in pp (95 % CI):

| Judge | HMMT Feb 2026 | AIME 2026 |
|---|---|---|
| **hidden-state probe (primary, chosen on dev)** | **+1.9 (−1.9, +5.8)** | −0.5 (−2.6, +1.2) |
| value model | +1.1 (−2.0, +4.3) | −2.7 (−5.2, −0.7) |
| value model (running mean) | +0.7 (−1.7, +3.1) | −2.6 (−5.2, −0.7) |
| confidence (window) | +0.1 (−3.3, +3.8) | −0.4 (−2.2, +1.0) |
| lowest window confidence | −3.0 (−7.7, +1.3) | −0.4 (−2.7, +1.4) |
| log-prob (window) | +0.8 (−2.7, +4.8) | −0.4 (−2.9, +1.8) |
| neg. entropy (window) | −0.5 (−3.7, +2.9) | −0.7 (−2.4, +0.6) |
| mean log-prob so far | −1.1 (−5.4, +2.6) | −2.4 (−6.0, +0.1) |
| random judge (control) | −1.5 (−5.2, +1.8) | – |
| perfect judge (oracle, upper bound) | +10.4 (+4.1, +17.8) | +2.1 (+0.4, +4.3) |

On HMMT the primary policy generates ~117k tokens per problem, between plain maj@2 (90k tokens, 82.1 %) and maj@3
(135k, 84.5 %); its voted accuracy is 85.5 %. The pre-registered criterion (CI above 0) is not met.

**Secondary grid** (c ∈ {1k, 2k, 4k, 8k}, B ∈ {4, 8, 16}, m ∈ {1, 2, 4}; 256 judge × setting combinations per set).
On HMMT the real judges range from −10.7 to +5.0 pp. The best are log-prob (window) at c = 2k, B = 16, m = 4
(+5.0), lowest window confidence at c = 1k, B = 16, m = 1 (+4.7) and the probe at c = 4k, B = 16, m = 1 (+4.3).
With this many comparisons on 33 problems, none of them is a finding. The best oracle setting is +12.6 pp
(+5.6, +20.8) at c = 1k, B = 16, m = 1. On AIME the real judges range from −4.6 to +2.3 pp.

**Controls (HMMT).** A random judge at four settings gives −6.1 to +0.6 pp, all CIs include 0. A judge that prefers
the shorter *final* length gives −1.6 to +1.2 pp. That judge is not available at the checkpoint, so this bounds how
much of any gain could come from preferring samples that finish soon. On the 14 problems with mixed outcomes, wrong
solutions are much longer than correct ones (87k vs 57k tokens on average).

**Exploratory (not pre-registered).**
- *A late checkpoint.* The value judge at c = 16,384 is worse than plain sampling on HMMT: B = 8, m = 1 gives
  −5.4 pp (−9.5, −1.9), and B = 4, m = 1 gives −1.8 pp (−4.4, +0.5). Reading the solution for longer costs more than
  the judge recovers.
- *The value model as a verifier of finished solutions* (best of k by the final value vs majority of k):

| k | HMMT Feb 2026 | Δ (95 % CI) | AIME 2026 | Δ (95 % CI) |
|---|---|---|---|---|
| 2 | 84.6 vs 82.1 % | +2.5 (+0.7, +4.4) | 95.0 vs 93.2 % | +1.7 (+0.7, +2.9) |
| 4 | 86.6 vs 85.1 % | +1.5 (−1.8, +5.1) | 96.0 vs 95.6 % | +0.4 (−0.2, +1.4) |
| 8 | 87.6 vs 86.8 % | +0.8 (−4.7, +5.8) | 96.5 vs 96.5 % | 0.0 |
| 16 | 87.6 vs 88.6 % | −1.0 (−7.9, +4.0) | 96.7 vs 96.7 % | 0.0 |

## Compute

All jobs ran on Google Colab, one GPU per job. GPU-hours are the wall time of each job, including installation.

| Job | GPU | GPU-hours |
|---|---|---|
| AIME 2026 reproduction (pilot + avg@32) | A100-SXM4-40GB | 4.7 |
| development pool (3,000 problems; rollouts on 2,000, 35M tokens) | RTX PRO 6000 Blackwell | 2.3 |
| smoke tests (parity gate, plumbing) | L4 | 0.5 |
| judge throughput test | RTX PRO 6000 Blackwell | 0.1 |
| judges, first attempt (stopped: autograd kept the logits) | RTX PRO 6000 Blackwell | 0.2 |
| judges: AIME 2026 (960 rollouts, 29.3M tokens) + dev pool (4,000, first 8,200 tokens) | RTX PRO 6000 Blackwell | 1.0 |
| restarts from prefixes, part 1 (262 of 384 continuations) | RTX PRO 6000 Blackwell | 1.9 |
| restarts from prefixes, part 2 (122 continuations, new seed base) | RTX PRO 6000 Blackwell | 1.1 |
| HMMT Feb 2026 problems 0–16 × 24 + judges | RTX PRO 6000 Blackwell | 2.1 |
| HMMT Feb 2026 problems 17–32 × 24 + judges | RTX PRO 6000 Blackwell | 3.1 |
| **total** | | **17.0** |

Judge throughput on the Blackwell GPU, with the value model and Limite together: 12.5k tokens/s.
