# ADR-001 — Structural elimination of roster fingerprinting (the OXF stack)

**Date:** 2026-07-23 · **Status:** ADOPTED (plan v2.3) · **Method:** 4 independent
design agents (causal-inference, information-theory, representation-learning,
market-pragmatist) → 7 unique proposals → 14 adversarial attacks (2 lenses each:
leakage-hunter, product-breaker) → synthesis.

## The problem
The tower sees ≤5 teammate IDs; the WITH-set nearly names the focal player. A tower
learning "this WITH-set ⇒ elite production" memorizes the focal without his ID; at
do(p leaves), old ice keeps his credit → both sides of star trades mispriced. Prior
defenses (dropout, P3 probe, K-1) were mitigations. Goal: make it impossible.

## Proposals and verdicts

| Proposal | Verdict | Decisive attack |
|---|---|---|
| Cross-fitted towers (K player folds) | **ADOPT, repaired** | transferable generic completion circuit learned from other folds executes for p at scoring; shared state (tables/norm/early-stop) reintroduces leak; probe-conditioned re-rolls = tuning on held-out labels |
| Focal-residualized outcomes (orthogonalization) | **ADOPT, hardened** | shrunk/lagged scalars leave identity-correlated residual that re-funds the circuit → must be UNSHRUNK + SPLIT-SAMPLE |
| Full cross-fit w/ single OOF-fused trunk | **ADOPT tower part; trunk repaired** | fingerprint relocates to the trunk: talent⊕bio quasi-ID next to Y on all 6.49M rows; star residual (shrinkage floor) fitted onto p-exclusive h_env clusters → K trunks, routed by fold(projected player) |
| TSMM (member talent scalars in shells, mask-ID-keep-shell) | **ADOPT scalars as feature eng; not structural alone** | provided scalar is not a sufficient statistic (shrinkage + chemistry inexpressible) ⇒ MI(identity; residual)>0 at optimum; echo injection if scalars are only season-lagged → exact pairwise LOO |
| j-anonymous VQ archetype WITH-stream | **REJECT as primary; keep as ablation arm** | census sigma-algebra ≠ model sigma-algebra (free vs_ids, raw seconds via log-pooling, continuous context) ⇒ "constant on cells" vacuous; closing the gap costs opponent resolution + exposure precision |
| Replacement-marginalized full-roster value tower | **RESEARCH ARM (v1.1+)** | objective-level fix is real (no hidden man in training) but trunk-signature relay survives and end-to-end task shaping is sacrificed; big rebuild |
| Separation-weighted training | **DEMOTE to precision add-on** | weight keyed to focal identity = identity-dependent training measure (side-channel); upweights exactly the rows where the star's unexplained surplus is largest — amplifier unless labels are orthogonalized first; window-level focal-free weights + OXF1 make it safe, later |

## The convergent insight
Routing-only fixes fail because the completion circuit is PLAYER-GENERIC and
transfers across folds. Reward-only fixes fail because any provided quality scalar
under-explains identity somewhere (shrinkage, form, chemistry) and the residual is
identity-shaped. **Elimination = conjunction:** remove the reward (orthogonalized
labels), remove the data (player-fold exclusion in every learned component that
touches Y), remove the side doors (shared state, synthetic dropout, prior echo,
probe-tuning). Each leg closes exactly the hole the others leave.

## Decision — the OXF stack (spec in ARCHITECTURE_PLAN v2.3 §2)
OXF1 orthogonalized labels (unshrunk, split-sample, OOF; Poisson via offset) ·
OXF2 K=5 shared-nothing player-fold towers · OXF3 fold-routed K trunks
(tower by fold(window's focal-of-record), trunk by fold(projected player)) ·
OXF4 dropout deleted, talent-gated UNK substitution · OXF5 pairwise-LOO member
talent scalars in shells · OXF6 leave-focal-out team priors · OXF7 probes as
falsification only, cluster-bootstrap by gamePk · OXF8 glued-duo census → K3
extrapolated/REFUSED_OOS (the identifiability floor no estimator escapes).

## Scoped claim
Per-player fingerprint pathway structurally eliminated (no reward, no data, no
shared state, no synthetic hidden-member rows). Residuals: τ̂ estimation error
(mean-zero ⇒ variance) and the OXF8 floor (flagged/refused, never guessed).

## Why this is the Vegas edge, not hygiene
The market prices movers on trailing stats — co-occurrence credit. A fingerprinting
tower inherits a cousin of that same bias, shrinking our model-vs-market
disagreement exactly where we bet. OXF removes co-occurrence-fingerprint credit
from the environment and prices the player only through his own orthogonalized
channel — so disagreements concentrate on star-adjacent movers and
stayers-after-departure, the two seams the market gets wrong.

## Cost
5 towers ×2–4h + 5 trunks ×1–2h GPU ≈ one weekend; ~1 week of trainer/loader code
(fold contract, τ̂ pre-pass, routers, probes) on top of script 19; surface adds
player folds, LOO team priors, pairwise-LOO member scalars (scripts 21/26).

*Full panel transcripts: workflow wf_82d866e8-3de journal (18 agents, ~1.1M tokens).*
