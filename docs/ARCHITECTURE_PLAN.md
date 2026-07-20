# The OptBot CIN — Final Architecture Plan (canonical, v2.1 · Jul 2026)

Single source of truth for Phases A/B/C. v2.1 incorporates the second adversarial
pass: K-1..K-3 (critical), S/A/B-series hardening. Every fix is listed inline where
it lives, marked [Kx]/[Sx]/[Ax]/[Bx].

---

## 0. Estimand

theta(p, E', t0) = E[ first-40-game on-ice impact | do(E ~ E'), history(p, <t0) ]
Shipped: xGF% + conformal band (coverage attached, extrapolation-flagged [K3])
+ value lines (penalty economy · finishing economy · age delta).

---

## 1. SCENARIO BUILDER

- Slot's real pre-t0 windows, bootstrap x2000, multi-incumbent pools (T22, dilutes
  incumbent bleed [S1]).
- Future-schedule weighting by JOINT (opponent, venue, rest-class) — never
  marginals; no Frankenstein windows [S2].
- Settling-in integrated over games 1..40 via gsc input [F2].
- Overrides: linemates / goalie / coach.
- **[K3] JOINT-SUPPORT SCORE**: co-occurrence prior over the pinned WITH-set +
  role-delta distance from the ledger distribution. Below threshold →
  band emitted with explicit `extrapolated: true`; far below → REFUSED_OOS.
  Marginal support checks alone are not enough for pinned combinations.
- Opponent five at serving is team-known, lineup-guessed — the weakest-typed
  input [A4]; opponent-line sampling sensitivity is a standing ablation.

## 2. PHASE A — Environment Tower (~2M params)

Member representation (teammates + opponents, <=5 each):
  STATIC CORE: learned 96d ID vector (shared table ~1.7k; starved)
  DYNAMIC SHELL (computed, as-of-date): age · position · handedness · TOI-role ·
    recent form (LEAVE-FOCAL-OUT or season-lagged — the echo rule [F1]) ·
    **member's own games_since_team_change [A1] — everyone settles, not just
    the focal player; also dilutes old-system residue in cores**
  → member MLP → seconds-biased attention pooling → h_with, h_vs (256d)

Context: score/zone/period/rink/rest/entry-mode/manpower embeds · team priors
(pace/xG/goalie) · focal gsc [F2] · **coach: id embedding + COACH SHELL
(tenure, career style priors) with shrinkage — ~100 coaches, thin samples,
and new-coach UNKs are exactly the interesting case [A5/F3]** → h_ctx (256d)

Fusion: 2 pre-norm attention blocks over [h_with, h_vs, h_ctx] + CLS → h_env (256d)

Absent by design: focal identity · season/era ids · convicted features.
Teammate dropout 20%. UNK = shell-only members (AHL inference mode).

## 3. PHASE B — CIN Trunk

x = [h_env ⊕ talent channel ⊕ bio] → 4x768 SwiGLU residual, FiLM(17 levers)
→ 10 heads (xG heteroscedastic Gaussian; counts Poisson-offset; Kendall-Gal).
- **[B2] micro-stat heads (hits/give/take) gradients DETACHED from the tower** —
  arena-scorer noise (T19) may not shape h_env; those heads learn on the trunk only.
- Exposure-seconds weights (stop-bias defense) · switch-weight x3 (safe: gsc is
  an input, adjustment is modeled not smeared [F2]).
- Era-split checkpoints (<=2021/22/23/24). **[B3] all ablation comparisons hold
  era fixed.**
- **[B4] Gate mapping, explicit:** GBDT twin (tabular-only) judges the TRUNK's
  value-add; the TOWER's own gate is ablation vs slot-average env. Two gates,
  two questions.
- Internal sigmas never face humans; conformal only.

## 4. PHASE C — Player Side

- NOW: talent scalars (proven −7.6%): decayed, league-demeaned OOF residuals, EB-shrunk.
- **[K1] HARD SHIP-GATE FOR v1: talent must be RE-DERIVED as residual vs the
  TOWER-based environment before v1 ships (EM round 2, now mandatory).**
  Reason: current residuals contain unclaimed linemate credit (baseline is
  identity-blind). v0: errors partially cancel. v1: the tower credits linemates
  → same value counted twice → systematic overprediction for star-adjacent
  players leaving their stars. The exact GM question ("is he real without X?")
  would be answered wrong. No v1 ship without re-derivation.
- Cheap layers: penalty economy ✅ · finishing economy (shot cards) · age delta.
- v2: style vector (8-16d, offense-typed for now [C2] — defensive-style source
  queued) → interacts with tower people-embeddings = FIT/chemistry, gated by
  identifiability census.

## 5. GATES — the full ship checklist for v1

1. 5-probe scorecard green every epoch (incl. zone-decay probe).
2. Tower ablation: beats slot-average env (era-fixed [B3]).
3. **[K1] Talent re-derived vs tower-env (EM round 2) and sniff re-passed.**
4. Beat v0 (5.74) on the 860-move harness, CI clear of zero.
5. Beat the GBDT twin (trunk gate [B4]).
6. Conformal refit on v1 errors; coverage 78-82% per bin.
7. **[K2] Role-delta slice reported**: error vs |role change| — quantifies the
  lever-transfer ceiling (T23's sharpest tooth) and what the v2 usage-model buys.
  (Runnable on v0 TODAY — baseline slice before GPU.)
Fail any → v0 ships in October, v1 iterates. The company never depends on the net.

## 6. DISCLOSED LIMITS (the honest floor of the design)

- T23: environments adapt to players (usage becomes HIS). Priced by conformal;
  v2 frontier = two-stage usage-then-outcome model.
- [A6] settling curve learned on trade-selected players (composition blended in).
- [A4] serving-time opponent lineups are sampled, not known.
- [G1/K3] bands are measured on real moves; unlike-anything hypotheticals carry
  the `extrapolated` flag rather than a false guarantee.

## 7. BUILD ORDER

1. Rollups: LOO member-form [F1] · member gsc [A1] · coach table+shell [A5] → v3.1
2. [K2] role-delta slice of the 860 backtest (CPU, tonight-class)
3. Trainer (scripts/19): everything above baked in → READY FOR GPU
4. Train + era-fixed ablations → tower gate
5. [K1] talent re-derivation vs tower-env → sniff → harness shootout (full gate list)
6. [K3] joint-support + extrapolation flags in serving
7. September seal · October public scoring
