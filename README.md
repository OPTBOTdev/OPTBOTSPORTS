# OptBot — The Portability CIN

**Give us a player and a destination. Before he plays a single shift there, we tell you his
5v5 impact — with a calibrated band, a frozen-clock backtest, and predictions sealed by hash
before the season.**

OptBot's Causal Intervention Network (CIN) answers one intervention query — *what does this
player become in that environment?* — and answers it in a way an NHL analytics department
cannot dismantle. This document is the technical whitepaper: the full confounder audit, the
neural architecture with the reasoning behind every choice, and the current build status.

---

## 1. What "the first causal AI that works" actually requires

Every failed causal-AI system dies one of four deaths: it answers interventions the data never
ran; it confuses talent with treatment; it extrapolates confidently outside support; or its
uncertainty is fiction. The CIN is built as four commitments, each one closing a grave:

| # | Commitment | One-line defense |
|---|---|---|
| C1 | **Natural-experiment identification** | We only answer interventions nature runs at scale — 729 team-switches in our own ledger. Tiered support gating refuses the rest. |
| C2 | **Talent deconfounding** | Talent enters the model through its own frozen, out-of-fold channel, so environment and lever weights cannot absorb it. |
| C3 | **Switch-weighted training** | The model is optimized on the windows where context *changed* — the exact regime the product queries. |
| C4 | **Conformal counterfactual calibration** | Bands are fit on historical movers, frozen at their move dates, and every band ships with its measured coverage. |

The defensible claim: *the first deployment-projection system whose counterfactuals are
identified on natural experiments, talent-deconfounded, support-gated, and calibration-audited
on the intervention distribution itself.* Claimed that narrowly, it survives the sharpest room.

---

## 2. The Confounder Audit

Accuracy earns the contract, and confounders are how accuracy dies quietly. This is the full
threat model: every mechanism we know of that could corrupt the projection, what it would do
to us, and its current disposition. **Status legend:** SOLVED (structural fix, verified) ·
MITIGATED (first-order handled, residual priced into bands) · GATED (refused rather than
answered) · PRICED (irreducible; carried honestly in the band) · OPEN (scheduled work).

### T1 — Deployment selection confounding
Coaches give the best usage to players they already believe are good. A naive model concludes
*OZ starts cause production*. **Defense:** C2 — talent has its own input pathway computed from
out-of-fold residuals, so usage coefficients no longer proxy for skill; plus C1 — we ship the
environment-swap query, not arbitrary usage knobs. **Status: MITIGATED** (one-step EM; iterating
the loop is v2).

### T2 — Talent–residual circularity
Talent is defined as residual from a model that itself should have controlled for talent. A
closed loop can bake bias into both. **Defense:** the residuals come from a *baseline that never
saw the player's identity* (environment-only OOF model), so the first iteration is clean by
construction; the refit-with-talent model is only used for projection, never to redefine talent
in the same pass. **Status: MITIGATED.**

### T3 — Linemate collinearity (the RAPM disease)
Players who always share the ice cannot be statistically separated; a career-long duo's credit
is split arbitrarily, and a projection that separates them inherits the arbitrariness.
**Defense:** windows are short and lineups churn within games; teammate identities are explicit
inputs (pooled, seconds-weighted); and the conformal band — fit on real movers, including
separated duos — prices what separation error remains. **Status: PRICED**, flagged: projections
for a player leaving a 4-year duo carry the widest honest bands.

### T4 — Ledger survivorship (collider bias)
Our backtest requires ≥20 GP after the move. Players who moved and *collapsed* got benched or
waived and exit the sample — conditioning on a consequence of the outcome. The backtest
therefore slightly flatters everyone, including Marcel. **Defense:** the bar and the model face
the identical filter, so the *comparison* (v0 − Marcel) is unbiased even where the *levels* are
optimistic; sensitivity re-run at ≥10 GP scheduled. **Status: MITIGATED**, sensitivity OPEN.

### T5 — Trade selection bias
Teams acquire players they believe fit their system; observed moves are biased toward *good*
fits. Our counterfactuals for arbitrary hypothetical destinations extrapolate beyond that
biased sample. **Defense:** honesty in scope — backtest numbers describe realistic-move
accuracy; the support gate marks hypotheticals that look unlike any historical move.
**Status: GATED.**

### T6 — Injury unobservables
Players are sometimes moved *because* they are quietly hurt; the post-move decline is injury,
not environment, and no public feature sees it. **Defense:** none exists at the individual
level — this is the canonical unobservable. It inflates band width honestly (the conformal fit
absorbs its frequency). **Status: PRICED.**

### T7 — Role-assignment error at t0
We must guess his line slot before he plays. Wrong slot, wrong environment template.
**Defense:** deterministic TOI-rank rule — deliberately identical crudeness in backtest and
product, so measured accuracy already contains this error; interactive role selection in the
demo lets the GM override. **Status: PRICED** (and it is a *feature* in the room: "slide him to
line 3" re-projects live).

### T8 — Regression to the mean masquerading as causality
A player moves after a career year; he declines anywhere. A naive reading credits the new
environment. **Defense:** EB shrinkage on the talent prior (n_eff/(n_eff+K)) is precisely an
anti-RTM device, and Marcel — itself an RTM machine — is the bar, so we only win by modeling
something *beyond* RTM. **Status: SOLVED** structurally.

### T9 — Era drift and COVID seasons
Scoring environments drift; 2019-21 seasons are short and weird. **Defense:** per-season
league demeaning of residuals (talent is league-relative by construction); era-split model
checkpoints; COVID seasons retained but down-weighted by their smaller n. **Status: SOLVED.**

### T10 — Goalie contamination of defensive talent
On-ice xGA measures shot suppression, but *goal* outcomes drag goaltending into any GA-based
signal. **Defense:** the product speaks xGA/xGF% only — goalie-independent by definition of
expected goals; goalie priors enter the environment side explicitly (tier, GSAA-EB).
**Status: SOLVED** for xG-based claims; we deliberately never promise goals.

### T11 — Score-state effects
Teams protect leads and chase deficits; raw rates confound score context with ability.
**Defense:** score bucket is a first-class context feature in every window and in the
destination template (built from the slot's real score-state distribution). **Status: SOLVED.**

### T12 — Out-of-support extrapolation
"What if he took 70% OZ starts?" — no player of his archetype ever did; any answer is fiction.
**Defense:** the tier system. T1 environment swaps are dense in data; T2 lever nudges are
answered only inside the archetype's observed [q05, q95]; T3 returns REFUSED_OOS — a refusal
object, not a number. **Status: GATED**, enforced in code.

### T13 — The noise floor
First-40-games on-ice xGF% carries ~3–4 points of pure sampling noise. No model beats the
floor; pretending otherwise is how systems die in diligence. **Defense:** we report RMSE
*relative to Marcel* with a bootstrap CI, disclose the floor estimate alongside, and the
conformal band inherits it honestly. **Status: PRICED** — and stated in the pitch, which reads
as sophistication, not weakness.

### T14 — Data integrity (the July 2026 excavation)
The audit found three silent corruptions that would have been fatal: (a) the player-game
observations file had **actuals = 0 in every row** — every "residual" was −mu, the true root
cause of the 12%-coverage uncertainty model; (b) the 10.3M-row window spine carried **1.49M
full-row duplicates** across all 8 seasons, double-counting exposure; (c) the baseline mu
**underpredicts by ~40%** globally (documented in the legacy repo, never fixed). **Defense:**
(a) rebuilt from per-game ground truth — 100% join rate, 7 seasons; (b) deduplicated, zero
conflicts, and the duplicate-carrying mu files are key-deduped at every load; (c) neutralized
by league-relative demeaning. A validation gate (script 00) now runs these exact checks before
any build. **Status: SOLVED**, with one residual OPEN: obs-level mu was originally summed over
duplicated windows for ~2,425 games — recompute scheduled; demeaning absorbs its mean effect.

### T15 — Baseline integrity
If Marcel is accidentally weak (or strong), the headline is fiction in either direction.
**Defense:** Marcel implemented to the standard recipe (5/4/3 weights, minutes-based regression,
age bump) plus a *stronger* strawman (Marcel + half team-delta) that a hostile quant would
propose; both scored under the identical freeze and GP filters as v0. **Status: SOLVED.**

### T16 — Deadline-context shift
Deadline acquisitions join contenders mid-playoff-push; usage and score contexts shift in ways
offseason moves don't. **Defense:** move_type is a ledger field; accuracy is reported sliced
(196 in-season trades vs 533 offseason moves), and the claim is scoped to where we win.
**Status: MITIGATED.**

### T17 — Cold starts
The legacy priors emitted zeros with collapsed SEs for 97.9% of early-season rows — every
model trained on them learned "new season = league-zero player." **Defense:** rebuilt prior
carries decayed prior-season value across the offseason; SE has a floor and *widens* at low
n_eff; rookies get near-total shrinkage and maximal bands rather than fake confidence.
**Status: SOLVED.**

### T18 — Uncertainty theater
The old Kalman bands claimed 80% and delivered 12% — worse than no bands, because they
manufacture false trust. **Defense:** internal model sigmas are banned from human-facing
output by contract; bands come only from split-conformal residuals of the frozen backtest,
binned by n_eff, and every band object carries its `achieved_coverage` field. A band without
its coverage number cannot be emitted. **Status: SOLVED**, enforced by code.

The pattern worth noticing: nothing above is hand-waved. Every threat is either structurally
closed, measurably priced into the band, or refused at the API. That discipline — not any
single model — is the moat.

---

## 3. Architecture — and why we are confident in each piece

The decomposition rule that governs everything: **every input is (a) what travels with the
player, (b) what belongs to the destination, or (c) a coaching knob.** The product's
intervention is *swap (b), keep (a), set (c)* — so the architecture must keep the three
categories separable by construction. The decomposition is the causal graph.

### 3.1 Data layer — the Perfect Window contract
One row = one (player, window): keys/exposure · PRE-timing context (knowable at the opening
faceoff) · people payload (seconds-desc teammate/opponent IDs) · lagged-and-shrunk priors, each
with its n_eff · OOF baseline mu · outcomes (never features). Five contracts enforced by a
validator that fails builds loudly; the schema is versioned. **Confidence: earned** — this
validator caught all three T14 corruptions on its first run against real data.

### 3.2 The Talent Prior (what Phase C became)
Not a neural network — a statistic with the right properties:

    raw(p,t)    = decayed, TOI-weighted mean of league-demeaned OOF residuals, games < t only
    shrunk(p,t) = n_eff/(n_eff+K) · raw(p,t)        K fit out-of-time, never hand-tuned
    se(p,t)     = max(sqrt(var/n_eff), floor)        widens at low n_eff, never collapses

Leak-proof by construction (state recorded *before* each game is ingested — unit-tested), frozen
at any date in O(1). **Confidence: empirical** — on the first correct residuals this project
ever produced, the prior ranked McDavid #1, Draisaitl #2, MacKinnon #3, with Kucherov, Crosby,
Makar and Barzal in the top 15, and Kucherov's *defensive* talent negative. Nobody tuned it to
do that.

### 3.3 The Environment Tower (what Phase A became)
The legacy Phase A was a two-stream contrastive encoder whose focal-player ID embedding
memorized everything — talent, system, coach — requiring a firewall and adversarial scrubbers
to contain it. The remap deletes the disease instead of treating it: **the focal player has no
embedding.** He enters only as talent scalars + bio + age (explicit, so aging transfers across
players). What remains of Phase A is the environment encoder:

    ctx categoricals (score/zone/period/rink/rest)  →  embeddings
    team priors (pace, xG rates, goalie tier/GSAA)  →  explicit features
    WITH/VS teammate & opponent sets                →  ID embeddings, seconds-weighted pooling
                                                     ↓
                                                  h_env (256d)

Teammate embeddings are kept because sets of 800+ players is exactly where embeddings earn
their keep — and teammates are, from the focal player's perspective, environment: swappable by
design. Acceptance test: linear probes must NOT recover focal talent or focal identity from
h_env. **Confidence: structural** — the leak channel isn't scrubbed, it's absent; and the tower
degrades gracefully to the OOF baseline mu when untrained (which is precisely MVP v0).

### 3.4 The CIN core (what Phase B became)

    h_env (environment tower)  ─┐
    talent channel (own path)  ─┼→ residual trunk 4×768, SwiGLU, FiLM-conditioned on 17 levers
    bio/age (explicit)         ─┘        ↓
                                10 heteroscedastic heads (xGF, xGA, GF, GA, SF, SA, micro)
                                learnable per-task weights (Kendall–Gal)

Design decisions and their reasons: **FiLM for levers** (knobs modulate the computation rather
than mixing into it — clean intervention semantics); **talent's own pathway** (C2: the trunk
cannot re-derive talent from usage, because talent arrives pre-computed and frozen); **switch
weighting** (windows within 10 games of an observed team change weigh 3× — C3); **4×768 before
8×1536** (era-split retrains must be cheap; the big ablation must *earn* its cost); **a GBDT
twin** trained on identical features as referee — if LightGBM matches the trunk, the trunk does
not ship on aesthetics; **era-split checkpoints** (trained strictly-before 2021/22/23/24) so
every backtested projection is freeze-legal. Internal sigmas rank uncertainty and weight
training; they never become a human-facing band (T18). **Confidence: conditional and honestly
so** — the neural core ships only if it beats v0 in the same harness. The company does not
depend on it.

### 3.5 Support gate and conformal layer
Tier 1 environment swaps: answered, with slot-support and prior-quality checks. Tier 2 lever
nudges: answered inside observed archetype ranges. Tier 3: `REFUSED_OOS`. Bands: split-conformal
on frozen-clock backtest errors, n_eff-binned, coverage always attached. **Confidence: this
layer is the difference between "causal AI" as marketing and as engineering.**

### 3.6 MVP v0 — the projection that cannot fail to run

    projected_xGF60 = E_slot[ mu_xgf60 | destination, line, pre-t0 ] + talent_off_shrunk
    projected_xGA60 = E_slot[ mu_xga60 | ... ] − talent_def_shrunk

Environment from the destination slot's real windows (bootstrap-resampled, correlations
intact), player from the frozen prior, band from conformal. Explainable in one sentence to a
GM. v1 replaces the first term with the trained tower + trunk *if and only if* it wins.

---

## 4. The path to MVP and the contract

**The number that wins:** beat Marcel (the industry-standard projection) on frozen-clock
team-switcher backtests with a bootstrap CI clear of zero — then hand the GM sealed,
hash-timestamped projections for the current season's movers and let October score us publicly.
Accuracy + receipts + honest bands is the contract-winning combination; any one alone is not.

Pipeline (each script gated by the previous):

    00 validate → 00b/00c repair → 01 ledger → 02 talent prior → 02b lines+windows
    → 03 baselines (the bar) → 04 interactive projection → 05 backtest (THE number)
    → 06 sealed predictions

## 5. Current status (July 8, 2026)

| Milestone | Status | Evidence |
|---|---|---|
| Repo, CI, contracts, tests | ✅ LIVE | github.com/OPTBOTdev/OPTBOTSPORTS — CI green, 7/7 tests |
| Data excavation & repair (T14, T17) | ✅ DONE | 1,491,039 dupes removed · actuals rebuilt, 100% join, 7 seasons |
| Talent prior + sniff test | ✅ PASSED | McDavid #1 · Draisaitl #2 · MacKinnon #3 |
| Trade/UFA ledger | ✅ BUILT | 729 qualifying moves (196 trades, 533 offseason) |
| The bar (Marcel) | ✅ MEASURED | **RMSE 5.67 xGF%** (carryover: 7.40) on all 729 |
| Perfect windows + lines | ✅ BUILT | 7.52M rows, 0 dup keys, 100% mu & line coverage |
| **Backtest: v0 vs the bar** | ✅ **CLAIM PROVEN** | **RMSE 5.30 vs Marcel 5.80 (−8.6%), 95% CI [−0.67, −0.33] — clear of zero, n=635** |
| — sliced by move type | ✅ | trades 5.14 · offseason 5.37 — strongest on in-season trades |
| Conformal bands (T18 replacement) | ✅ CALIBRATED | **79.8% achieved at 80% target** (vs legacy Kalman's 12%) |
| K refit vs next-season aggregates | ⏳ NEXT | current fit hit grid edge — noisy per-game target |
| Obs-mu recompute on deduped windows | ⏳ NEXT | residual T14 item |
| Environment tower + CIN trunk training | ⏳ GPU QUEUED | trainers spec'd; era-split protocol defined |
| Sealed 2026-27 predictions | ⏳ SEPT | script ready, waits on bands |

## 6. Honest risk register (what we tell investors before they ask)

1. ~~If v0 does not clear Marcel with CI room, the claim narrows~~ — **resolved July 8: v0 cleared the bar with the full CI below zero.** The residual version of this risk: the margin (−8.6%) must survive the scheduled sensitivity re-runs (≥10 GP filter, K refit, obs-mu recompute).
2. The noise floor bounds every promise: we sell *calibrated* foresight, never certainty.
3. The neural CIN is upside, not dependency: v0 runs on audited statistics end to end.
4. AHL/ECHL extension is an architecture-ready data acquisition, not a research bet — the same swap query with a league-translation layer, which is exactly what partner data funds.

---

*OptBot · Portability CIN · technical whitepaper v1.0 · July 2026 · Confidential*
