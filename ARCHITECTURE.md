# The Portability CIN — architecture & novelty claims

## What "causal AI that works" means here

Every failed "causal AI" system fails the same way: it answers intervention questions the
data never ran. Our CIN is novel not because of a bigger network but because of four
design commitments that together make its causal answers *earned*:

**C1. Nature-run interventions only (identification by natural experiment).**
The NHL runs our experiment for us: hundreds of players per season switch teams, lines,
coaches, and roles. The CIN is trained and *evaluated* on these observed context switches.
Query classes are tiered:
  - Tier 1 (shipped): environment swap — new team / linemates / role. Dense natural
    coverage; validated directly on the trade backtest.
  - Tier 2 (gated): within-support lever nudges (e.g. OZ share within the player's
    archetype's observed range). Answered with support flags.
  - Tier 3 (refused): out-of-support levers. The engine returns REFUSED_OOS, never a number.

**C2. Talent-deconfounded outcome model (one-step EM).**
Deployment is assigned by coaches who already know who is good — the classic confounder.
We break it: (a) fit outcome model B0 without talent; (b) aggregate its OOF residuals per
player into an EB-shrunken talent prior z; (c) refit B1 with z as an input. B1's
context/lever coefficients no longer absorb talent, because talent has its own channel.
(Iterating to convergence is v2; one step removes the first-order bias.)

**C3. Switch-weighted training.**
Windows near an observed context switch (first N games after a trade, line change, coach
change) get upweighted in B's loss. The model is explicitly optimized to be right where
contexts CHANGED — the exact regime the product queries — rather than dominated by
steady-state windows where context and talent are collinear.

**C4. Conformal counterfactual calibration.**
Bands are fit on the *switch population* (historical movers, frozen-at-t0 predictions),
binned by n_eff. So the 80% band shown for a hypothetical trade has measured ~80% coverage
on real trades — not on average windows, and not from any model's internal sigma.
(The prior Kalman attempt covered 12% at a claimed 80%. It is deleted, not fixed.)

Claim we can defend on a stage: *the first deployment-projection system whose
counterfactuals are (1) identified on natural experiments, (2) talent-deconfounded,
(3) support-gated, and (4) calibration-audited on the intervention distribution itself.*

## The stack

```
                       PERFECT WINDOWS (contracts/)  ~85 cols, 5 contracts
                                   |
        +--------------------------+---------------------------+
        |                          |                           |
  OOF TEAM BASELINE          PHASE A v2 (models/phase_a)   TALENT PRIOR (priors/)
  (existing, LOSO,           two-stream encoder,           EB-shrunk OOF residuals,
   D:\baseline_model_output)  probe-gated                   decayed, per strength
        |                          |                           |
        +----------+---------------+---------------------------+
                   |
             PHASE B v2 (models/phase_b)  = the CIN core
             FiLM residual MLP 4x768 (ablate vs 8x1536), 10 heteroscedastic heads,
             learnable task weights (Kendall-Gal), PAR anti-leak penalty,
             inputs: h_usage + 17 levers + curated context + TALENT PRIOR (C2),
             switch-weighted loss (C3), era-split checkpoints for freeze-legal backtests,
             GBDT twin (LightGBM) as level referee
                   |
        +----------+-----------------------------+
        |                                        |
  SCENARIO BUILDER (cin/scenario.py)       BACKTEST (backtest/)
  destination role template overwrites     ledger x freeze x era-models,
  OLD-team deployment priors; bootstrap    vs Marcel/carryover,
  real windows from the role slot          bootstrap CI on the delta
        |                                        |
  PROJECTION (cin/project.py)  <-- conformal bands (cin/conformal.py, C4)
        |                                        |
  SUPPORT GATE (cin/support.py, C1 tiers)  SEALED PREDICTIONS (seal/)
```

## MVP vs v1 split

- MVP number (October): OOF baseline + talent prior + scenario builder + conformal.
  No neural net on the critical path. Cannot fail to run.
- v1: Phase B v2 replaces the linear combination IF it beats it in the same harness.
  Phase A v2 ships only after the 5-probe leakage scorecard passes.

## Phase A v2 — decisions carried from audit

Keep: two-stream identity/context firewall, 9-token layout (d=384), seconds-share
attention bias, GRL scrubbers. Change: train 2017-2024 (not 2023-only); enable hard
negatives in InfoNCE; wire identity-usefulness head to next-game residual targets;
5-probe scorecard gates every checkpoint (probe h_skill -> teamId LOW, oz_share LOW,
next-season prior HIGH, next-season retrieval HIGH, twin-distance ratio LOW);
fix top-K-by-seconds at the data layer.

## Phase B v2 — decisions carried from audit

Keep: FiLM conditioning, heteroscedastic NLL, CF-aware early stopping, 17-lever block.
Change: add talent-prior channel (C2); 4x768 default with 8x1536 ablation; learnable
cross-task weights; switch-weighting (C3); era-split retrains; delete Kalman variance
from the serving path; GBDT twin trained on identical inputs as referee.
