# Phase A — exactly what trains it, and how bias is kept out

Companion to ARCHITECTURE_PLAN.md §2. This is the audited answer to two questions:
**(1) what features does the tower eat, exactly** (as assembled by script 21 into
`trainer_surface/`), and **(2) where does each kind of bias die** — star players
bleeding into the environment, low-minute rows dominating, old-team residue, and
future-peeking.

## 1. The feature manifest (what the tower sees per window)

### The people (the intervention itself)
| Feature | Form | Bias defense baked in |
|---|---|---|
| `with_ids`, `vs_ids` | learned 96d ID vectors, ≤5 each side | shared starved table (~1.7k players); focal player NEVER present |
| `with_seconds`, `vs_seconds` | seconds-share attention bias | a 10-second cameo teammate cannot pull the pooled vector like a full-shift one |
| member form (`with_form`, `vs_form`) | season-lagged on-ice xGF60 | **F-1 echo rule**: computed ONLY from prior seasons, so a teammate's form never contains this season's shared ice with the focal player |
| member `gsc` | games since member's own team change | everyone settles, not just the focal player (A-1) |
| member age / hand / position | computed from birthDate at game date; static bio | time-invariant → leak-proof by construction (still audited) |

### The situation
`period`, `period_time_bucket`, `score_bucket`, `start_regime`, `lever_zone_start`,
`fo_loc_enum`, `stoppage_class_at_start`, `after_icing` (+ derived `ai_OZ_start`/
`ai_DZ_start`), `bench_rights`, `long_change`, `home_away`, `rinkid`, `skater_diff`.

### The schedule & bench
`rest_days_team`, `b2b_team` (computed from game-date gaps, capped at 7),
`games_since_team_change` (focal settling — **modeled, not smeared**, F-2),
`coach_id` + `coach_tenure_games` (F-3), and the 10 team priors
(pace / xG rates / goalie tiers, all `_prior` = strictly pre-game, lag-audited).

### Entry-side timing (legal) vs exit-side (banned)
`shift_count_in_window`, `time_since_last_shift_s`, `onice_elapsed_at_window_start`,
`entered_after_start`, `entry_offset_s` — coach decisions made BEFORE the exposure.
Exit-side twins (`exited_before_end`, `exit_offset_s`, `stint_duration_st`) are
**banned**: players leave early *because* of what happened.

### Absent by design
- **The focal player. Nothing about him enters the tower.** No ID, no priors, no
  deployment history. His talent arrives in Phase B through a separate pipe.
- `duration` as an input (carried only as `exposure_duration_w` for loss
  weighting — stop-bias measured at +51%).
- `season` (era identity undefined at serving; era enters via team priors).
- All in-window event counts (outcomes wearing feature costumes).

## 2. "Do we need residuals?" — yes, and they already exist. Where each bias dies:

### Star players biasing the environment (the big one)
Three separate walls:
1. **The focal player is not in his own environment.** The tower literally cannot
   see who it's describing the ice for — so McDavid's greatness can't leak into
   "what Edmonton's environment is worth" *through his own row*.
2. **The echo wall (F-1).** His greatness CAN reflect off teammates: Draisaitl's
   recent on-ice numbers are partly McDavid. That's why member form is
   season-lagged — measured: naive-current correlation 0.310 → lagged 0.267, and
   the remaining correlation is real line-building assortativity the tower SHOULD
   see, not echo. Probe P3 tests exactly the *incremental* leakage every epoch.
3. **The residual wall (K-1).** Talent itself IS a residual: what the player
   produced **minus** what an average player would have produced in his exact
   windows (OOF baseline mu), decayed, league-demeaned, EB-shrunk. And the K-1
   ship-gate requires talent to be **re-derived against the trained tower's own
   environment** (EM round 2) before v1 ships — so linemate credit that the tower
   claims is subtracted back out of the player, never counted twice. This is
   precisely the "is he real without his star linemate?" correction.

### Low-minute rows dominating (they can't — four dampers)
1. **Exposure weighting:** every window's loss is weighted by its seconds. A 6-second
   window contributes ~1/100th of a 10-minute stretch. Rate noise from tiny
   denominators never steers the gradient.
2. **Heteroscedastic heads:** the model predicts its own noise level; short-exposure
   windows get wide predicted variance, which *automatically* down-weights their
   gradient (Gaussian NLL divides the error by sigma²).
3. **EB shrinkage on talent:** a player with 40 career minutes gets pulled almost
   entirely to league average (fitted K in effective minutes); he cannot carry an
   extreme talent number into training or serving.
4. **Seconds-biased pooling:** within a window's member set, attention logits get
   `+log(seconds)` — brief on-ice overlaps barely register in h_with/h_vs.

### Old-team residue in a mover's numbers
`games_since_team_change` (his AND each member's) is an explicit input, so the
settling-in curve is learned as a curve — instead of contaminating every other
weight. Switch-weighted training (×3 on post-move windows) makes the model pay
attention to exactly the regime we sell.

### Future-peeking
Every prior column carries a lag-suffix contract, and script 13 measures
past-vs-future prediction symmetry for every numeric column (planted cheater
screams at ~0.90 gap; real features must sit under 0.02). New surface columns
(coach tenure, rest days, member bio/form) go through the same audit before the
trainer may read them — `assert_legal()` refuses unclassified features at import.

## 3. What the tower trains AGAINST (targets, never features)
Per window: MoneyPuck xG for/against + rush/rebound/chaos splits (primary,
one licensed ruler end to end), in-house xG as auxiliary witness, goal/shot
counts (Poisson with exposure offsets), and micro-stats (hits/give/take) whose
gradients are **detached from the tower** so arena-scorer bias (T19, measured
2.25× giveaway drift) can never shape the environment representation.

*Assembled by `scripts/21_build_trainer_surface.py`; per-season coverage and
file hashes in `artifacts/trainer_surface_manifest.json`.*
