# The OptBot CIN — Final Architecture Plan (canonical, v2.0 · Jul 2026)

The single source of truth for how Phases A/B/C are built, trained, gated, and
composed. Supersedes scattered specs. Incorporates audit fixes F-1..F-6 and names
the platform's deepest assumption (T23).

---

## 0. The estimand everything serves

For player p, freeze date t0, hypothetical destination environment E':

    theta(p, E', t0) = E[ first-40-game on-ice impact | do(E ~ E'), history(p, <t0) ]

shipped as xGF% with a conformal band whose coverage is measured, plus deterministic
value lines (penalty economy; finishing economy F-5; age adjustment F-6).

---

## 1. PHASE A — the Environment Tower  (h_env, 256d)

Job: encode "the ice around the focal player" so that swapping it is exact.

### Inputs (tower_schema.py is the law)
- PEOPLE: <=5 teammates + <=5 opponents. Each member =
  [ static ID core (96d, shared learned table, ~1.7k players)
    ⊕ dynamic shell: age-at-date, position, handedness, TOI-role,
      recent-form prior — **LEAVE-FOCAL-OUT (F-1)** or season-lagged ]
  -> member MLP -> seconds-biased attention pooling -> h_with, h_vs
- CONTEXT: score/zone-start/period/rink/rest/entry-mode/manpower embeddings
  + team priors (pace, xG rates, goalie tier/GSAA) via LayerNorm+MLP
  + **games_since_team_change (F-2: the settling-in curve as explicit input)**
  + **coach_id + tenure (F-3, from game_coach_ids_2017_2025)**
- FUSION: 2 pre-norm attention blocks over [h_with, h_vs, h_ctx] + CLS -> h_env

### Hard exclusions (unchanged, enforced)
No focal identity anywhere. No banned features (traced convictions). No season/era
embedding. UNK path: unseen IDs -> position x role fallback core + FULL dynamic
shell (the AHL inference mode — unseen players are shell-only, graceful).
Teammate dropout 15-25% (anti-fingerprint + serving realism).
Size: ~2-3M params, deliberately starved.

### Phase A gates
5-probe scorecard/epoch (incl. zone-decay probe) + ablation vs slot-average env.

---

## 2. PHASE B — the CIN Trunk

    x = [ h_env  ⊕  talent-channel(z_off, z_def, se, n_eff)  ⊕  bio(age, age², hand) ]
    4 x 768 pre-norm SwiGLU residual blocks, FiLM-conditioned on the 17-lever vector
    -> 10 heads (mu, log_sigma): xGF/xGA Gaussian-heteroscedastic;
       counts Poisson-with-exposure-offset; Kendall-Gal learnable task weights

- Separated pathways = C2 structural deconfounding (talent cannot be re-derived
  from usage; env cannot contain the focal player).
- Loss weights: exposure seconds (stop-bias defense) x switch-weight 3x within
  10 games of a change — **now safe because gsc is an input (F-2), the model
  attributes adjustment effects to the curve, not to ambient reweighting.**
- Era-split checkpoints (<=2021/22/23/24) — freeze-legal backtesting forever.
- GBDT twin on the identical legal tabular surface = referee.
- Internal sigmas: training + ranking only. Human-facing bands = conformal, always.

### Phase B gates
Beat v0 (5.74 on the 860-move harness) CI-clear; beat the twin; probes green.
Loses => v0 ships; B iterates. The company never depends on the net.

---

## 3. PHASE C — the Player Side (computed first, learned later)

- NOW (validated): talent scalars — decayed, league-demeaned OOF residuals,
  EB-shrunk. This IS the proven -7.6%.
- CHEAP LAYERS (computed, no GPU): penalty economy (shipped) · finishing economy
  from shot cards (F-5) · age-curve delta (F-6).
- v2 (post-A/B, gated): style vector 8-16d distilled from the 13,708 shot-
  personality cards + process priors; interacts with tower people-embeddings
  for FIT/chemistry. Gated by identifiability census + its own probes.
- v2.5: iterate talent<->model EM one more round; LOO team priors (T21).

---

## 4. SERVING FLOW (one query, end to end)

    query(p, dest, line, linemates?, coach?, date)
      -> scenario: slot windows < t0, bootstrap 2000
         + **known-future-schedule weighting (F-4: upcoming opponents are
           public at t0 — legally visible future)**
         + gsc set to 1..40 across horizon (adjustment curve integrated)
         + goalie/linemate/coach overrides
      -> support gate (T1 answer / T2 in-range / T3 REFUSED_OOS)
      -> v1 trunk forward passes (or v0 fallback: mu-average + talent)
      -> xGF% + conformal band (with achieved coverage attached)
      -> + penalty economy line + finishing line + age note

---

## 5. THE DEEPEST ASSUMPTION — T23, disclosed

Environments adapt to players: after arrival, his usage becomes HIS, lines
reshuffle around him. We hold the environment fixed at the role template; reality
flexes it. Priced (conformal is fit on real moves where adaptation happened);
kept honest (identical crudeness in backtest and product). v2 frontier: two-stage
model — predict his USAGE at the destination first, then outcomes given usage.
No one in this industry models this; we at least name it.

---

## 6. BUILD ORDER

1. LOO teammate-form rollup (F-1) + coach join (F-3) -> perfect windows v3.1
2. Tower+trunk trainer (scripts/19): assert_legal, probes/epoch, era splits,
   gsc input, teammate dropout, hybrid likelihoods  [READY FOR GPU]
3. Train + ablations (features earn their place empirically)
4. The shootout on the 860-move harness
5. F-4 scenario schedule-weighting, F-5/F-6 value lines (any afternoon)
6. September: seal 2026-27. October: public scoring.
