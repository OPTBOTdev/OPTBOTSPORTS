# OptBot — Portability CIN

**One query, proven:** give us a player and a destination context, and before he plays a
shift there we project his 5v5 on-ice impact — with calibrated uncertainty and receipts.

This repo is the clean rebuild of the Phase A/B/C research stack (audited July 2026),
reorganized around the October MVP and the Causal Intervention Network (CIN) design in
[ARCHITECTURE.md](ARCHITECTURE.md). Old→new file mapping: [MIGRATION_MAP.md](MIGRATION_MAP.md).

## Layout

```
optbot/
  contracts/   window schema contract + hard validators (the "perfect window")
  data/        dataset QA suite, perfect-window builder, line/role table, trade ledger
  priors/      EB-shrunken residual talent prior + cold-start fixes
  baselines/   Marcel + carryover (the bars we must beat)
  cin/         scenario builder, projection engine, support gating, conformal bands
  models/      phase_a (identity/context encoder v2), phase_b (outcome net v2 + GBDT twin)
  backtest/    freeze-disciplined trade backtest harness + metrics
  seal/        hash-sealed prediction pipeline (2026-27 movers)
scripts/       numbered entrypoints, run in order 00..06
configs/       single source of truth for paths + hyperparams
```

## The MVP pipeline (run order)

```
00_validate_data.py     # QA every input parquet; refuses to proceed on contract breaks
01_build_ledger.py      # trade/UFA ledger 2021-2025
02_build_talent_prior.py# EB-shrunken residual talent from phaseC player-game observations
03_run_baselines.py     # Marcel + carryover on the ledger  -> the bar
04_project_trades.py    # CIN projection v0 for every ledger move (frozen at t0)
05_backtest.py          # score vs baselines, fit conformal bands, produce THE number
06_seal_predictions.py  # freeze + SHA-256 the 2026-27 mover projections
```

## Ground rules (enforced, not aspirational)

1. Every player-performance feature is lagged (`*_lag`/`*_prior`) and EB-shrunk, and ships
   with its `n_eff` — checked by `contracts/window_schema.py`, build fails otherwise.
2. Every context feature is knowable at the window's opening faceoff (PRE-timing rule).
3. Outcome columns are targets, never inputs. The validator greps the feature lists.
4. All backtest inputs are frozen strictly before each move's `t0`. Era-split models only.
5. Uncertainty shown to a human comes from conformal residuals, never from a model's
   internal sigma. Coverage is reported next to every band.
6. Counterfactual queries outside observed support are flagged, not answered.

## Data roots (see configs/default.yaml)

- `D:\baseline_model_output\`  — OOF baseline + player-game residuals 2017-18..2024-25 (MVP spine)
- `D:\combined_player_windows_2017_2024.parquet` — 10.3M-row window spine (59 cols)
- `D:\phaseB\` — Dec-gen Phase B net, lever curves, Phase C matrices (reference/v1)
- `D:\Final\`  — Phase A embeddings + priors per season
