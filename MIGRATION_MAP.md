# Old -> New migration map

| Old asset | State | New home / action |
|---|---|---|
| `D:\baseline_model_output\phaseC_player_game_observations*.parquet` | GOOD (2017-18..2024-25, all strengths, n_eff) | consumed by `optbot/priors/talent.py` — MVP spine |
| `D:\baseline_model_output\player_windows_with_baseline_*.parquet` | GOOD (OOF mu/sigma per player-window) | consumed by `optbot/cin/scenario.py` + `project.py` |
| `D:\combined_player_windows_2017_2024.parquet` (59 cols) | GOOD window spine | base table for `optbot/data/build_perfect_windows.py` |
| `Downloads\Phaseb\build_causal_priors.py` (lag/EB discipline) | GOLD pattern | pattern reused in `optbot/priors/talent.py`; cold-start fix added |
| `Downloads\Phaseb\train_baseline_model.py` (LOSO GLM+LGBM) | GOOD, keep as-is | referenced, not ported; its OOF outputs are inputs here |
| `Downloads\Phaseb\phase_d\add_lines_pairs_units_from_onice_cooccurrence.py` | GOOD idea, buried | promoted to `optbot/data/lines.py` (first-class role table) |
| `D:\phaseb.py` + `D:\nuib.py` (PolicyAlignedOutcomeNet) | GOOD bones, EV-only | re-implemented slim in `optbot/models/phase_b/outcome_net.py` (+talent channel, +switch weights) |
| `D:\phaseB\lever_curves*` | VALIDATED | kept as evidence; support ranges feed `optbot/cin/support.py` |
| `Phase_A_final\encoder.py/tokenizer.py` (two-stream) | GOOD design, code mismatch, no eval | re-implemented slim in `optbot/models/phase_a/`; probes NEW in `probes.py` |
| `Phase_A_final\train_phaseA.py` vs `training_heads.py` | BROKEN (TypeError/NameError mismatch) | superseded; do not run |
| `D:\baseline_model_output\kalman_diagnostics.json` (12% coverage) | BROKEN | DELETED from path; replaced by `optbot/cin/conformal.py` |
| `D:\phaseB\trade_backtest_big_trades_2024*.parquet` (8 trades, no freeze) | PROTOTYPE | superseded by `optbot/backtest/harness.py` |
| `D:\phaseB\toi_test_*` (Matthews counterfactuals) | PROTOTYPE | generalized in `optbot/cin/scenario.py` |
| `Downloads\Phaseb\perfect_center_baseline_preds.py` | FOOT-GUN (label-aware) | not ported; forbidden by contract 4 |
| Phase D0 checkpoints (`D:\run\`) | NEWEST, tangential | parked for v2 (system encoder) |

## Known data bugs the new pipeline must not inherit
1. Cold-start priors: 97.9% zeros + constant SE early-season -> `priors/coldstart.py` carries
   decayed prior-season value; SE floors and WIDENS at low n_eff.
2. SE collapse (89.6% of n_eff 0.1-1 rows have SE==0) -> same fix.
3. Teammate lists first-K-not-top-K -> `contracts` require seconds-desc sort, validator checks.
4. `toi_sec_prior_raw` missing -> perfect windows carry `p_minutes_prior` explicitly; no silent fallbacks (validator).
5. 8.99% NaN xG env rows (PP/PK) -> MVP is 5v5; validator quantifies NaN by strength anyway.
