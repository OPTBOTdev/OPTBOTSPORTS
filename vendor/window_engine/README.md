# Vendored Window Engine (certified Jul 2026)

Source: C:\Users\lilli\Downloads\API (single-copy original — this is the versioned backup).
Certification: 99.7% seconds / 0 misattributed goals across 986,920 player-windows (scripts/10).

Pipeline: rankings -> nhl_dump_everything -> build_pbp_with_onice -> perfect_windows
  (driver: nhl_build_perfect_windows_season.py) -> fill_player_windows_xg -> collect_player_windows_train
Season entrypoint: nhl_make_season_final.py --season <S> --windows --skip-existing

REGRESSION GATE (run after ANY windows rebuild):
  1. season_audit_2024_full_tokens.py   (zero-issue baseline: artifacts/audits/season2024_full_token_audit_ALL1271)
  2. audit_goal_crediting.py            (baseline: 3,856 goals, 51 shootout-only hard cases)
  3. batch_audit_50_games.py            (baseline: batch50_v3 all-zero defects)
  4. D:/optbot scripts/09 + 10          (external + crown-jewel certifications)
Known bounded gaps: shootout goals lack on-ice; 2019-20 tracking asterisk.
