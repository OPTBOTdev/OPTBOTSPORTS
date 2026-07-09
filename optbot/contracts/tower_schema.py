"""Environment-tower input schema — the tower's LEGAL DIET, frozen as code.

Classification rule (the scenario-builder litmus test): every input must be
(SYNTH) exactly constructible for a hypothetical destination from pre-t0 data,
(MARG)  servable as a distribution via slot bootstrap, or it is
(BANNED).

Verdicts below encode the July 2026 trace of the legacy builder
(perfect_windows.py / fill_player_windows_xg.py). See README T-audit.
"""

# ---- SYNTH: exact for any hypothetical destination -------------------------
SYNTH_CONTEXT = [
    "period", "period_time_bucket", "score_bucket", "start_regime",
    "lever_zone_start", "fo_loc_enum", "stoppage_class_at_start",
    "after_icing", "ai_OZ_start", "ai_DZ_start",        # traced PRE-safe (icing flags)
    "bench_rights", "long_change", "home_away", "rinkid", "skater_diff",
]
SYNTH_SCHEDULE = ["rest_days_team", "b2b_team"]
SYNTH_TEAM_PRIORS = [
    "team_xgf60_prior_ev", "team_xga60_prior_ev", "team_pace60_ev_prior",
    "opp_team_xgf60_prior_ev", "opp_team_xga60_prior_ev", "opp_team_pace60_ev_prior",
    "team_gsaa_per60_prior_eb", "opp_gsaa_per60_prior_eb",
    "team_goalie_tier", "opp_goalie_tier",
]
SYNTH_PEOPLE = ["with_ids", "with_seconds", "vs_ids", "vs_seconds"]  # the intervention

# ---- MARG: served as slot-bootstrap distributions, never point-known -------
# ENTRY-side timing is legal context (a coach's decision, made before the focal
# exposure begins): captures 'inserted late / mid-chaos / post-icing rescue'
# situational structure. 100% populated in source; 37% of rows enter mid-window.
MARG = [
    "seconds", "duration",                    # exposure (loss weights / offsets)
    "shift_count_in_window", "time_since_last_shift_s", "last_shift_len_s",
    "onice_elapsed_at_window_start", "entered_after_start", "entry_offset_s",
    "stint_duration_max", "is_multistint",    # blended-row flag: trainer downweights
]

# ---- BANNED (with reasons — cite these when someone asks) ------------------
BANNED = {
    "matchup_quality_pct": "LEAKY: full-game opponent TOI percentile x realized "
                           "within-window overlap weights (traced perfect_windows.py:2098)",
    "standing_prior": "BORDERLINE-LEAKY: exact-date branch unverifiable; gamePk "
                      "fallback uses after_pk<=current (perfect_windows.py:1367)",
    "end_event_type": "outcome: how the window ended",
    "season": "era identity undefined at serving time; era enters via team priors",
    "__shot_style__": "all shots_*_cnt / xg_*_sum / plays_* / tsf_* / x_*_sum: "
                      "in-window events = outcomes wearing feature costumes",
    "__focal_id__": "focal player has NO embedding — talent channel only (C2)",
    "__focal_deployment_priors__": "p_prior_oz_share etc. describe OLD-team usage; "
                                   "destination role template supplies deployment",
    # EXIT-side timing is a consequence, not a decision: players leave early
    # BECAUSE of what happened (goal against -> change; trapped -> whistle).
    # Partially outcome-caused => banned. Entry-side stays legal (see MARG).
    "exited_before_end": "exit timing is outcome-contaminated",
    "exit_offset_s": "exit timing is outcome-contaminated",
    "stint_duration_st": "realized within-window stint variance — outcome-adjacent",
}

TOWER_INPUTS = SYNTH_CONTEXT + SYNTH_SCHEDULE + SYNTH_TEAM_PRIORS + SYNTH_PEOPLE + MARG


def assert_legal(feature_list):
    """Call in every trainer: refuse banned or unknown features loudly."""
    banned_hit = [f for f in feature_list if f in BANNED
                  or f.startswith(("shots_", "xg_", "tsf_", "x_", "plays_",
                                   "with_event_", "own_"))]
    if banned_hit:
        raise ValueError(f"BANNED features in tower inputs: {banned_hit}")
    unknown = [f for f in feature_list if f not in TOWER_INPUTS]
    if unknown:
        raise ValueError(f"unclassified features (classify before use): {unknown}")
