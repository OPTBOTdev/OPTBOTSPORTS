# How Player IDs and Opponent IDs are Kept in Windows and Shots

## Overview

In `perfect_windows.py`, player IDs and opponent IDs are tracked in windows and filtered using thresholds. Here's how the system works:

## 1. Player IDs in Windows

### Where They're Stored

Player IDs are stored in several fields in each window row:

- **`teammates_onice_ids_start`**: List of teammate player IDs at window start
- **`teammates_onice_ids_w`**: List of teammate player IDs weighted by time-on-ice during the window
- **`opponents_onice_ids_start`**: List of opponent player IDs at window start  
- **`opponents_onice_ids_w`**: List of opponent player IDs weighted by time-on-ice during the window
- **`playerId`**: The focal player ID for this window row (each player gets their own row)

### How IDs Are Tracked

1. **On-ice tracking**: For each second in a window, the code tracks which players are on ice:
   ```python
   for s in range(start_s, end_s):
       on = onice_map.get(s, ())
       if p not in on: continue
       for q in on:
           if q == p: continue
           overlap_with[q] += 1  # teammate overlap
       for k in opp_map.get(s, ()):
           overlap_vs[k] += 1  # opponent overlap
   ```

2. **Overlap counting**: The system counts how many seconds each teammate/opponent was on ice with the focal player.

## 2. Filtering Logic: When IDs Get "Cut"

The key filtering happens in the `build_weighted_list()` function (lines 2046-2080). IDs are **kept** or **cut** based on two criteria:

### Thresholds

```python
SEC_FLOOR_BASE   = 5        # Minimum seconds for normal windows
RAW_SHARE_MIN    = 0.0225   # Minimum 2.25% time share for normal windows
SHORT_WIN_SEC    = 10       # Threshold for "short" windows
SHORT_SHARE_MIN  = 0.40     # Minimum 40% time share for short windows
SHORT_SEC_FRAC   = 0.50     # 50% of window duration for short windows
TOPK_FALLBACK    = 1        # Keep top-1 if nothing else qualifies
```

### Filtering Rules

**For normal windows (≥10 seconds):**
- **KEPT** if: `overlap_seconds >= 5` AND `raw_share >= 0.0225` (2.25%)
- **CUT** if: Either condition fails

**For short windows (<10 seconds):**
- **KEPT** if: `overlap_seconds >= max(2, 50% of window)` AND `raw_share >= 0.40` (40%)
- **CUT** if: Either condition fails
- **FALLBACK**: If nothing qualifies but window ≥2 seconds, keep the top player by overlap (if ≥2 seconds)

### Example

```python
# Normal window: 60 seconds
# Player A was on ice for 3 seconds (5% share) → CUT (needs ≥5 seconds)
# Player B was on ice for 6 seconds (10% share) → KEPT (≥5 sec AND ≥2.25%)
# Player C was on ice for 1 second (1.7% share) → CUT (needs ≥5 seconds)

# Short window: 8 seconds  
# Player A was on ice for 4 seconds (50% share) → KEPT (≥4 sec AND ≥40%)
# Player B was on ice for 2 seconds (25% share) → CUT (needs ≥40% share)
```

## 3. Window-Level Filtering

### Team Windows Get Pruned

Some team windows are removed entirely (lines 2508-2514):

```python
kept_team_rows = []
for r in team_rows:
    is_ea = (r["strength_team"] in ("EA","EN_for"))
    if is_ea and (r["duration"] < EA_PRUNE_MIN) and window_zero_events(...):
        continue  # CUT: Empty attacker windows <6 seconds
    kept_team_rows.append(r)
```

**CUT**: Empty attacker (EA/EN_for) windows <6 seconds with zero events

### Player Rows Follow Team Windows

Player rows are only kept if their corresponding team window exists (lines 2516-2523):

```python
keep_key = {(r["window_id"], r["team_side"]): True for r in team_rows}
for r in player_rows:
    key = (r["window_id"], r["team_side"])
    if key not in keep_key:
        continue  # CUT: No matching team window
    kept_player_rows.append(r)
```

**CUT**: Player rows whose team window was pruned

## 4. Shots and Events

### Shot Attribution

For shots (lines 640-685), player IDs are extracted from event details:

```python
shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId")
```

Shots are credited to players who were **on ice** at the time of the shot:
- **AF/AA** (attempts for/against): Based on which side the shooter was on
- **SF/SA** (shots for/against): Only on-target shots
- **xGF/xGA**: Expected goals

### Block Attribution

For blocked shots (lines 656-682), the blocker ID is extracted:

```python
blk_pid = (
    det.get("blockingPlayerId")
    or det.get("blockedByPlayerId")
    or det.get("blockerPlayerId")
)
```

**BF/BA** (blocks for/against) are credited based on which side the blocker was on.

## 5. Summary: When IDs Get Cut

### IDs Get CUT When:

1. **Low time-on-ice**: Player was on ice <5 seconds in normal windows, or <50% of short windows
2. **Low share**: Player's time share <2.25% in normal windows, or <40% in short windows  
3. **Window pruned**: The entire team window was removed (empty EA windows <6 sec)
4. **No overlap**: Player never overlapped with the focal player during the window

### IDs Are KEPT When:

1. **Sufficient time**: ≥5 seconds overlap in normal windows, or ≥50% of short windows
2. **Sufficient share**: ≥2.25% share in normal windows, or ≥40% in short windows
3. **Fallback**: Top player by overlap in short windows (if ≥2 seconds overlap)

## 6. Output Fields

The final output includes these ID lists (lines 2328-2339):

- **`teammates_onice_ids_start`**: Sorted list of teammate IDs at start
- **`opponents_onice_ids_start`**: Sorted list of opponent IDs at start
- **`teammates_onice_ids_w`**: Filtered list of teammate IDs (weighted)
- **`opponents_onice_ids_w`**: Filtered list of opponent IDs (weighted)
- **`teammates_onice_w`**: Normalized weights (sum to 1.0)
- **`opponents_onice_w`**: Normalized weights (sum to 1.0)
- **`teammates_onice_sec_w`**: Seconds on ice for each teammate
- **`opponents_onice_sec_w`**: Seconds on ice for each opponent

These lists are written as pipe-separated strings in CSV output (line 2541).

