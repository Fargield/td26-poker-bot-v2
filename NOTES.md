# NOTES

## Done (2026-11-07)
- Cloned repo; 129 + 117 tests pass.
- Added league: `league/strategies.py`, `league/make_bots.py` (generates `league/bots/*.py`), `tools/league.py` (points runner).
- Baseline of current bot (300 deals x 3 seeds, 11 opponents): 31.5/33. Wins everything except `exploiter` (3 chops).
- vs exploiter (500 deals x 6 seeds): default ~break-even (3.0/6 pts). `range_on=0` worse (2.5/6, -0.8 bb); looser bet/raise thresholds worse (2.0/6).
- Backup of strategy files: `backups/2026-11-07_134122/`.

## Decided
- Simple threshold tweaks do not help vs exploiter; leak needs structural change (bluffs/slowplay/checked-range narrowing) if pursued.

## Next
- Make the exploiter stronger/more realistic, then try plan step 2 (balanced bluffs, MDF) and step 5 (fuzz tests, check_build.sh).
- Nothing committed or pushed yet.
