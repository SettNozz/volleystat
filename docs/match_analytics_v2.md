# Match analytics v2

The pipeline now keeps three independent kinds of truth:

1. **Observation** — ball, pose, and contact timing.
2. **Context** — match phase, possession, touch number, physical action, and
   volleyball role.
3. **Result** — reception quality and the event-sourced point ledger.

Legacy `event_types` remain unchanged. New clip manifests are schema v2 and
old manifests are upgraded in memory by `manifest_schema.upgrade_manifest()`.

## Point confirmation

The next confirmed server is the winner of the preceding rally. The final
rally stays unresolved until another serve is observed. An explicit service
fault can award the last point immediately. A set boundary never uses the next
set's first server as evidence for the prior point.

`team_a` and `team_b` are court sides, not permanent team identities. Use a
match config to map them to stable IDs and flip that map after teams change
sides:

```powershell
python analyze_video.py --video game.mp4 `
  --match-config configs/match_config.example.json
```

Segments may contain `match_phase`, `set_id`, `team_identity_map`, or a
combination. Use `match_phase: non_game` for warm-up, referee delays, timeouts,
and between-set footage that should not contribute to match statistics.

## Annotation

The Flask event annotator now exposes:

- legacy action class;
- match phase;
- physical action;
- contextual role;
- reception quality.

Press `C` to switch between the isolated contact clip and the complete rally
context from the source video.

## Model

`action_recognition.hierarchical.HierarchicalEventModel` has a long-context
phase head and local heads for contact, court side, physical action,
contextual role, and reception quality. Missing labels are masked by
`hierarchical_loss`; weak FSM context can therefore coexist with human labels
without turning unknown values into negative examples. Score is deliberately
not a model head.
