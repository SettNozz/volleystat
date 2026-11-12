# Match setup, tactics and developer review

CourtSense has separate workspaces for watching a match, inspecting player
statistics, supplying match context and exploring tactics. Open a saved result
from the library, then use **Match setup** and **Tactics** in its navigation.
The pages use the original video and saved analysis; no model inference runs
when context or reviews change.

## Match setup

Open `/match-setup/<job_id>` before interpreting match statistics:

1. Seek through the video and select interval boundaries with the sliders,
   frame inputs or **Use current as start/end**. **Play selection** checks the
   selected range.
2. Mark warm-up, breaks and other excluded footage as **Warm-up / break /
   non-game**. For match play, supply a set ID and the teams occupying the far
   and near sides. Add or update the interval in the draft.
3. Set team display names, default side assignments and starting points. Keep
   each team's stable ID when it changes court sides; use an interval mapping
   to record the switch. Split a set into multiple intervals with the same set
   ID when its side mapping changes.
4. Assign saved roster IDs to teams and confirm the entries you recognize.
5. Choose **Save match setup** to apply the complete draft. **Reload saved
   setup** discards unsaved changes and loads the current saved version.

Intervals include both endpoint frames and must not overlap. For example,
`0–899` and `900–1799` are adjacent intervals. Unmarked footage keeps its
existing context; it is not automatically excluded. An attempt's context is
taken at its serve/start frame, while individual contacts use their own frame.
Set boundaries and non-game periods are manual input, not phase predictions.

The two court sides are `team_a` (far) and `team_b` (near). The
`team_identity_map` maps them to stable team IDs, and a segment can override
that map. Display names can change without changing those IDs. Starting
points are applied to the first represented set and the overall reviewed
score; later sets begin from zero.

Roster confirmation stores a match-specific team assignment for an existing
saved player ID. It does not create a new tracked player or verify tracking,
contact authors, event detection or event types. Player names, jerseys and
positions edited in **Player statistics** still live in browser
`localStorage`; they are separate from saved roster confirmation.

**Download match-config JSON** downloads the current draft, including unsaved
changes. It does not save that draft to the viewer. The CLI's
`--match-config <json>` supports `team_identity_map` and `segments` for match
context. The download also carries viewer metadata such as team display names,
starting score, roster confirmation and rotations; these fields do not
automatically acquire CLI behavior. See [usage](usage.md#analysis-options) and
`configs/match_config.example.json` for the pipeline configuration contract.

## Tactics

Open `/tactics/<job_id>` to inspect setter choices and rotation outcomes.
Set-to-attack evidence pairs adjacent set/pass and attack contacts inside one
unambiguous saved rally. Filters cover set, stable team, setter, attack lane,
reception quality and manually labelled rotation. A distribution count opens
its matching clips; **Watch full rally** and **Open attack in match viewer**
provide wider context. Shares use the currently filtered pairs for each
team/setter group.

Attack lanes are approximate camera-left, center and camera-right thirds of
the calibrated net width, measured at the attack ball's image X coordinate.
They are not volleyball position numbers or measured 3D attack locations.
Geometry supplies no team-side decision here. Setters come from the saved or
reviewed contact author; an unidentified author or conflicting confirmed
roster team remains explicit. Reception quality requires compatible saved
serve-reception evidence; otherwise it is unknown.

A pair needs a continuous saved ball path between the two contacts. Missing
or off-camera ball positions, a track restart, ambiguous rally membership,
conflicting sides or a match-context boundary exclude unsupported pairs.
Accepted paths retain observed/derived provenance. Coverage reports accepted
candidates, rejection reasons and unknown fields; it is not an accuracy score.

Under **Confirm rotation over a time range**, choose a stable team, R1–R6 and
the confirmed start/end times. Labels persist with the match setup; remove a
label to leave that interval unlabelled. Rotations never advance automatically.
Use one numbering convention throughout the recording. Same-team rotation
intervals must not overlap; opposite teams can have labels over the same time.

Rotation outcomes use source-scoped point reconciliation:

- **Side-out:** points won while receiving / classified receiving points.
- **Break point:** points won while serving / classified serving points.
- Unknown winners stay outside those denominators and appear in separate
  counts. Unknown serving teams have a separate service-role count. Conflicting
  or overlapping score windows leave the outcome unresolved.

Set, team and rotation filters affect rotation outcomes. Setter, lane and
reception-quality filters affect set-to-attack pairs only. Rally evidence
shows the point source and whether the winner was human-confirmed. A confirmed
winner does not confirm an attack kill or contact author.

## Developer review

The **Developer review** tab and developer tools are hidden by default. Add
`?developer=1` to a workspace URL to reveal them; this preference persists in
the browser's `courtsense-developer` localStorage entry. `?developer=0` hides
them again. `/developer-review/<job_id>` opens the workspace directly and also
enables the preference. This is a navigation preference, not authentication or
an access-control boundary.

The queue prioritizes source conflicts, unknown authors, low event confidence,
same-frame contacts and unresolved point evidence. Priority expresses review
impact, not a calibrated probability. Filter by issue/status, play a short
clip or step one frame at a time, then:

- Correct a contact's frame, type, court side or author; detection and type
  confirmation have separate checkboxes.
- Use **Not a contact** to exclude a false event, or **Add missed contact at
  current time** to insert one.
- Review a point winner independently of contact evidence.
- **Dismiss queue item** clears a task without confirming a label.
- **Reset to saved proposal** reverses that contact's workspace corrections.
  Resetting an added contact excludes it from the reviewed view. History stays
  available.

Timing, type, side, author and detection have separate review provenance.
Changing an author does not verify detection or type. Moving a contact clears
its old frame-specific attribution unless an author is explicitly supplied for
the new frame. Point-winner confirmation does not verify any contact axis or
resolve an ambiguous contact/rally join.

Frame-specific crop candidates and proposed contact times are cleared whenever
timing changes, including when a new author is explicitly confirmed. The
author-only **Reset to proposal** in Player contacts restores the original
automatic attribution while preserving other event-review axes. A rejected
serve cannot keep awarding an automatic point through its old confirmation;
an independently confirmed human point winner remains separate evidence.

## Persistence and shared projections

`src/viewer_app/match_workspace.py` reads the saved stats/annotations pair,
existing author/point reviews and workspace history into one current
projection. Context and contact edits append to
`<stem>_workspace_reviews.jsonl` beside the results, using schema
`match_workspace_v1`, timestamps and SHA256 values for both source files.
Existing `<stem>_player_contact_reviews.jsonl` and
`<stem>_point_reviews.jsonl` records are still consumed. Once workspace history
exists, further contact-author corrections use that workspace history.

Writes check the source pair and a revision covering all three review logs.
Viewer writes are serialized. A changed source file or intervening review
rejects a stale save with HTTP 409; reload and inspect the latest view before
retrying. The append log preserves older saved context and edits rather than
replacing them. Event indexes remain stable; removal flags an event instead of
renumbering the original array.

Reloaded viewer contacts, player statistics, highlights, Excel and tactics use
the reviewed projection. Non-game exclusions and stable team/set context feed
the derived views. With saved match context, the viewer displays a reviewed
score projection using current point evidence and starting points. Original
FSM score/heatmap evidence remains separate. Saving context or review does not
rewrite the analysis JSON, human gold annotations, training datasets or model
outputs, and does not retrain or rerun models.

The viewer's automatic heatmap refresh is read-only once any source-bound
review log exists. It computes from active reviewed contacts for display;
opening the match viewer therefore does not invalidate saved review hashes.

| Page | API |
|---|---|
| `/match-setup/<job_id>` | `GET/POST /api/match_context/<job_id>` |
| `/developer-review/<job_id>` | `GET/POST /api/developer_review/<job_id>` |
| `/tactics/<job_id>` | `GET /api/tactics/<job_id>` |

Setup POST requests carry `source`, `revision` and `config`; developer POST
requests carry `source`, `revision` and `edit`. Point-winner reviews use
`POST /api/statistics/<job_id>`. Read the current response before writing; do
not synthesize fingerprints or revision tokens. See
[statistics evidence](statistics_evidence.md) for counting and point-review
rules.
