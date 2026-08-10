# Remote Ollama VLM on a MacBook

The Windows pipeline samples frames from each local event clip and sends only
JPEG images plus a prompt to Ollama on the Mac. Video files and manifests stay
on Windows. Output is written to a new, resumable manifest.

## Recommended model

For an M1 Pro with 32 GB unified memory, start with
`qwen3-vl:8b-instruct` for bulk pre-labeling. It is small enough to stay
responsive while leaving memory for macOS. Benchmark `qwen3-vl:30b-a3b-instruct`
later on a labeled sample if the accuracy gain is worth the lower throughput.
Qwen3-VL requires Ollama 0.12.7 or newer, so update Ollama if `pull` rejects
the model.

## MacBook setup

1. Install and open Ollama from <https://ollama.com/download/mac>.
2. Clone/copy this repository to the Mac, open Terminal in it, and run:

   ```bash
   chmod +x scripts/mac/setup_ollama_vlm.sh
   ./scripts/mac/setup_ollama_vlm.sh
   ```

   The script exposes Ollama on port 11434, keeps the model warm for 30 minutes,
   restarts Ollama, downloads `qwen3-vl:8b-instruct`, and prints the Mac IP.

Manual equivalent:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
launchctl setenv OLLAMA_KEEP_ALIVE "30m"
# Quit and reopen the Ollama app after launchctl.
ollama pull qwen3-vl:8b-instruct
ipconfig getifaddr en0
```

Use this only on a trusted LAN: the native Ollama API has no authentication.
Allow incoming connections for Ollama if the macOS firewall asks.

## Windows configuration and smoke test

In PowerShell, replace the IP with the value printed on the Mac:

```powershell
$env:OLLAMA_BASE_URL = "http://192.168.1.25:11434"
$env:OLLAMA_VLM_MODEL = "qwen3-vl:8b-instruct"
python -m action_recognition.vlm.prelabel_events --check
```

To persist the values for future terminals:

```powershell
setx OLLAMA_BASE_URL "http://192.168.1.25:11434"
setx OLLAMA_VLM_MODEL "qwen3-vl:8b-instruct"
```

Open a new terminal after `setx`.

## Validate before a long run

First score the VLM on existing human labels:

```powershell
python scripts/evaluation/vlm_clip_probe.py `
  --ollama-url $env:OLLAMA_BASE_URL `
  --model $env:OLLAMA_VLM_MODEL `
  --count 50
```

Then pre-label ten new clips:

```powershell
python -m action_recognition.vlm.prelabel_events `
  --manifest data/event_clips_part2/manifest.json `
  --limit 10
```

If the result looks useful, resume without `--limit`:

```powershell
python -m action_recognition.vlm.prelabel_events `
  --manifest data/event_clips_part2/manifest.json
```

The command writes `manifest_vlm_predicted.json` after every clip. Re-running
the same command skips completed predictions, rule labels, and human labels.

Review the suggestions:

```powershell
python -m action_recognition.annotator.event_clip_annotator `
  --manifest data/event_clips_part2/manifest_vlm_predicted.json
```

Open <http://localhost:5056>. VLM confidence and reason appear beside the clip.

## Ball-localization experiment

Event classification is the primary use. VLM ball localization should remain an
evaluation-only fallback until it beats the detector on held-out ground truth:

```powershell
python scripts/evaluation/vlm_ball_probe.py `
  --gt results/ball_eval/<file>_gt.json `
  --ollama-url $env:OLLAMA_BASE_URL `
  --model $env:OLLAMA_VLM_MODEL
```

## Troubleshooting

- `Cannot reach Ollama`: verify both machines are on the same LAN, use
  `curl http://<mac-ip>:11434/api/version` from Windows, and check the Mac
  firewall.
- `model ... is not installed`: run `ollama pull qwen3-vl:8b-instruct` on Mac.
- Requests are slow: keep six frames at width 768; try `--frames 5 --width 640`.
- Wrong labels: run the 50-clip probe before changing prompts or trying the 30B
  model. Never merge VLM output into training data without human review.
