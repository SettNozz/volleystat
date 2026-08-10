#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_VLM_MODEL:-qwen3-vl:8b-instruct}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS."
  exit 1
fi

if command -v ollama >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama)"
elif [[ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]]; then
  OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
else
  echo "Ollama is not installed."
  echo "Install it from https://ollama.com/download/mac, open it once, then rerun this script."
  exit 1
fi

launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
launchctl setenv OLLAMA_KEEP_ALIVE "30m"

echo "Configured Ollama to listen on the trusted LAN."
echo "Restarting the Ollama app so the environment takes effect..."
osascript -e 'quit app "Ollama"' >/dev/null 2>&1 || true
sleep 2
open -a Ollama

echo "Waiting for Ollama..."
for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama did not start. Open the Ollama app manually and rerun this script."
  exit 1
fi

"$OLLAMA_BIN" pull "$MODEL"

IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
echo
echo "Ready: $MODEL"
if [[ -n "$IP" ]]; then
  echo "Windows endpoint: http://${IP}:11434"
  echo "Test from Windows PowerShell:"
  echo "  \$env:OLLAMA_BASE_URL='http://${IP}:11434'"
  echo "  python -m action_recognition.vlm.prelabel_events --check"
else
  echo "Could not determine the Mac IP. Find it in System Settings > Wi-Fi > Details."
fi
