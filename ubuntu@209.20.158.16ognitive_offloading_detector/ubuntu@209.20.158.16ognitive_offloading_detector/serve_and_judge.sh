#!/usr/bin/env bash
# Run on a Lambda Cloud GPU instance via lambda_run.py.
# Sets up vLLM serving Llama-3.3-70B-Instruct (FP8 pre-quantized to fit 80GB)
# and runs cross_judge.py with three out-of-GPT-family judges:
#   Anthropic Haiku, Gemini Flash, and the local Llama-70B vLLM endpoint.
#
# After this script finishes, results land in
#   results/cross_judge_ultrachat_4judge/
# and lambda_run.py rsyncs them back.
set -euo pipefail

cd ~/cognitive_offloading_detector

MODEL_ID="neuralmagic/Llama-3.3-70B-Instruct-FP8-dynamic"
LOCAL_MODEL_DIR=~/models/llama-3.3-70b-fp8
PORT=8000

echo "=== [1/5] Installing GPU-side deps (vllm, hf-transfer) ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet vllm hf-transfer huggingface_hub

echo "=== [2/5] Downloading $MODEL_ID (FP8 ~70 GB; expect 5-15 min) ==="
mkdir -p "$LOCAL_MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_ID" \
    --local-dir "$LOCAL_MODEL_DIR" \
    --local-dir-use-symlinks False \
    --quiet

echo "=== [3/5] Starting vLLM in background ==="
mkdir -p logs
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$LOCAL_MODEL_DIR" \
    --served-model-name "Llama-3.3-70B-Instruct-FP8" \
    --port $PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    > logs/vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID  (log: logs/vllm.log)"

echo "=== [4/5] Waiting for vLLM /v1/models to respond (up to 10 min) ==="
for i in $(seq 1 120); do
    if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "  vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

if ! curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM did not become ready in 10 min."
    echo "--- last 60 lines of vllm.log ---"
    tail -60 logs/vllm.log || true
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

echo "=== [5/5] Running cross_judge - vllm judge only on existing pool ==="
echo "  (Anthropic + Gemini grades already exist in results/cross_judge_ultrachat/"
echo "   and will be skipped via resumability; only vllm/Llama runs.)"
export VLLM_API_KEY="dummy-vllm-does-not-check"
python3 cross_judge.py --source jsonl \
    --data results/cross_judge_ultrachat/conversation_pool.jsonl \
    --judges anthropic:claude-haiku-4-5 \
             gemini:gemini-2.0-flash \
             vllm:Llama-3.3-70B-Instruct-FP8 \
    --out-dir results/cross_judge_ultrachat/ \
    --sleep 0.2

echo "=== Stopping vLLM ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

echo "=== Done. Results in results/cross_judge_ultrachat_3judge_with_llama/ ==="
