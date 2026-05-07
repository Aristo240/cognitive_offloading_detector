#!/usr/bin/env bash
# Docker-based version. Uses vLLM's official OpenAI-compatible image so we
# skip the entire pip dep-stack (driver/torch/numpy/transformers/vllm
# compatibility was a nightmare with raw pip).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/cognitive_offloading_detector

MODEL_ID="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
PORT=8000
DOCKER_IMG="vllm/vllm-openai:latest"

# Check for Docker; if missing, install
if ! command -v docker >/dev/null 2>&1; then
    echo "[$(date)] installing docker"
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker ubuntu
fi

echo "[$(date)] verifying nvidia-container-toolkit"
if ! sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "[$(date)] installing nvidia-container-toolkit"
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

echo "[$(date)] pulling $DOCKER_IMG (~7 GB)"
sudo docker pull "$DOCKER_IMG"

echo "[$(date)] stopping any prior vllm container"
sudo docker rm -f vllm-server 2>/dev/null || true

# Resolve HF token from .env (already on remote; placed by the manual scp step)
HF_TOKEN_VAL=$(grep '^HF_TOKEN=' .env | cut -d= -f2-)

echo "[$(date)] starting vLLM container with $MODEL_ID"
mkdir -p ~/hf_cache
sudo docker run -d --name vllm-server \
    --gpus all \
    --shm-size=16g \
    -p ${PORT}:8000 \
    -v ~/hf_cache:/root/.cache/huggingface \
    -e HF_TOKEN="$HF_TOKEN_VAL" \
    -e VLLM_DISABLE_USAGE_STATS=1 \
    "$DOCKER_IMG" \
    --model "$MODEL_ID" \
    --served-model-name Llama-3.3-70B-Instruct-FP8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92

echo "[$(date)] waiting for /v1/models (model load + download ~10-15 min)"
for i in $(seq 1 240); do
    if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[$(date)] vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

if ! curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[$(date)] vLLM did not come up in 20 min" >&2
    sudo docker logs vllm-server 2>&1 | tail -60 >&2
    exit 1
fi

echo "[$(date)] available models:"
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool | head -20

echo "[$(date)] running cross_judge"
set -a; source .env; set +a
export VLLM_API_KEY="dummy"
python3 cross_judge.py --source jsonl \
    --data results/cross_judge_ultrachat/conversation_pool.jsonl \
    --judges anthropic:claude-haiku-4-5 gemini:gemini-2.0-flash vllm:Llama-3.3-70B-Instruct-FP8 \
    --out-dir results/cross_judge_ultrachat/ \
    --sleep 0.1

echo "[$(date)] stopping vllm container"
sudo docker stop vllm-server || true

echo "[$(date)] DONE"
echo SCRIPT_COMPLETE
