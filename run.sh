#!/bin/sh
BODY=''':'
# ===================== BASH BODY  =====================
set -euo pipefail

REPO_URL="https://github.com/take2make/gpu-sharing-sla.git"

MODEL_A="unsloth/Llama-3.2-1B-Instruct"
MODEL_B="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_C="BAAI/bge-small-en-v1.5"

PORT_A=8000
PORT_B=8001
PORT_C=8002

CONC_A=40
CONC_B=14
CONC_C=12

GPU_A=0.41
GPU_B=0.31
GPU_C=0.09

echo "=== [1/5] git clone ==="
git clone --depth 1 "$REPO_URL" /content/AE_HW
cd /content/AE_HW
mkdir -p out

echo "=== [2/5] pip install ==="
pip install --no-input uv
uv pip install --system -r requirements.txt

echo "=== run MPS ==="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
nvidia-cuda-mps-control -d 2>&1 && echo "MPS started" || echo "MPS unavailable in this container"

echo "=== [3/5] launch A ==="
vllm serve "$MODEL_A" --port $PORT_A --attention-backend TRITON_ATTN \
  --gpu-memory-utilization $GPU_A --max-model-len 1024 > out/serverA.log 2>&1 &
for i in $(seq 1 180); do curl -sf http://localhost:$PORT_A/health >/dev/null && echo "A ready" && break; sleep 5; done
nvidia-smi --query-gpu=memory.used --format=csv

echo "=== [3/5] launch B ==="
vllm serve "$MODEL_B" --port $PORT_B--attention-backend TRITON_ATTN \
  --gpu-memory-utilization $GPU_B --max-model-len 1024 > out/serverA.log 2>&1 &
for i in $(seq 1 180); do curl -sf http://localhost:$PORT_B/health >/dev/null && echo "B ready" && break; sleep 5; done
nvidia-smi --query-gpu=memory.used --format=csv

echo "=== [3/5] launch C ==="
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=8 \
vllm serve "$MODEL_C" --port $PORT_C --attention-backend TRITON_ATTN\
  --gpu-memory-utilization $GPU_C > out/serverA.log 2>&1 &
for i in $(seq 1 180); do curl -sf http://localhost:$PORT_C/health >/dev/null && echo "C ready" && break; sleep 5; done
nvidia-smi --query-gpu=memory.used --format=csv

sleep 10

echo "=== [4/5] bench A alone + co-residency ==="
# we need to add extra time since it takes time to run bench
{ timeout 50 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1 || true
  nvidia-smi
} > out/nvidia_smi.txt &

vllm bench serve \
  --model "$MODEL_A" \
  --base-url http://localhost:$PORT_A \
  --dataset-name random \
  --random-input-len 256 --random-output-len 128 \
  --random-range-ratio 0 \
  --ignore-eos \
  --seed 1234 \
  --num-prompts 200 --max-concurrency $CONC_A \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --num-warmups 50 \
  --save-result --result-filename out/A_alone.json || echo "A_alone bench FAILED"
sleep 10

echo "=== bench B alone ==="
vllm bench serve --model "$MODEL_B" --base-url http://localhost:$PORT_B \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --random-range-ratio 0 --ignore-eos --seed 1234 \
  --num-prompts 200 --max-concurrency $CONC_B \
  --save-result --result-filename out/B_alone.json || echo "B_alone bench FAILED"
sleep 10

echo "=== bench C alone ==="
vllm bench serve --backend openai-embeddings \
  --model "$MODEL_C" \
  --base-url http://localhost:$PORT_C \
  --endpoint /v1/embeddings \
  --dataset-name random --random-input-len 256 \
  --num-prompts 1000 --max-concurrency $CONC_C \
  --save-result --result-filename out/C_alone.json || echo "C_alone bench FAILED"
sleep 10

echo "=== start B + C load ==="
vllm bench serve --model "$MODEL_B" --base-url http://localhost:$PORT_B \
  --dataset-name random --random-input-len 512 --random-output-len 256 \
  --random-range-ratio 0 --ignore-eos --seed 1234 \
  --num-prompts 300 --max-concurrency $CONC_B \
  --save-result --result-filename out/B_contended.json &
vllm bench serve --backend openai-embeddings --model "$MODEL_C" --base-url http://localhost:$PORT_C \
  --endpoint /v1/embeddings --dataset-name random --random-input-len 256 \
  --num-prompts 15000 --max-concurrency $CONC_C \
  --save-result --result-filename out/C_contended.json &

echo "=== wait for GPU saturation ==="
for i in $(seq 1 300); do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  echo "util=${UTIL}% (poll $i)"
  [ "$UTIL" -ge 80 ] && echo "saturated" && break
  sleep 5
done

echo "=== bench A under contention ==="
vllm bench serve --model "$MODEL_A" --base-url http://localhost:$PORT_A \
  --dataset-name random --random-input-len 256 --random-output-len 128 \
  --random-range-ratio 0 --ignore-eos --seed 1234 \
  --num-prompts 200 --max-concurrency $CONC_A \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --num-warmups 8 --result-filename out/A_contended.json

echo "=== wait for GPU to drain ==="
for i in $(seq 1 300); do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  echo "util=${UTIL}% (drain poll $i)"
  [ "$UTIL" -le 5 ] && echo "idle" && break
  sleep 5
done

echo "=== [5/5] pack to tgz ==="
tar czf out.tgz out
echo "===OUT_TGZ_BEGIN==="
base64 -w0 out.tgz
echo
echo "===OUT_TGZ_END==="

echo "=== [6/6] stop all servers ==="
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "vllm bench" 2>/dev/null || true
pkill -f "nvidia-smi" 2>/dev/null || true

exit 0
'''
# ===================== PYTHON PATH =====================
import subprocess, pathlib

body = BODY.split("\n", 1)[1]          # drop the leading ":'" line
path = pathlib.Path("/content/_run_body.sh")
path.write_text(body)

p = subprocess.Popen(["bash", str(path)], cwd="/content",
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
raise SystemExit(p.wait())