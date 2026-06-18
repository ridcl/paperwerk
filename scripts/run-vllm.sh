#!/bin/bash
export VLLM_PLATFORM=cuda

# vllm serve google/gemma-4-E4B-it \
#     --max-model-len 32768 \
#     --enable-auto-tool-choice \
#     --tool-call-parser gemma4 \
#     --tensor-parallel-size 2

# vllm serve /data/models/vqa-20260529-qwen3vl-4b/ \
vllm serve ridcl/paperwerk-vqa \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 2

# vllm serve /data/models/kvp10k-qwen3vl-4b-retrained/ \
#     --max-model-len 32768 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --tensor-parallel-size 2