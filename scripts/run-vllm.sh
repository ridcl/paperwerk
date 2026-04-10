#!/bin/bash
vllm serve google/gemma-4-E4B-it \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --tensor-parallel-size 2

# vllm serve /data/models/kvp10k-qwen3vl-4b/ \
#     --max-model-len 32768 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --tensor-parallel-size 2