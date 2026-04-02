#!/bin/bash
vllm serve /data/models/kvp10k-qwen3vl-4b/ \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes