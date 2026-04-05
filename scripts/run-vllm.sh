#!/bin/bash
vllm serve /data/models/kvp10k-qwen3vl-4b/ \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 2