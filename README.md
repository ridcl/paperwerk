Useful commands

Run Gemma 4 locally (e.g. for query generation):
```
docker build . -f Dockerfile.serve -t paperwerk-serve:latest

docker run --gpus all  --shm-size=16g -p 8000:8000 paperwerk-serve:latest /venv/bin/vllm serve google/gemma-4-E4B-it     --max-model-len 65536     --enable-auto-tool-choice       --tool-call-parser gemma4     --tensor-parallel-size 2
```