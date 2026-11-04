FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
WORKDIR /app
COPY models/Qwen3-14B /models/Qwen3-14B
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY ingest.py llmviews.py predict.py evaluation.py predict.sh model_manifest.json /app/
RUN chmod +x /app/predict.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
ENTRYPOINT ["/app/predict.sh"]
