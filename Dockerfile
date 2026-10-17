# Hugging Face Spaces (Docker SDK) — free CPU tier
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface ANONYMIZED_TELEMETRY=False \
    PIP_NO_CACHE_DIR=1

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

USER user
# Bake the embedding + reranker models into the image -> faster cold starts after sleep
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user app ./app
COPY --chown=user static ./static
COPY --chown=user data/processed ./data/processed

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
