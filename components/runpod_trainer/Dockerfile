FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Upgrade torch to 2.6+ for CVE-2025-32434 compliance, then install deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "torch>=2.6.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler and data
COPY handler.py /app/handler.py
COPY data/ /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV DDI_DATA_PATH=/app/data/drugbank_ddi_complete.jsonl

CMD ["python", "-u", "handler.py"]
