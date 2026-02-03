FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler and data
COPY handler.py /app/handler.py
COPY data/ /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV DDI_DATA_PATH=/app/data/drugbank_ddi_complete.jsonl

CMD ["python", "-u", "handler.py"]
