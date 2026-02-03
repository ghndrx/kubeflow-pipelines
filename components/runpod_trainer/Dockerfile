FROM runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install dependencies (torch 2.8 in base image satisfies CVE-2025-32434)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler and data
COPY handler.py /app/handler.py
COPY data/ /app/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV DDI_DATA_PATH=/app/data/drugbank_ddi_complete.jsonl

CMD ["python", "-u", "handler.py"]
