FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (ffmpeg is used for audio preprocessing)
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Install Python deps (cached layer)
COPY requirements.txt ./requirements.txt
RUN uv venv /opt/venv \
  && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

# Copy application code
COPY . .

# Runs the CLI pipeline; pass args like: --audio /path/to/file
CMD ["python", "main.py"]
