FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (ffmpeg is used for audio preprocessing)
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Install Python deps via uv sync (cached layer)
# If uv.lock exists, uv will use it; otherwise it will resolve from pyproject.toml.
COPY pyproject.toml uv.lock* ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev --no-install-project

ENV PATH="/opt/venv/bin:${PATH}"

# Copy application code
COPY . .

# Run the FastAPI server
EXPOSE 8000
ENTRYPOINT ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
