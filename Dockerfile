FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.lock

COPY . ./
RUN python -m pip install --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "bl03u_masstool.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
