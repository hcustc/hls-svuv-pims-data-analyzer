FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-input --default-timeout=120 --retries=10 -r requirements.txt

COPY . ./

EXPOSE 8000

CMD ["uvicorn", "bl03u_masstool.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
