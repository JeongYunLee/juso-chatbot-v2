FROM python:3.12-slim

# 멀티플랫폼 빌드 환경 설정
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install numpy setuptools wheel cython gunicorn && \
    pip install --no-cache-dir --no-binary=kiwipiepy -r requirements.txt

COPY . .
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]

