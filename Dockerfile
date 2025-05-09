FROM python:3.10-slim

LABEL maintainer="DeepSeek-R1 Trading Team"
LABEL version="1.0.0"
LABEL description="DeepSeek-R1 기반 금융거래 강화학습 모델 서비스"

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# API 서버 설정을 위한 환경변수
ENV MODEL_PATH="models/default/latest"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV LOG_LEVEL="INFO"
ENV API_KEY="default_development_key"
ENV OPTIMIZATION_LEVEL="medium"

# 포트 노출
EXPOSE 8000

# 시작 명령어
CMD ["sh", "-c", "python -m src.api.app run --model-path $MODEL_PATH --host $HOST --port $PORT --optimization $OPTIMIZATION_LEVEL"]
