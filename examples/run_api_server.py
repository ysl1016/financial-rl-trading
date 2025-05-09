import argparse
import os
import sys
import logging
from pathlib import Path

# 상위 디렉토리를 import path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import run_app

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-R1 트레이딩 API 서버 실행")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="모델 경로 (예: models/default/latest)")
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="서버 호스트 (기본값: 0.0.0.0)")
    
    parser.add_argument("--port", type=int, default=8000,
                        help="서버 포트 (기본값: 8000)")
    
    parser.add_argument("--optimization", type=str, default="medium",
                        choices=["none", "low", "medium", "high"],
                        help="모델 최적화 수준 (기본값: medium)")
    
    parser.add_argument("--reload", action="store_true",
                        help="코드 변경 시 자동 재시작 (개발용)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모델 경로 확인
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"모델 경로를 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    logger.info(f"API 서버 시작 중... (모델: {model_path}, 최적화 수준: {args.optimization})")
    
    # API 서버 실행
    run_app(
        model_path=model_path,
        host=args.host,
        port=args.port,
        optimization_level=args.optimization,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
