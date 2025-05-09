import argparse
import os
import sys
import logging
import time
from pathlib import Path

# 상위 디렉토리를 import path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.streaming import DataStream, StreamProcessor, StreamingConfig
from src.deployment.model_packaging import ModelPackager
from src.deployment.model_optimization import ModelOptimizer
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.anomaly_detection import AnomalyDetector
from src.monitoring.alerting import AlertManager, AlertConfig, AlertLevel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("streaming_server.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-R1 트레이딩 스트리밍 서버 실행")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="모델 경로 (예: models/default/latest)")
    
    parser.add_argument("--data-source", type=str, default="binance",
                        choices=["binance", "alpaca"],
                        help="데이터 소스 (기본값: binance)")
    
    parser.add_argument("--symbols", type=str, required=True,
                        help="트레이딩할 심볼 목록 (쉼표로 구분)")
    
    parser.add_argument("--interval", type=str, default="1m",
                        help="데이터 수집 간격 (기본값: 1m)")
    
    parser.add_argument("--api-key", type=str, default=None,
                        help="데이터 소스 API 키")
    
    parser.add_argument("--api-secret", type=str, default=None,
                        help="데이터 소스 API 시크릿")
    
    parser.add_argument("--use-testnet", action="store_true",
                        help="테스트넷 사용 여부")
    
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="로그 디렉토리 (기본값: logs)")
    
    parser.add_argument("--optimization", type=str, default="medium",
                        choices=["none", "low", "medium", "high"],
                        help="모델 최적화 수준 (기본값: medium)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모델 경로 확인
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"모델 경로를 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    # 심볼 파싱
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # 로그 디렉토리 생성
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 스트리밍 설정
    config = StreamingConfig(
        symbols=symbols,
        interval=args.interval,
        lookback_periods=100,
        buffer_size=1000,
        data_source=args.data_source,
        api_key=args.api_key,
        api_secret=args.api_secret,
        use_testnet=args.use_testnet,
        log_level="INFO"
    )
    
    logger.info(f"스트리밍 서버 초기화 중... (심볼: {symbols}, 간격: {args.interval})")
    
    # 모델 로드
    logger.info(f"모델 로드 중: {model_path}")
    model_packager = ModelPackager()
    
    try:
        # 모델 이름과 버전 추출
        path_parts = Path(model_path).parts
        if len(path_parts) >= 2:
            model_name = path_parts[-2]
            version = path_parts[-1]
        else:
            model_name = "unknown"
            version = "latest"
        
        # 모델 로드
        model, env_config, metadata = model_packager.load_model(model_name, version)
        
        # 최적화 적용
        if args.optimization != "none":
            logger.info(f"모델 최적화 적용 중 (수준: {args.optimization})...")
            model_optimizer = ModelOptimizer()
            model = model_optimizer.optimize_for_inference(model, args.optimization)
        
        logger.info(f"모델 로드 완료: {model_name}/{version}")
        
        # 데이터 스트림 초기화 및 시작
        logger.info("데이터 스트림 초기화 중...")
        data_stream = DataStream(config)
        
        # 스트림 처리기 초기화
        logger.info("스트림 처리기 초기화 중...")
        stream_processor = StreamProcessor(model, data_stream)
        
        # 성능 추적기 초기화
        performance_tracker = PerformanceTracker(
            asset_name="-".join(symbols),
            initial_capital=100000.0,
            log_dir=str(log_dir / "performance")
        )
        
        # 스트림 시작
        logger.info("데이터 스트림 시작...")
        data_stream.start()
        
        # 처리기 시작
        logger.info("스트림 처리기 시작...")
        stream_processor.start()
        
        # 콜백 함수 정의
        def on_prediction(symbol, prediction):
            logger.info(f"예측 결과: {symbol} - {prediction['action']} (확신도: {prediction['confidence']:.4f})")
            
            # 성능 추적기에 예측 추가
            performance_tracker.add_prediction(
                timestamp=prediction["timestamp"],
                symbol=symbol,
                predicted_action=prediction["action"],
                confidence=prediction["confidence"]
            )
        
        # 스트림 처리기의 예측 결과를 성능 추적기에 연결
        def check_predictions():
            while True:
                for symbol in symbols:
                    prediction = stream_processor.get_latest_prediction(symbol)
                    if prediction:
                        on_prediction(symbol, prediction)
                
                time.sleep(10)  # 10초마다 확인
        
        # 메인 루프
        logger.info("스트리밍 서버 실행 중...")
        try:
            check_predictions()
        except KeyboardInterrupt:
            logger.info("사용자 중단. 종료 중...")
        finally:
            # 정리
            stream_processor.stop()
            data_stream.stop()
            
            # 성능 지표 저장
            performance_tracker.update_metrics()
            performance_tracker.save_metrics()
            
            logger.info("스트리밍 서버 종료.")
    
    except Exception as e:
        logger.error(f"스트리밍 서버 오류: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
