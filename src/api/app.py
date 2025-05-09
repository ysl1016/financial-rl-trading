import os
import logging
import uvicorn
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader

from pydantic import BaseModel, Field
from datetime import datetime

from ..deployment.model_packaging import ModelPackager
from ..deployment.model_optimization import ModelOptimizer
from ..models.grpo_agent import GRPOAgent

# 로깅 설정
logger = logging.getLogger(__name__)

# API 키 인증 설정
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 모델 인스턴스와 메타데이터를 저장할 전역 변수
model_instance = None
model_metadata = None
env_config = None

class TradingRequest(BaseModel):
    """트레이딩 의사결정 요청 모델"""
    market_data: Dict[str, List[float]] = Field(..., 
        description="시장 데이터 (OHLCV 및 기술적 지표)")
    portfolio_state: Dict[str, Any] = Field(..., 
        description="현재 포트폴리오 상태 (포지션, 현금 등)")
    constraints: Optional[Dict[str, Any]] = Field(None, 
        description="추가 제약 조건 (위험 허용도, 최대 포지션 등)")
    timestamp: Optional[datetime] = Field(None, 
        description="요청 시간")

class TradingResponse(BaseModel):
    """트레이딩 의사결정 응답 모델"""
    action: str = Field(..., 
        description="결정된 행동 (매수, 매도, 보유)")
    position_size: float = Field(..., 
        description="추천 포지션 크기 (0.0 ~ 1.0)")
    confidence: float = Field(..., 
        description="결정에 대한 확신도 (0.0 ~ 1.0)")
    expected_value: float = Field(..., 
        description="예상 가치")
    rationale: Dict[str, Any] = Field(..., 
        description="결정 근거 및 추가 정보")
    timestamp: datetime = Field(..., 
        description="응답 시간")

class BatchTradingRequest(BaseModel):
    """배치 트레이딩 의사결정 요청 모델"""
    requests: List[TradingRequest] = Field(..., 
        description="트레이딩 요청 목록")

class BatchTradingResponse(BaseModel):
    """배치 트레이딩 의사결정 응답 모델"""
    responses: List[TradingResponse] = Field(..., 
        description="트레이딩 응답 목록")
    batch_processing_time_ms: float = Field(..., 
        description="배치 처리 시간 (밀리초)")

class ModelInfo(BaseModel):
    """모델 정보 응답 모델"""
    model_name: str = Field(..., description="모델 이름")
    version: str = Field(..., description="모델 버전")
    created_at: str = Field(..., description="생성 시간")
    framework_version: str = Field(..., description="프레임워크 버전")
    parameters: Dict[str, Any] = Field(..., description="모델 매개변수")
    capabilities: List[str] = Field(..., description="모델 기능")
    performance_metrics: Dict[str, Any] = Field(..., description="성능 지표")

def get_api_key(api_key_header: str = Depends(api_key_header)):
    """API 키 검증"""
    expected_api_key = os.getenv("API_KEY", "default_development_key")
    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

def preprocess_market_data(market_data: Dict[str, List[float]]) -> np.ndarray:
    """시장 데이터 전처리"""
    # 모든 특성을 NumPy 배열로 변환
    features = []
    
    # 필요한 특성 확인 및 추가
    required_features = [
        'RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm', 
        'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
        'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm'
    ]
    
    for feature in required_features:
        if feature in market_data:
            features.append(market_data[feature][-1])  # 가장 최근 값 사용
        else:
            # 누락된 특성은 0으로 처리
            logger.warning(f"Missing feature: {feature}, using 0 as default")
            features.append(0.0)
    
    # 포트폴리오 상태를 위한 추가 특성
    # 현재 구현에서는 position과 portfolio_return을 위한 공간 추가
    features.append(0.0)  # position 자리 (실제 값은 예측 시에 설정)
    features.append(0.0)  # portfolio_return 자리
    
    return np.array(features, dtype=np.float32)

def action_to_response(action: int, action_probs: np.ndarray, state_value: float) -> Dict[str, Any]:
    """모델 출력을 응답 형식으로 변환"""
    # 행동 매핑 (0=보유, 1=매수, 2=매도)
    action_mapping = {0: "hold", 1: "buy", 2: "sell"}
    action_name = action_mapping.get(action, "unknown")
    
    # 확신도 계산 (선택된 행동의 확률)
    confidence = float(action_probs[action])
    
    # 포지션 크기 결정 (확신도에 비례)
    position_size = 0.0
    if action_name == "buy":
        position_size = confidence
    elif action_name == "sell":
        position_size = -confidence
    
    response = {
        "action": action_name,
        "position_size": position_size,
        "confidence": confidence,
        "expected_value": float(state_value),
        "rationale": {
            "action_probabilities": {
                "hold": float(action_probs[0]),
                "buy": float(action_probs[1]),
                "sell": float(action_probs[2])
            },
            "state_value": float(state_value)
        },
        "timestamp": datetime.now()
    }
    
    return response

def create_app(model_path: str, optimization_level: str = "medium") -> FastAPI:
    """FastAPI 애플리케이션 생성"""
    app = FastAPI(
        title="DeepSeek-R1 Financial Trading API",
        description="DeepSeek-R1 기반 금융거래 강화학습 모델을 위한 REST API",
        version="1.0.0"
    )
    
    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 허용할 도메인 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip 압축 미들웨어 추가
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 모델 로드
    @app.on_event("startup")
    async def load_model():
        global model_instance, model_metadata, env_config
        
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
            
            # 추론 최적화 적용
            if optimization_level != "none":
                model_optimizer = ModelOptimizer()
                model = model_optimizer.optimize_for_inference(model, optimization_level)
            
            model_instance = model
            model_metadata = metadata
            
            logger.info(f"모델 로드 완료: {model_name}/{version}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    @app.get("/health")
    async def health_check():
        """API 헬스 체크 엔드포인트"""
        if model_instance is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "error", "message": "Model not loaded"}
            )
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    @app.get("/info", response_model=ModelInfo)
    async def get_model_info(_: str = Depends(get_api_key)):
        """모델 정보 조회 엔드포인트"""
        if model_instance is None or model_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # 모델 정보 구성
        info = {
            "model_name": model_metadata.get("model_name", "unknown"),
            "version": model_metadata.get("version", "unknown"),
            "created_at": model_metadata.get("created_at", "unknown"),
            "framework_version": model_metadata.get("framework_version", "unknown"),
            "parameters": {
                "state_dim": model_instance.network.policy[0].in_features,
                "action_dim": model_instance.action_dim,
                "hidden_dim": model_instance.network.policy[2].in_features,
                "gamma": model_instance.gamma,
                "reward_scale": model_instance.reward_scale,
                "penalty_scale": model_instance.penalty_scale
            },
            "capabilities": [
                "market_prediction", 
                "portfolio_optimization", 
                "risk_management"
            ],
            "performance_metrics": model_metadata.get("performance_metrics", {})
        }
        
        return info
    
    @app.post("/predict", response_model=TradingResponse)
    async def predict(request: TradingRequest, _: str = Depends(get_api_key)):
        """단일 트레이딩 의사결정 엔드포인트"""
        if model_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        try:
            # 시장 데이터 전처리
            state = preprocess_market_data(request.market_data)
            
            # 포트폴리오 상태 통합
            current_position = request.portfolio_state.get("position", 0.0)
            portfolio_return = request.portfolio_state.get("portfolio_return", 0.0)
            
            # 상태 벡터 업데이트
            state[-2] = current_position
            state[-1] = portfolio_return
            
            # 모델 추론
            action = model_instance.select_action(state, deterministic=True)
            
            # 행동 확률 계산 (추가 정보용)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model_instance.device)
                action_probs = model_instance.network(state_tensor).cpu().numpy()[0]
                
                # 가치 추정 (현재 모델 구조에서 가능하다면)
                try:
                    # 원-핫 인코딩된 행동 생성
                    action_onehot = np.zeros(model_instance.action_dim)
                    action_onehot[action] = 1.0
                    action_onehot_tensor = torch.FloatTensor(action_onehot).unsqueeze(0).to(model_instance.device)
                    
                    # 가치 추정
                    state_value = model_instance.network.estimate_q_value(
                        state_tensor, action_onehot_tensor
                    ).cpu().numpy()[0][0]
                except:
                    # 가치 추정을 할 수 없는 경우
                    state_value = 0.0
            
            # 응답 생성
            response_data = action_to_response(action, action_probs, state_value)
            
            return TradingResponse(**response_data)
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {str(e)}"
            )
    
    @app.post("/predict/batch", response_model=BatchTradingResponse)
    async def predict_batch(request: BatchTradingRequest, _: str = Depends(get_api_key)):
        """배치 트레이딩 의사결정 엔드포인트"""
        if model_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        import time
        start_time = time.time()
        
        try:
            responses = []
            
            for single_request in request.requests:
                # 각 요청에 대해 predict 로직 실행
                state = preprocess_market_data(single_request.market_data)
                
                current_position = single_request.portfolio_state.get("position", 0.0)
                portfolio_return = single_request.portfolio_state.get("portfolio_return", 0.0)
                
                state[-2] = current_position
                state[-1] = portfolio_return
                
                action = model_instance.select_action(state, deterministic=True)
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model_instance.device)
                    action_probs = model_instance.network(state_tensor).cpu().numpy()[0]
                    
                    try:
                        action_onehot = np.zeros(model_instance.action_dim)
                        action_onehot[action] = 1.0
                        action_onehot_tensor = torch.FloatTensor(action_onehot).unsqueeze(0).to(model_instance.device)
                        
                        state_value = model_instance.network.estimate_q_value(
                            state_tensor, action_onehot_tensor
                        ).cpu().numpy()[0][0]
                    except:
                        state_value = 0.0
                
                response_data = action_to_response(action, action_probs, state_value)
                responses.append(TradingResponse(**response_data))
            
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            return BatchTradingResponse(
                responses=responses,
                batch_processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"배치 예측 중 오류 발생: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction error: {str(e)}"
            )
    
    return app

def run_app(model_path: str, host: str = "0.0.0.0", port: int = 8000, 
           optimization_level: str = "medium", reload: bool = False):
    """API 서버 실행"""
    import torch
    global import torch
    
    app = create_app(model_path, optimization_level)
    
    logger.info(f"API 서버 시작 (host={host}, port={port})")
    uvicorn.run(
        "src.api.app:create_app", 
        host=host, 
        port=port,
        reload=reload,
        factory=True,
        factory_kwargs={"model_path": model_path, "optimization_level": optimization_level}
    )
