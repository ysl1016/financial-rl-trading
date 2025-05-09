import asyncio
import json
import logging
import numpy as np
import pandas as pd
import websockets
import time
from typing import Dict, List, Any, Optional, Callable, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread, Event

from ...utils.indicators import calculate_technical_indicators
from ...models.grpo_agent import GRPOAgent

logger = logging.getLogger(__name__)

class StreamingConfig(BaseModel):
    """스트리밍 구성 모델"""
    symbols: List[str] = Field(..., description="데이터를 수집할 심볼 목록")
    interval: str = Field("1m", description="데이터 수집 간격 (1m, 5m, 15m, 1h, 1d 등)")
    lookback_periods: int = Field(100, description="기술적 지표 계산을 위한 룩백 기간")
    buffer_size: int = Field(1000, description="각 심볼당 데이터 버퍼 크기")
    reconnect_delay: int = Field(5, description="연결 끊김 시 재연결 시도 간격 (초)")
    data_source: str = Field("binance", description="데이터 소스 (binance, yahoo, alpaca 등)")
    api_key: Optional[str] = Field(None, description="API 키 (필요한 경우)")
    api_secret: Optional[str] = Field(None, description="API 시크릿 (필요한 경우)")
    use_testnet: bool = Field(False, description="테스트넷 사용 여부")
    custom_endpoint: Optional[str] = Field(None, description="사용자 정의 WebSocket 엔드포인트")
    calculate_indicators: bool = Field(True, description="기술적 지표 계산 여부")
    log_level: str = Field("INFO", description="로깅 레벨")

class DataStream:
    """
    실시간 금융 데이터 스트림 처리 클래스.
    WebSocket을 통해 실시간 시장 데이터를 수신하고 처리합니다.
    """
    
    def __init__(self, config: StreamingConfig):
        """
        DataStream 초기화
        
        Args:
            config: 스트리밍 구성
        """
        self.config = config
        self.data_buffers = {symbol: pd.DataFrame() for symbol in config.symbols}
        self.latest_data = {symbol: {} for symbol in config.symbols}
        self.subscribers: Set[Callable] = set()
        self.is_running = False
        self.stop_event = Event()
        
        # 로깅 설정
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # 연결 설정
        self._setup_connection()
    
    def _setup_connection(self):
        """데이터 소스 연결 설정"""
        if self.config.data_source.lower() == "binance":
            self.endpoint = self._get_binance_endpoint()
        elif self.config.data_source.lower() == "alpaca":
            self.endpoint = self._get_alpaca_endpoint()
        else:
            # 사용자 정의 엔드포인트 사용
            if self.config.custom_endpoint:
                self.endpoint = self.config.custom_endpoint
            else:
                raise ValueError(f"Unsupported data source: {self.config.data_source}")
    
    def _get_binance_endpoint(self) -> str:
        """Binance WebSocket 엔드포인트 생성"""
        base_url = "wss://stream.binance.com:9443/ws" if not self.config.use_testnet else "wss://testnet.binance.vision/ws"
        
        # 여러 심볼 스트림을 결합
        streams = []
        for symbol in self.config.symbols:
            symbol = symbol.lower()
            interval = self.config.interval.lower()
            streams.append(f"{symbol}@kline_{interval}")
        
        # 결합된 스트림 URL 생성
        if len(streams) > 1:
            combined_streams = "/".join(streams)
            return f"{base_url}/{combined_streams}"
        else:
            return f"{base_url}/{streams[0]}"
    
    def _get_alpaca_endpoint(self) -> str:
        """Alpaca WebSocket 엔드포인트 생성"""
        return "wss://stream.data.alpaca.markets/v2/iex"
    
    async def _connect(self):
        """WebSocket 연결 및 데이터 처리"""
        while not self.stop_event.is_set():
            try:
                logger.info(f"WebSocket 연결 시도: {self.endpoint}")
                async with websockets.connect(self.endpoint) as websocket:
                    # Alpaca API 인증
                    if self.config.data_source.lower() == "alpaca" and self.config.api_key and self.config.api_secret:
                        await self._authenticate_alpaca(websocket)
                    
                    # 스트림 구독
                    await self._subscribe(websocket)
                    
                    # 연결 성공 로그
                    logger.info("WebSocket 연결 성공, 데이터 수신 시작")
                    
                    # 메시지 처리 루프
                    while not self.stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            await self._process_message(message)
                        except asyncio.TimeoutError:
                            # 연결 유지를 위한 핑
                            try:
                                pong_waiter = await websocket.ping()
                                await asyncio.wait_for(pong_waiter, timeout=10)
                            except asyncio.TimeoutError:
                                logger.warning("Ping 시간 초과, 연결 재시도")
                                break
                        except Exception as e:
                            logger.error(f"메시지 수신 오류: {e}")
                            break
            
            except Exception as e:
                if self.stop_event.is_set():
                    break
                
                logger.error(f"WebSocket 연결 오류: {e}")
                logger.info(f"{self.config.reconnect_delay}초 후 재연결 시도")
                
                # 재연결 전 대기
                await asyncio.sleep(self.config.reconnect_delay)
    
    async def _authenticate_alpaca(self, websocket):
        """Alpaca API 인증"""
        auth_message = {
            "action": "auth",
            "key": self.config.api_key,
            "secret": self.config.api_secret
        }
        await websocket.send(json.dumps(auth_message))
        
        # 인증 응답 확인
        response = await websocket.recv()
        response_data = json.loads(response)
        
        if "data" in response_data and "status" in response_data["data"]:
            if response_data["data"]["status"] == "authorized":
                logger.info("Alpaca API 인증 성공")
            else:
                logger.error(f"Alpaca API 인증 실패: {response_data}")
                raise Exception("Alpaca API 인증 실패")
    
    async def _subscribe(self, websocket):
        """데이터 스트림 구독"""
        if self.config.data_source.lower() == "alpaca":
            # Alpaca 스트림 구독
            subscribe_message = {
                "action": "subscribe",
                "bars": self.config.symbols  # 분봉 데이터 구독
            }
            await websocket.send(json.dumps(subscribe_message))
        elif self.config.data_source.lower() == "binance":
            # Binance는 연결 시 자동 구독, 추가 작업 필요 없음
            pass
    
    async def _process_message(self, message):
        """
        수신된 WebSocket 메시지 처리
        
        Args:
            message: 수신된 JSON 메시지
        """
        try:
            data = json.loads(message)
            
            if self.config.data_source.lower() == "binance":
                await self._process_binance_message(data)
            elif self.config.data_source.lower() == "alpaca":
                await self._process_alpaca_message(data)
            else:
                logger.warning(f"지원되지 않는 데이터 소스: {self.config.data_source}")
        
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 오류: {message}")
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")
    
    async def _process_binance_message(self, data):
        """
        Binance WebSocket 메시지 처리
        
        Args:
            data: 수신된 메시지 데이터
        """
        try:
            # 캔들스틱 데이터 확인
            if "k" in data:
                kline = data["k"]
                symbol = data["s"].upper()  # 심볼명 (대문자)
                
                # 심볼이 구독 목록에 있는지 확인
                if symbol not in self.config.symbols:
                    return
                
                # 캔들스틱 정보 추출
                candle_data = {
                    "timestamp": int(kline["t"]) / 1000,  # 밀리초를 초로 변환
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "closed": kline["x"]  # 캔들이 완성되었는지 여부
                }
                
                # 캔들이 완성된 경우에만 처리
                if candle_data["closed"]:
                    # 데이터프레임으로 변환하여 버퍼에 추가
                    df_row = pd.DataFrame([candle_data])
                    df_row.set_index(pd.to_datetime(df_row["timestamp"], unit="s"), inplace=True)
                    
                    # 버퍼에 추가
                    if self.data_buffers[symbol].empty:
                        self.data_buffers[symbol] = df_row
                    else:
                        self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], df_row])
                    
                    # 버퍼 크기 제한
                    if len(self.data_buffers[symbol]) > self.config.buffer_size:
                        self.data_buffers[symbol] = self.data_buffers[symbol].iloc[-self.config.buffer_size:]
                    
                    # 기술적 지표 계산
                    if self.config.calculate_indicators and len(self.data_buffers[symbol]) >= self.config.lookback_periods:
                        # OHLCV 열 이름 변경
                        ohlcv_data = self.data_buffers[symbol].rename(
                            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
                        )
                        
                        # 지표 계산
                        indicators = calculate_technical_indicators(ohlcv_data)
                        
                        # 가장 최근 행 추출
                        latest_row = ohlcv_data.iloc[-1].to_dict()
                        latest_indicators = {}
                        
                        # 계산된 지표 중 _norm 접미사가 있는 것만 선택
                        for col in indicators.columns:
                            if col.endswith('_norm'):
                                latest_indicators[col] = indicators[col].iloc[-1]
                        
                        # 최신 데이터 업데이트
                        self.latest_data[symbol] = {**latest_row, **latest_indicators}
                        
                        # 구독자에게 알림
                        self._notify_subscribers(symbol, self.latest_data[symbol])
        
        except Exception as e:
            logger.error(f"Binance 메시지 처리 오류: {e}")
    
    async def _process_alpaca_message(self, data):
        """
        Alpaca WebSocket 메시지 처리
        
        Args:
            data: 수신된 메시지 데이터
        """
        try:
            # 분봉 데이터 확인
            if "data" in data and data.get("stream") == "bars":
                for bar in data["data"]:
                    symbol = bar["S"]  # 심볼명
                    
                    # 심볼이 구독 목록에 있는지 확인
                    if symbol not in self.config.symbols:
                        continue
                    
                    # 캔들스틱 정보 추출
                    candle_data = {
                        "timestamp": pd.Timestamp(bar["t"]).timestamp(),
                        "open": float(bar["o"]),
                        "high": float(bar["h"]),
                        "low": float(bar["l"]),
                        "close": float(bar["c"]),
                        "volume": float(bar["v"]),
                        "closed": True  # Alpaca는 항상 완성된 캔들만 전송
                    }
                    
                    # 데이터프레임으로 변환하여 버퍼에 추가
                    df_row = pd.DataFrame([candle_data])
                    df_row.set_index(pd.to_datetime(df_row["timestamp"], unit="s"), inplace=True)
                    
                    # 버퍼에 추가
                    if self.data_buffers[symbol].empty:
                        self.data_buffers[symbol] = df_row
                    else:
                        self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], df_row])
                    
                    # 버퍼 크기 제한
                    if len(self.data_buffers[symbol]) > self.config.buffer_size:
                        self.data_buffers[symbol] = self.data_buffers[symbol].iloc[-self.config.buffer_size:]
                    
                    # 기술적 지표 계산
                    if self.config.calculate_indicators and len(self.data_buffers[symbol]) >= self.config.lookback_periods:
                        # OHLCV 열 이름 변경
                        ohlcv_data = self.data_buffers[symbol].rename(
                            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
                        )
                        
                        # 지표 계산
                        indicators = calculate_technical_indicators(ohlcv_data)
                        
                        # 가장 최근 행 추출
                        latest_row = ohlcv_data.iloc[-1].to_dict()
                        latest_indicators = {}
                        
                        # 계산된 지표 중 _norm 접미사가 있는 것만 선택
                        for col in indicators.columns:
                            if col.endswith('_norm'):
                                latest_indicators[col] = indicators[col].iloc[-1]
                        
                        # 최신 데이터 업데이트
                        self.latest_data[symbol] = {**latest_row, **latest_indicators}
                        
                        # 구독자에게 알림
                        self._notify_subscribers(symbol, self.latest_data[symbol])
        
        except Exception as e:
            logger.error(f"Alpaca 메시지 처리 오류: {e}")
    
    def _notify_subscribers(self, symbol: str, data: Dict[str, Any]):
        """
        구독자에게 새 데이터 알림
        
        Args:
            symbol: 심볼명
            data: 최신 데이터
        """
        for subscriber in self.subscribers:
            try:
                subscriber(symbol, data)
            except Exception as e:
                logger.error(f"구독자 알림 오류: {e}")
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 최신 데이터 조회
        
        Args:
            symbol: 심볼명
            
        Returns:
            최신 데이터 (OHLCV + 기술적 지표)
        """
        if symbol in self.latest_data:
            return self.latest_data[symbol]
        return {}
    
    def get_historical_data(self, symbol: str, lookback: int = None) -> pd.DataFrame:
        """
        특정 심볼의 역사적 데이터 조회
        
        Args:
            symbol: 심볼명
            lookback: 조회할 이전 데이터 수 (None이면 전체)
            
        Returns:
            역사적 데이터 DataFrame
        """
        if symbol in self.data_buffers:
            if lookback is not None and lookback > 0:
                return self.data_buffers[symbol].iloc[-lookback:]
            return self.data_buffers[symbol]
        return pd.DataFrame()
    
    def subscribe(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        데이터 업데이트 구독
        
        Args:
            callback: 콜백 함수 (symbol, data)
        """
        self.subscribers.add(callback)
        return self
    
    def unsubscribe(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        데이터 업데이트 구독 해제
        
        Args:
            callback: 등록된 콜백 함수
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
        return self
    
    def start(self):
        """데이터 스트림 시작"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            
            # 백그라운드 스레드에서 비동기 루프 실행
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._connect())
                loop.close()
            
            self.thread = Thread(target=run_async_loop, daemon=True)
            self.thread.start()
            
            logger.info("데이터 스트림 시작됨")
        return self
    
    def stop(self):
        """데이터 스트림 중지"""
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            
            if hasattr(self, 'thread'):
                self.thread.join(timeout=5)
            
            logger.info("데이터 스트림 중지됨")
        return self


class StreamProcessor:
    """
    스트리밍 데이터를 처리하여 모델 예측을 수행하는 클래스.
    데이터 스트림을 구독하고 새 데이터가 도착할 때마다 모델 예측을 실행합니다.
    """
    
    def __init__(self, model: GRPOAgent, data_stream: DataStream):
        """
        StreamProcessor 초기화
        
        Args:
            model: 예측에 사용할 GRPOAgent 인스턴스
            data_stream: 구독할 DataStream 인스턴스
        """
        self.model = model
        self.data_stream = data_stream
        self.latest_predictions = {}
        self.prediction_queue = Queue()
        self.is_processing = False
        self.stop_event = Event()
        
        # 콜백 함수 등록
        self.data_stream.subscribe(self._on_data_update)
    
    def _on_data_update(self, symbol: str, data: Dict[str, Any]):
        """
        새로운 데이터 업데이트 처리
        
        Args:
            symbol: 심볼명
            data: 업데이트된 데이터
        """
        # 큐에 예측 작업 추가
        self.prediction_queue.put((symbol, data))
    
    def _process_predictions(self):
        """예측 큐 처리 루프"""
        while not self.stop_event.is_set():
            try:
                # 큐에서 작업 가져오기 (1초 타임아웃)
                try:
                    symbol, data = self.prediction_queue.get(timeout=1)
                except Queue.Empty:
                    continue
                
                # 모델 입력 준비
                state = self._prepare_model_input(symbol, data)
                
                # 모델 예측 실행
                with torch.no_grad():
                    action = self.model.select_action(state)
                    
                    # 행동 확률 계산
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
                    action_probs = self.model.network(state_tensor).cpu().numpy()[0]
                    
                    # 가치 추정
                    try:
                        action_onehot = np.zeros(self.model.action_dim)
                        action_onehot[action] = 1.0
                        action_onehot_tensor = torch.FloatTensor(action_onehot).unsqueeze(0).to(self.model.device)
                        
                        state_value = self.model.network.estimate_q_value(
                            state_tensor, action_onehot_tensor
                        ).cpu().numpy()[0][0]
                    except:
                        state_value = 0.0
                
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
                
                # 예측 결과 저장
                prediction = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "action": action_name,
                    "position_size": position_size,
                    "confidence": confidence,
                    "expected_value": float(state_value),
                    "action_probabilities": {
                        "hold": float(action_probs[0]),
                        "buy": float(action_probs[1]),
                        "sell": float(action_probs[2])
                    },
                    "data": data
                }
                
                # 최신 예측 업데이트
                self.latest_predictions[symbol] = prediction
                
                # 예측 완료 표시
                self.prediction_queue.task_done()
                
                logger.debug(f"예측 완료: {symbol}, 행동: {action_name}, 확신도: {confidence:.4f}")
                
            except Exception as e:
                logger.error(f"예측 처리 오류: {e}")
    
    def _prepare_model_input(self, symbol: str, data: Dict[str, Any]) -> np.ndarray:
        """
        모델 입력 준비
        
        Args:
            symbol: 심볼명
            data: 최신 데이터
            
        Returns:
            모델 입력 상태 벡터
        """
        # 필요한 특성 선택
        features = []
        
        # 필요한 특성 확인 및 추가
        required_features = [
            'RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm', 
            'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
            'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm'
        ]
        
        for feature in required_features:
            if feature in data:
                features.append(data[feature])
            else:
                # 누락된 특성은 0으로 처리
                logger.warning(f"Missing feature: {feature}, using 0 as default")
                features.append(0.0)
        
        # 포트폴리오 상태를 위한 추가 특성
        # 현재 구현에서는 position과 portfolio_return에 대한 자리만 마련 (0으로 설정)
        features.append(0.0)  # position 자리
        features.append(0.0)  # portfolio_return 자리
        
        return np.array(features, dtype=np.float32)
    
    def get_latest_prediction(self, symbol: str) -> Dict[str, Any]:
        """
        특정 심볼의 최신 예측 조회
        
        Args:
            symbol: 심볼명
            
        Returns:
            최신 예측 결과
        """
        if symbol in self.latest_predictions:
            return self.latest_predictions[symbol]
        return {}
    
    def get_all_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 심볼의 최신 예측 조회
        
        Returns:
            {심볼: 예측 결과} 형태의 딕셔너리
        """
        return self.latest_predictions
    
    def start(self):
        """예측 처리 시작"""
        if not self.is_processing:
            self.is_processing = True
            self.stop_event.clear()
            
            # 백그라운드 스레드에서 처리 루프 실행
            self.thread = Thread(target=self._process_predictions, daemon=True)
            self.thread.start()
            
            logger.info("스트림 처리기 시작됨")
        return self
    
    def stop(self):
        """예측 처리 중지"""
        if self.is_processing:
            self.is_processing = False
            self.stop_event.set()
            
            if hasattr(self, 'thread'):
                self.thread.join(timeout=5)
            
            logger.info("스트림 처리기 중지됨")
        return self
