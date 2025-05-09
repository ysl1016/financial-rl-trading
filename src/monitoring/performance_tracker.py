import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """트레이딩 성과 지표"""
    
    # 수익률 지표
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    
    # 리스크 지표
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0  # 95% Value at Risk
    
    # 트레이딩 효율성 지표
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    # 거래 통계
    num_trades: int = 0
    num_winning_trades: int = 0
    num_losing_trades: int = 0
    avg_trade_duration: float = 0.0  # 일 단위
    
    # 시간대별 성과
    daily_performance: Dict[str, float] = field(default_factory=dict)
    monthly_performance: Dict[str, float] = field(default_factory=dict)
    
    # 모델 성능 지표
    prediction_accuracy: float = 0.0
    avg_confidence: float = 0.0
    
    # 메타데이터
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    asset_name: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {k: v for k, v in asdict(self).items() if not (k == 'daily_returns' and len(v) > 100)}
    
    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=indent)


class PerformanceTracker:
    """
    모델 성능 추적 클래스.
    트레이딩 모델의 성과를 추적하고 분석합니다.
    """
    
    def __init__(self, 
                asset_name: str, 
                initial_capital: float = 100000.0,
                risk_free_rate: float = 0.02,
                log_dir: Optional[str] = None):
        """
        PerformanceTracker 초기화
        
        Args:
            asset_name: 자산명 (추적 식별용)
            initial_capital: 초기 자본금
            risk_free_rate: 무위험 이자율 (연율)
            log_dir: 로그 저장 디렉토리
        """
        self.asset_name = asset_name
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.log_dir = Path(log_dir) if log_dir else None
        
        # 성과 추적 데이터
        self.portfolio_values = [initial_capital]
        self.portfolio_returns = []
        self.timestamps = [datetime.now()]
        
        # 거래 기록
        self.trades = []
        
        # 예측 기록
        self.predictions = []
        
        # 메트릭 계산 상태
        self.metrics = TradeMetrics(asset_name=asset_name)
        self.metrics.start_date = datetime.now().isoformat()
        
        # 로그 디렉토리 생성
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def add_portfolio_update(self, 
                           timestamp: datetime, 
                           portfolio_value: float):
        """
        포트폴리오 가치 업데이트 추가
        
        Args:
            timestamp: 타임스탬프
            portfolio_value: 포트폴리오 가치
        """
        # 타임스탬프와 포트폴리오 가치 추가
        self.timestamps.append(timestamp)
        self.portfolio_values.append(portfolio_value)
        
        # 수익률 계산
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            current_value = self.portfolio_values[-1]
            
            if prev_value > 0:
                returns = (current_value / prev_value) - 1
            else:
                returns = 0.0
            
            self.portfolio_returns.append(returns)
        
        # 일일 성과 기록
        day_key = timestamp.strftime('%Y-%m-%d')
        if day_key not in self.metrics.daily_performance:
            self.metrics.daily_performance[day_key] = 0.0
        
        if len(self.portfolio_values) > 1:
            daily_return = self.portfolio_returns[-1]
            self.metrics.daily_performance[day_key] += daily_return
        
        # 월간 성과 기록
        month_key = timestamp.strftime('%Y-%m')
        if month_key not in self.metrics.monthly_performance:
            self.metrics.monthly_performance[month_key] = 0.0
        
        if len(self.portfolio_values) > 1:
            monthly_return = self.portfolio_returns[-1]
            self.metrics.monthly_performance[month_key] += monthly_return
    
    def add_trade(self, 
                 timestamp: datetime, 
                 symbol: str, 
                 action: str, 
                 quantity: float, 
                 price: float, 
                 costs: float,
                 trade_id: Optional[str] = None):
        """
        거래 기록 추가
        
        Args:
            timestamp: 거래 시간
            symbol: 심볼
            action: 행동 (buy, sell)
            quantity: 수량
            price: 가격
            costs: 거래 비용
            trade_id: 거래 ID (None이면 자동 생성)
        """
        # 거래 ID 생성
        if trade_id is None:
            trade_id = f"trade_{len(self.trades) + 1}_{int(timestamp.timestamp())}"
        
        # 거래 정보 구성
        trade = {
            "trade_id": trade_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "costs": costs,
            "total_value": price * quantity
        }
        
        # 거래 기록에 추가
        self.trades.append(trade)
        
        # 이전 거래 검색 (매수-매도 쌍 확인)
        if action == "sell" and len(self.trades) > 1:
            # 같은 심볼의 가장 최근 매수 찾기
            buy_trades = [(i, t) for i, t in enumerate(self.trades[:-1]) 
                         if t["symbol"] == symbol and t["action"] == "buy"]
            
            if buy_trades:
                # 가장 최근 매수 선택
                buy_idx, buy_trade = buy_trades[-1]
                
                # 매도 수량이 매수 수량보다 작거나 같은지 확인
                if quantity <= buy_trade["quantity"]:
                    # 매수-매도 쌍 완성, 수익/손실 계산
                    buy_price = buy_trade["price"]
                    sell_price = price
                    
                    # 수익/손실 계산 (비용 포함)
                    trade_profit = (sell_price - buy_price) * quantity - costs - buy_trade["costs"]
                    trade_return = trade_profit / (buy_price * quantity) if buy_price * quantity > 0 else 0
                    
                    # 거래 기간 계산
                    duration = (timestamp - buy_trade["timestamp"]).total_seconds() / (60 * 60 * 24)  # 일 단위
                    
                    # 완료된 거래 정보 업데이트
                    self.trades[-1]["paired_trade_id"] = buy_trade["trade_id"]
                    self.trades[-1]["profit"] = trade_profit
                    self.trades[-1]["return"] = trade_return
                    self.trades[-1]["duration"] = duration
                    
                    # 매수 거래 업데이트 (쌍이 이루어짐을 표시)
                    self.trades[buy_idx]["paired_trade_id"] = trade_id
                    
                    # 승리/패배 업데이트
                    if trade_profit > 0:
                        self.metrics.num_winning_trades += 1
                    else:
                        self.metrics.num_losing_trades += 1
    
    def add_prediction(self, 
                      timestamp: datetime, 
                      symbol: str, 
                      predicted_action: str, 
                      confidence: float,
                      actual_price_change: Optional[float] = None):
        """
        예측 기록 추가
        
        Args:
            timestamp: 예측 시간
            symbol: 심볼
            predicted_action: 예측된 행동 (buy, sell, hold)
            confidence: 예측 확신도
            actual_price_change: 실제 가격 변화 (알 수 있는 경우)
        """
        # 예측 정보 구성
        prediction = {
            "timestamp": timestamp,
            "symbol": symbol,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "actual_price_change": actual_price_change
        }
        
        # 예측이 맞았는지 평가 (실제 가격 변화를 알 수 있는 경우)
        if actual_price_change is not None:
            correct_prediction = (
                (predicted_action == "buy" and actual_price_change > 0) or
                (predicted_action == "sell" and actual_price_change < 0) or
                (predicted_action == "hold" and abs(actual_price_change) < 0.001)  # 작은 변화는 보유로 간주
            )
            prediction["correct"] = correct_prediction
        
        # 예측 기록에 추가
        self.predictions.append(prediction)
    
    def update_metrics(self):
        """성과 지표 업데이트"""
        # 거래 수 업데이트
        self.metrics.num_trades = len([t for t in self.trades if "profit" in t])
        
        # 시작일/종료일 업데이트
        if self.timestamps:
            self.metrics.start_date = min(self.timestamps).isoformat()
            self.metrics.end_date = max(self.timestamps).isoformat()
        
        # 수익률 지표 계산
        if len(self.portfolio_values) > 1:
            # 총 수익률
            initial_value = self.portfolio_values[0]
            final_value = self.portfolio_values[-1]
            
            if initial_value > 0:
                self.metrics.total_return = (final_value / initial_value) - 1
            else:
                self.metrics.total_return = 0.0
            
            # 연간 수익률 (운영 일수 기준)
            if len(self.timestamps) > 1:
                days = (self.timestamps[-1] - self.timestamps[0]).days
                if days > 0:
                    self.metrics.annualized_return = (1 + self.metrics.total_return) ** (365 / days) - 1
            
            # 일일 수익률 저장
            self.metrics.daily_returns = self.portfolio_returns.copy()
        
        # 리스크 지표 계산
        if len(self.portfolio_returns) > 1:
            # 변동성 (표준편차)
            self.metrics.volatility = np.std(self.portfolio_returns) * np.sqrt(252)  # 연간 변동성
            
            # Sharpe 비율
            if self.metrics.volatility > 0:
                excess_return = self.metrics.annualized_return - self.risk_free_rate
                self.metrics.sharpe_ratio = excess_return / self.metrics.volatility
            
            # Sortino 비율 (하방 위험만 고려)
            downside_returns = [r for r in self.portfolio_returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    self.metrics.sortino_ratio = (self.metrics.annualized_return - self.risk_free_rate) / downside_deviation
            
            # 최대 손실률 (Maximum Drawdown)
            if len(self.portfolio_values) > 1:
                max_value = self.portfolio_values[0]
                max_drawdown = 0.0
                
                for value in self.portfolio_values[1:]:
                    if value > max_value:
                        max_value = value
                    
                    drawdown = (max_value - value) / max_value if max_value > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                self.metrics.max_drawdown = max_drawdown
                
                # Calmar 비율
                if max_drawdown > 0:
                    self.metrics.calmar_ratio = self.metrics.annualized_return / max_drawdown
            
            # Value at Risk (95%)
            if self.portfolio_returns:
                self.metrics.var_95 = np.percentile(self.portfolio_returns, 5)
        
        # 거래 효율성 지표 계산
        if self.metrics.num_trades > 0:
            # 승률
            self.metrics.win_rate = self.metrics.num_winning_trades / self.metrics.num_trades
            
            # 수익/손실 비율
            winning_trades = [t for t in self.trades if "profit" in t and t["profit"] > 0]
            losing_trades = [t for t in self.trades if "profit" in t and t["profit"] <= 0]
            
            avg_win = np.mean([t["profit"] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t["profit"] for t in losing_trades])) if losing_trades else 0
            
            self.metrics.avg_win = avg_win
            self.metrics.avg_loss = avg_loss
            
            if avg_loss > 0:
                self.metrics.profit_loss_ratio = avg_win / avg_loss
            
            # 기대 수익 (Expectancy)
            self.metrics.expectancy = (self.metrics.win_rate * avg_win) - ((1 - self.metrics.win_rate) * avg_loss)
            
            # 평균 거래 기간
            trades_with_duration = [t for t in self.trades if "duration" in t]
            if trades_with_duration:
                self.metrics.avg_trade_duration = np.mean([t["duration"] for t in trades_with_duration])
        
        # 모델 성능 지표 계산
        predictions_with_result = [p for p in self.predictions if "correct" in p]
        if predictions_with_result:
            self.metrics.prediction_accuracy = np.mean([1 if p["correct"] else 0 for p in predictions_with_result])
        
        if self.predictions:
            self.metrics.avg_confidence = np.mean([p["confidence"] for p in self.predictions])
    
    def get_metrics(self) -> TradeMetrics:
        """
        현재 성과 지표 조회
        
        Returns:
            계산된 성과 지표
        """
        # 지표 업데이트 후 반환
        self.update_metrics()
        return self.metrics
    
    def save_metrics(self, 
                    filename: Optional[str] = None) -> str:
        """
        성과 지표를 파일로 저장
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        # 지표 업데이트
        self.update_metrics()
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.asset_name}_metrics_{timestamp}.json"
        
        # 저장 경로 결정
        if self.log_dir:
            file_path = self.log_dir / filename
        else:
            file_path = Path(filename)
        
        # JSON으로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        logger.info(f"성과 지표가 저장되었습니다: {file_path}")
        return str(file_path)
    
    def save_portfolio_history(self, 
                             filename: Optional[str] = None) -> str:
        """
        포트폴리오 이력을 CSV 파일로 저장
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        # 데이터프레임 생성
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'portfolio_value': self.portfolio_values
        })
        
        if len(self.portfolio_returns) > 0:
            df['return'] = [0.0] + self.portfolio_returns
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.asset_name}_portfolio_history_{timestamp}.csv"
        
        # 저장 경로 결정
        if self.log_dir:
            file_path = self.log_dir / filename
        else:
            file_path = Path(filename)
        
        # CSV로 저장
        df.to_csv(file_path, index=False)
        
        logger.info(f"포트폴리오 이력이 저장되었습니다: {file_path}")
        return str(file_path)
    
    def save_trades(self, 
                   filename: Optional[str] = None) -> str:
        """
        거래 기록을 CSV 파일로 저장
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        # 데이터프레임 생성
        df = pd.DataFrame(self.trades)
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.asset_name}_trades_{timestamp}.csv"
        
        # 저장 경로 결정
        if self.log_dir:
            file_path = self.log_dir / filename
        else:
            file_path = Path(filename)
        
        # CSV로 저장
        df.to_csv(file_path, index=False)
        
        logger.info(f"거래 기록이 저장되었습니다: {file_path}")
        return str(file_path)
    
    def load_metrics(self, 
                    filepath: str) -> TradeMetrics:
        """
        저장된 성과 지표 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 성과 지표
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # TradeMetrics 객체 생성
        metrics = TradeMetrics(**data)
        
        return metrics
    
    def generate_report(self, 
                       include_plots: bool = True) -> Dict[str, Any]:
        """
        종합 성과 보고서 생성
        
        Args:
            include_plots: 차트 데이터 포함 여부
            
        Returns:
            종합 보고서 데이터
        """
        # 성과 지표 업데이트
        self.update_metrics()
        
        # 기본 보고서 데이터
        report = {
            "asset_name": self.asset_name,
            "report_date": datetime.now().isoformat(),
            "start_date": self.metrics.start_date,
            "end_date": self.metrics.end_date,
            "metrics": self.metrics.to_dict(),
            "summary": {
                "total_return_pct": f"{self.metrics.total_return * 100:.2f}%",
                "annualized_return_pct": f"{self.metrics.annualized_return * 100:.2f}%",
                "sharpe_ratio": f"{self.metrics.sharpe_ratio:.2f}",
                "max_drawdown_pct": f"{self.metrics.max_drawdown * 100:.2f}%",
                "win_rate_pct": f"{self.metrics.win_rate * 100:.2f}%",
                "num_trades": self.metrics.num_trades
            }
        }
        
        # 차트 데이터 추가 (선택적)
        if include_plots:
            report["plot_data"] = {
                "portfolio_value": {
                    "x": [t.isoformat() for t in self.timestamps],
                    "y": self.portfolio_values
                },
                "returns": {
                    "x": [t.isoformat() for t in self.timestamps[1:]],
                    "y": self.portfolio_returns
                },
                "daily_performance": {
                    "x": list(self.metrics.daily_performance.keys()),
                    "y": list(self.metrics.daily_performance.values())
                },
                "monthly_performance": {
                    "x": list(self.metrics.monthly_performance.keys()),
                    "y": list(self.metrics.monthly_performance.values())
                }
            }
        
        return report
