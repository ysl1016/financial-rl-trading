import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Callable


class BaseRewardFunction(ABC):
    """보상 함수를 위한 추상 기본 클래스"""
    
    def __init__(self, **kwargs):
        """보상 함수 초기화"""
        self.config = kwargs
    
    @abstractmethod
    def calculate(self, 
                  current_state: Dict, 
                  next_state: Dict, 
                  action: np.ndarray, 
                  info: Dict) -> float:
        """
        보상 계산
        
        Args:
            current_state: 현재 상태 정보
            next_state: 다음 상태 정보
            action: 수행된 행동
            info: 추가 정보
            
        Returns:
            float: 계산된 보상값
        """
        pass


class ReturnReward(BaseRewardFunction):
    """포트폴리오 수익률 기반 보상 함수"""
    
    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    
    def calculate(self, current_state, next_state, action, info):
        # 포트폴리오 가치 변화율 계산
        old_value = current_state.get('portfolio_value', 0)
        new_value = next_state.get('portfolio_value', 0)
        
        if old_value <= 0:
            return 0.0
        
        return self.scale * ((new_value / old_value) - 1.0)


class SharpeReward(BaseRewardFunction):
    """Sharpe 비율 기반 보상 함수"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.0, 
                 window_size: int = 20, 
                 scale: float = 1.0,
                 min_periods: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate / 252  # 일별 무위험 수익률
        self.window_size = window_size
        self.scale = scale
        self.min_periods = min_periods
        self.returns_history = []
    
    def calculate(self, current_state, next_state, action, info):
        # 현재 수익률 계산
        old_value = current_state.get('portfolio_value', 0)
        new_value = next_state.get('portfolio_value', 0)
        
        if old_value <= 0:
            return 0.0
        
        # 수익률 계산 및 히스토리에 추가
        current_return = (new_value / old_value) - 1.0
        self.returns_history.append(current_return)
        
        # 필요 이상으로 긴 히스토리는 잘라냄
        if len(self.returns_history) > self.window_size:
            self.returns_history = self.returns_history[-self.window_size:]
        
        # 충분한 데이터가 있는지 확인
        if len(self.returns_history) < self.min_periods:
            return 0.0
        
        # Sharpe 비율 계산
        mean_return = np.mean(self.returns_history)
        std_return = np.std(self.returns_history) + 1e-9  # 0으로 나누기 방지
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        return self.scale * sharpe


class SortinoReward(BaseRewardFunction):
    """Sortino 비율 기반 보상 함수 (하방 위험만 고려)"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.0, 
                 window_size: int = 20, 
                 scale: float = 1.0,
                 min_periods: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate / 252  # 일별 무위험 수익률
        self.window_size = window_size
        self.scale = scale
        self.min_periods = min_periods
        self.returns_history = []
    
    def calculate(self, current_state, next_state, action, info):
        # 현재 수익률 계산
        old_value = current_state.get('portfolio_value', 0)
        new_value = next_state.get('portfolio_value', 0)
        
        if old_value <= 0:
            return 0.0
        
        # 수익률 계산 및 히스토리에 추가
        current_return = (new_value / old_value) - 1.0
        self.returns_history.append(current_return)
        
        # 필요 이상으로 긴 히스토리는 잘라냄
        if len(self.returns_history) > self.window_size:
            self.returns_history = self.returns_history[-self.window_size:]
        
        # 충분한 데이터가 있는지 확인
        if len(self.returns_history) < self.min_periods:
            return 0.0
        
        # Sortino 비율 계산 (하방 위험만 고려)
        mean_return = np.mean(self.returns_history)
        
        # 하방 편차 계산 (음수 수익률만 사용)
        negative_returns = [r for r in self.returns_history if r < 0]
        if not negative_returns:
            downside_std = 1e-9  # 하방 위험이 없는 경우
        else:
            downside_std = np.sqrt(np.mean(np.square(negative_returns))) + 1e-9
        
        sortino = (mean_return - self.risk_free_rate) / downside_std
        
        return self.scale * sortino


class DrawdownControlReward(BaseRewardFunction):
    """최대 손실(drawdown) 제어 보상 함수"""
    
    def __init__(self, 
                 max_drawdown: float = 0.05,  # 5% 최대 허용 손실
                 penalty_scale: float = 10.0,  # 패널티 스케일
                 **kwargs):
        super().__init__(**kwargs)
        self.max_drawdown = max_drawdown
        self.penalty_scale = penalty_scale
        self.peak_value = 0.0
    
    def calculate(self, current_state, next_state, action, info):
        # 포트폴리오 가치 업데이트
        new_value = next_state.get('portfolio_value', 0)
        
        # 피크 가치 업데이트
        self.peak_value = max(self.peak_value, new_value)
        
        # 현재 drawdown 계산
        if self.peak_value == 0:
            current_drawdown = 0.0
        else:
            current_drawdown = 1.0 - (new_value / self.peak_value)
        
        # 최대 허용 손실을 초과하는 경우 패널티
        if current_drawdown > self.max_drawdown:
            return -self.penalty_scale * (current_drawdown - self.max_drawdown)
        
        # 손실이 없거나 허용 범위 내인 경우 작은 보상
        return 0.01 * (self.max_drawdown - current_drawdown)


class VolatilityControlReward(BaseRewardFunction):
    """변동성 제어 보상 함수"""
    
    def __init__(self, 
                 target_volatility: float = 0.01,  # 목표 일일 변동성 (1%)
                 window_size: int = 20,
                 scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.window_size = window_size
        self.scale = scale
        self.returns_history = []
    
    def calculate(self, current_state, next_state, action, info):
        # 현재 수익률 계산
        old_value = current_state.get('portfolio_value', 0)
        new_value = next_state.get('portfolio_value', 0)
        
        if old_value <= 0:
            return 0.0
        
        # 수익률 계산 및 히스토리에 추가
        current_return = (new_value / old_value) - 1.0
        self.returns_history.append(current_return)
        
        # 필요 이상으로 긴 히스토리는 잘라냄
        if len(self.returns_history) > self.window_size:
            self.returns_history = self.returns_history[-self.window_size:]
        
        # 충분한 데이터가 없는 경우
        if len(self.returns_history) < 2:
            return 0.0
        
        # 현재 변동성 계산
        current_volatility = np.std(self.returns_history)
        
        # 목표 변동성과의 차이에 기반한 보상/패널티
        volatility_diff = self.target_volatility - current_volatility
        
        # 변동성이 목표보다 낮으면 작은 보상, 높으면 패널티
        return self.scale * volatility_diff


class TradingCostPenalty(BaseRewardFunction):
    """거래 비용 패널티 함수"""
    
    def __init__(self, 
                 cost_scale: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.cost_scale = cost_scale
    
    def calculate(self, current_state, next_state, action, info):
        # 거래 비용 계산
        trading_cost = info.get('trading_cost', 0.0)
        
        # 비용에 대한 패널티 (음수 보상)
        return -self.cost_scale * trading_cost


class PositionDurationPenalty(BaseRewardFunction):
    """포지션 유지 기간 패널티 함수"""
    
    def __init__(self, 
                 penalty_per_day: float = 0.0001,
                 **kwargs):
        super().__init__(**kwargs)
        self.penalty_per_day = penalty_per_day
    
    def calculate(self, current_state, next_state, action, info):
        # 각 자산의 포지션 유지 기간
        position_durations = current_state.get('position_durations', [])
        
        if not position_durations:
            return 0.0
        
        # 활성 포지션(0이 아닌)에 대한 패널티 합계
        positions = current_state.get('positions', [])
        total_penalty = 0.0
        
        for position, duration in zip(positions, position_durations):
            if position != 0:  # 활성 포지션인 경우
                total_penalty -= self.penalty_per_day * duration
        
        return total_penalty


class DiversificationReward(BaseRewardFunction):
    """포트폴리오 다양화 보상 함수"""
    
    def __init__(self, 
                 target_weight: float = None,  # 목표 균등 가중치 (None이면 자동 계산)
                 scale: float = 1.0,
                 max_concentration: float = 0.5,  # 단일 자산 최대 비중
                 **kwargs):
        super().__init__(**kwargs)
        self.target_weight = target_weight
        self.scale = scale
        self.max_concentration = max_concentration
    
    def calculate(self, current_state, next_state, action, info):
        # 포트폴리오 가중치
        weights = info.get('weights', [])
        
        if not weights:
            return 0.0
        
        # 기본 목표 가중치 (균등 분배)
        if self.target_weight is None:
            self.target_weight = 1.0 / len(weights)
        
        # 다양화 측정: 가중치 편차 제곱합 (낮을수록 균등 분배)
        sum_squared_deviation = 0.0
        concentration_penalty = 0.0
        
        for weight in weights:
            # 목표 가중치와의 편차 제곱
            if weight > 0:  # 롱 포지션만 고려
                sum_squared_deviation += (weight - self.target_weight) ** 2
            
            # 과도한 집중 투자 패널티
            if weight > self.max_concentration:
                concentration_penalty -= 0.5 * (weight - self.max_concentration)
        
        # 다양화 점수 (낮은 편차 = 높은 점수)
        diversification_score = 1.0 / (1.0 + sum_squared_deviation)
        
        return self.scale * (diversification_score + concentration_penalty)


class CompositeReward(BaseRewardFunction):
    """복합 보상 함수: 여러 보상 함수의 가중 합"""
    
    def __init__(self, 
                 reward_functions: List[Tuple[BaseRewardFunction, float]], 
                 **kwargs):
        """
        복합 보상 함수 초기화
        
        Args:
            reward_functions: (보상함수, 가중치) 튜플의 리스트
            **kwargs: 추가 매개변수
        """
        super().__init__(**kwargs)
        self.reward_functions = reward_functions
    
    def calculate(self, current_state, next_state, action, info):
        total_reward = 0.0
        
        # 각 보상 함수에 대한 가중 합계 계산
        for reward_func, weight in self.reward_functions:
            reward = reward_func.calculate(current_state, next_state, action, info)
            total_reward += weight * reward
        
        return total_reward


class RewardFactory:
    """다양한 유형의 보상 함수를 생성하는 팩토리 클래스"""
    
    @staticmethod
    def create_reward(reward_type: str, **kwargs) -> BaseRewardFunction:
        """
        보상 함수 객체 생성
        
        Args:
            reward_type: 보상 함수 유형
            **kwargs: 보상 함수별 매개변수
            
        Returns:
            BaseRewardFunction: 생성된 보상 함수
        """
        reward_map = {
            'return': ReturnReward,
            'sharpe': SharpeReward,
            'sortino': SortinoReward,
            'drawdown': DrawdownControlReward,
            'volatility': VolatilityControlReward,
            'trading_cost': TradingCostPenalty,
            'position_duration': PositionDurationPenalty,
            'diversification': DiversificationReward
        }
        
        if reward_type not in reward_map:
            raise ValueError(f"지원되지 않는 보상 함수 유형: {reward_type}")
        
        return reward_map[reward_type](**kwargs)
    
    @staticmethod
    def create_composite_reward(reward_configs: List[Dict]) -> CompositeReward:
        """
        복합 보상 함수 생성
        
        Args:
            reward_configs: 각 보상 함수의 구성 설정 리스트
            
        Returns:
            CompositeReward: 생성된 복합 보상 함수
        """
        reward_functions = []
        
        for config in reward_configs:
            reward_type = config.pop('type')
            weight = config.pop('weight', 1.0)
            reward_func = RewardFactory.create_reward(reward_type, **config)
            reward_functions.append((reward_func, weight))
        
        return CompositeReward(reward_functions=reward_functions)


# 사용 예시
if __name__ == "__main__":
    # 간단한 복합 보상 함수 생성 예제
    reward_configs = [
        {'type': 'return', 'scale': 1.0, 'weight': 1.0},
        {'type': 'sharpe', 'risk_free_rate': 0.02, 'window_size': 30, 'weight': 0.5},
        {'type': 'drawdown', 'max_drawdown': 0.05, 'weight': 1.0},
        {'type': 'trading_cost', 'cost_scale': 1.0, 'weight': 0.2},
        {'type': 'diversification', 'weight': 0.3}
    ]
    
    # 복합 보상 함수 생성
    reward_function = RewardFactory.create_composite_reward(reward_configs)
    
    # 테스트용 상태 및 정보
    current_state = {
        'portfolio_value': 100000.0,
        'positions': [1, -1, 0],  # 3개 자산에 대한 포지션
        'position_durations': [5, 2, 0]  # 포지션 유지 기간
    }
    
    next_state = {
        'portfolio_value': 101000.0,  # 1% 수익
        'positions': [1, -1, 0],
        'position_durations': [6, 3, 0]
    }
    
    action = np.array([0.5, -0.3, 0.0])  # 각 자산에 대한 목표 가중치
    
    info = {
        'weights': [0.5, 0.3, 0.0],  # 포트폴리오 가중치
        'trading_cost': 50.0,  # 거래 비용
    }
    
    # 보상 계산
    reward = reward_function.calculate(current_state, next_state, action, info)
    print(f"계산된 복합 보상: {reward:.6f}")
