import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Union, Optional


class MultiAssetTradingEnv(gym.Env):
    """
    다중 자산 트레이딩을 위한 강화학습 환경
    
    Features:
    - 다중 자산 동시 트레이딩 지원
    - 포트폴리오 가중치 최적화
    - 자산 간 상관관계 모델링
    - 리밸런싱 비용 및 제약 모델링
    """
    
    def __init__(
        self, 
        asset_data: Dict[str, pd.DataFrame], 
        initial_capital: float = 100000.0,
        trading_cost: float = 0.0005,
        slippage: float = 0.0001,
        risk_free_rate: float = 0.02,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.02,
        max_asset_weight: float = 0.5,  # 단일 자산 최대 비중
        correlation_lookback: int = 30,  # 상관관계 계산을 위한 룩백 기간
        rebalancing_cost: float = 0.0002,  # 리밸런싱 추가 비용
        min_trading_amount: float = 1000.0,  # 최소 거래 금액
    ):
        super().__init__()
        
        # 자산 데이터 초기화
        self.asset_names = list(asset_data.keys())
        self.n_assets = len(self.asset_names)
        self.asset_data = asset_data
        
        # 모든 자산 데이터가 동일한 기간을 다루는지 확인
        self._validate_data_alignment()
        
        # 첫 번째 자산의 길이를 기준으로 설정
        self.n_steps = len(next(iter(asset_data.values())))
        
        # 환경 파라미터 설정
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252  # 일별 무위험 수익률
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_asset_weight = max_asset_weight
        self.correlation_lookback = correlation_lookback
        self.rebalancing_cost = rebalancing_cost
        self.min_trading_amount = min_trading_amount
        
        # 행동 공간: 각 자산별 포지션 비율 (-1 ~ 1, 음수는 숏 포지션)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # 상태 공간: 모든 자산의 기술적 지표 + 현재 포트폴리오 상태 정보
        # 각 자산당 기술적 지표 12개 + 포지션 + 전체 성과 지표
        n_features_per_asset = 12
        state_dim = (self.n_assets * n_features_per_asset) + (self.n_assets + 3)  # 포지션 + 포트폴리오 지표 3개
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 초기화
        self.reset()
        
    def _validate_data_alignment(self):
        """모든 자산 데이터가 동일한 기간과 길이를 가지는지 확인"""
        if not self.asset_data:
            raise ValueError("자산 데이터가 제공되지 않았습니다.")
            
        first_asset_data = next(iter(self.asset_data.values()))
        first_asset_dates = first_asset_data.index
        
        for asset_name, asset_df in self.asset_data.items():
            if len(asset_df) != len(first_asset_data):
                raise ValueError(f"자산 '{asset_name}'의 데이터 길이가 다릅니다.")
            
            if not asset_df.index.equals(first_asset_dates):
                raise ValueError(f"자산 '{asset_name}'의 날짜 인덱스가 일치하지 않습니다.")
                
            # 필수 기술적 지표 검증
            required_columns = [
                'RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm',
                'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
                'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm', 'Close'
            ]
            
            for col in required_columns:
                if col not in asset_df.columns:
                    raise ValueError(f"자산 '{asset_name}'에 필수 열 '{col}'이 없습니다.")
    
    def reset(self):
        """환경을 초기 상태로 재설정"""
        self.index = 0
        
        # 초기 자산별 포지션 및 가치 초기화
        self.positions = np.zeros(self.n_assets)  # 각 자산의 포지션 (-1: 숏, 0: 없음, 1: 롱)
        self.holdings = np.zeros(self.n_assets)   # 각 자산의 보유량
        self.asset_values = np.zeros(self.n_assets)  # 각 자산의 현재 가치
        
        # 포트폴리오 상태 초기화
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.portfolio_values = [self.initial_capital]
        self.portfolio_weights = np.zeros(self.n_assets)  # 각 자산의 포트폴리오 내 비중
        
        # 거래 이력 및 성과 지표 초기화
        self.trades = []
        self.daily_returns = []
        self.asset_returns = {asset: [] for asset in self.asset_names}
        self.position_durations = np.zeros(self.n_assets)
        self.entry_prices = np.zeros(self.n_assets)
        
        # 상관관계 매트릭스 초기화
        self.correlation_matrix = np.eye(self.n_assets)
        
        # 현재 가격 정보 가져오기
        self.current_prices = self._get_current_prices()
        
        return self._get_observation()
        
    def _get_current_prices(self) -> np.ndarray:
        """현재 시점의 모든 자산 가격을 가져옴"""
        prices = np.zeros(self.n_assets)
        
        for i, asset_name in enumerate(self.asset_names):
            prices[i] = float(self.asset_data[asset_name].loc[self.asset_data[asset_name].index[self.index], 'Close'])
            
        return prices
    
    def _update_correlation_matrix(self):
        """자산 간 상관관계 매트릭스 업데이트"""
        if self.index < self.correlation_lookback:
            # 충분한 데이터가 없는 경우 단위 행렬 사용
            return
        
        # 각 자산의 종가 데이터 수집
        price_data = {}
        start_idx = max(0, self.index - self.correlation_lookback)
        
        for i, asset_name in enumerate(self.asset_names):
            asset_df = self.asset_data[asset_name]
            price_data[asset_name] = asset_df.iloc[start_idx:self.index+1]['Close'].values
        
        # 수익률 계산
        returns_data = {}
        for asset_name, prices in price_data.items():
            returns_data[asset_name] = np.diff(prices) / prices[:-1]
        
        # 상관관계 행렬 계산
        returns_df = pd.DataFrame(returns_data)
        self.correlation_matrix = returns_df.corr().fillna(0).values
    
    def _get_observation(self):
        """현재 상태 관측값 반환"""
        # 모든 자산의 기술적 지표 수집
        technical_indicators = []
        
        for asset_name in self.asset_names:
            asset_df = self.asset_data[asset_name]
            row = asset_df.iloc[self.index]
            
            indicators = [
                float(row['RSI_norm']),
                float(row['ForceIndex2_norm']),
                float(row['%K_norm']),
                float(row['%D_norm']),
                float(row['MACD_norm']),
                float(row['MACDSignal_norm']),
                float(row['BBWidth_norm']),
                float(row['ATR_norm']),
                float(row['VPT_norm']),
                float(row['VPT_MA_norm']),
                float(row['OBV_norm']),
                float(row['ROC_norm'])
            ]
            
            technical_indicators.extend(indicators)
        
        # 현재 포트폴리오 상태 추가
        portfolio_info = np.concatenate([
            self.positions,  # 각 자산의 현재 포지션
            [
                self.cash / self.portfolio_value,  # 현금 비율
                self._calculate_portfolio_return(),  # 전체 포트폴리오 수익률
                self._calculate_portfolio_volatility()  # 포트폴리오 변동성
            ]
        ])
        
        # 최종 관측값
        observation = np.concatenate([technical_indicators, portfolio_info])
        
        return observation.astype(np.float32)
    
    def _calculate_portfolio_return(self):
        """포트폴리오 수익률 계산"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        return (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
    
    def _calculate_portfolio_volatility(self):
        """포트폴리오 변동성 계산"""
        if len(self.daily_returns) < 2:
            return 0.0
        
        return np.std(self.daily_returns[-20:]) if len(self.daily_returns) >= 20 else np.std(self.daily_returns)
    
    def _calculate_reward(self, old_value, new_value, actions):
        """현재 스텝에 대한 보상 계산"""
        # 기본 수익률
        returns = (new_value / old_value) - 1
        
        # Sharpe Ratio 구성요소
        if len(self.daily_returns) > 0:
            returns_std = np.std(self.daily_returns) + 1e-9
            sharpe = (returns - self.risk_free_rate) / returns_std
        else:
            sharpe = 0
        
        # 포지션 유지 페널티 (각 자산에 대해)
        holding_penalty = 0
        for i in range(self.n_assets):
            if self.positions[i] != 0:
                holding_penalty -= 0.0001 * self.position_durations[i]
        
        # 과도한 거래 페널티
        trading_penalty = -0.0002 * np.sum(np.abs(actions))
        
        # 과도한 집중 투자 페널티
        concentration_penalty = 0
        for weight in self.portfolio_weights:
            if weight > self.max_asset_weight:
                concentration_penalty -= 0.0003 * (weight - self.max_asset_weight)
        
        # 상관관계 패널티: 높은 상관관계를 가진 자산들에 집중 투자하는 것 방지
        correlation_penalty = 0
        for i in range(self.n_assets):
            if self.positions[i] == 0:
                continue
                
            for j in range(i+1, self.n_assets):
                if self.positions[j] == 0:
                    continue
                    
                # 두 자산이 모두 같은 방향으로 포지션을 가지고 있고, 상관관계가 높은 경우
                if self.positions[i] * self.positions[j] > 0 and self.correlation_matrix[i, j] > 0.7:
                    correlation_penalty -= 0.0001 * self.correlation_matrix[i, j]
        
        # 최종 보상
        reward = returns + 0.1 * sharpe + holding_penalty + trading_penalty + concentration_penalty + correlation_penalty
        
        return reward
        
    def step(self, actions):
        """한 스텝 실행: 행동 수행 및 다음 상태, 보상, 종료 여부 반환"""
        # 이전 포트폴리오 가치 저장
        old_portfolio_value = self.portfolio_value
        
        # 현재 가격 업데이트
        self.current_prices = self._get_current_prices()
        
        # 상관관계 매트릭스 업데이트
        self._update_correlation_matrix()
        
        # 행동 처리: 목표 포트폴리오 가중치로 변환
        target_weights = self._normalize_weights(actions)
        
        # 현재 포트폴리오 가중치 계산
        current_asset_values = self.holdings * self.current_prices
        self.portfolio_value = self.cash + np.sum(current_asset_values)
        current_weights = current_asset_values / self.portfolio_value
        
        # 리밸런싱 실행
        self._rebalance_portfolio(current_weights, target_weights)
        
        # 스톱로스 체크 및 적용
        self._check_stop_loss()
        
        # 다음 날로 이동
        self.index += 1
        
        # 포지션 유지 기간 업데이트
        for i in range(self.n_assets):
            if self.positions[i] != 0:
                self.position_durations[i] += 1
        
        # 다음 날 가격으로 포트폴리오 가치 업데이트
        if self.index < self.n_steps:
            next_prices = self._get_current_prices()
            asset_values = self.holdings * next_prices
            new_portfolio_value = self.cash + np.sum(asset_values)
            
            # 포트폴리오 가치 및 수익률 업데이트
            self.portfolio_values.append(new_portfolio_value)
            daily_return = (new_portfolio_value / old_portfolio_value) - 1
            self.daily_returns.append(daily_return)
            
            # 포트폴리오 가치 업데이트
            self.portfolio_value = new_portfolio_value
            
            # 포트폴리오 가중치 업데이트
            self.portfolio_weights = asset_values / (new_portfolio_value + 1e-9)
        
        # 보상 계산
        reward = self._calculate_reward(old_portfolio_value, self.portfolio_value, actions)
        
        # 에피소드 종료 여부 확인
        done = self.index >= self.n_steps - 1
        
        return self._get_observation(), reward, done, {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'weights': self.portfolio_weights.copy(),
            'cash_ratio': self.cash / self.portfolio_value,
            'trades': self.trades.copy()
        }
    
    def _normalize_weights(self, actions):
        """행동을 정규화된 포트폴리오 가중치로 변환"""
        # 행동을 [-1, 1] 범위에서 포트폴리오 가중치로 변환
        # 음수값은 숏 포지션, 양수값은 롱 포지션을 의미
        
        # 절대값 합계로 정규화
        abs_actions = np.abs(actions)
        total_abs_weight = np.sum(abs_actions)
        
        if total_abs_weight > 1e-6:
            # 레버리지를 max_position_size 이내로 제한
            weights = actions * self.max_position_size / max(1.0, total_abs_weight)
        else:
            weights = np.zeros_like(actions)
            
        return weights
    
    def _rebalance_portfolio(self, current_weights, target_weights):
        """포트폴리오를 목표 가중치에 맞게 리밸런싱"""
        for i, (curr_w, target_w) in enumerate(zip(current_weights, target_weights)):
            # 가중치 차이 계산
            weight_diff = target_w - curr_w
            
            # 무시할 수 있을 정도로 작은 변화는 건너뜀
            if abs(weight_diff) < 0.01:
                continue
            
            # 거래 금액 계산
            trade_value = weight_diff * self.portfolio_value
            trade_price = self.current_prices[i] * (1 + self.slippage * np.sign(weight_diff))
            
            # 최소 거래 금액 체크
            if abs(trade_value) < self.min_trading_amount:
                continue
            
            # 매수(롱) 또는 매도(숏)
            n_shares = trade_value / trade_price
            
            # 거래 비용 계산
            transaction_cost = abs(trade_value) * (self.trading_cost + 
                                                  (self.rebalancing_cost if self.positions[i] != 0 else 0))
            
            # 현금 및 보유량 업데이트
            self.cash -= (trade_value + transaction_cost)
            self.holdings[i] += n_shares
            
            # 포지션 상태 업데이트
            self.positions[i] = np.sign(self.holdings[i])
            
            # 진입 가격 기록 (포지션이 변경된 경우)
            if (self.positions[i] != 0 and self.position_durations[i] == 0) or (np.sign(n_shares) != self.positions[i]):
                self.entry_prices[i] = trade_price
                self.position_durations[i] = 0
            
            # 거래 기록
            self.trades.append({
                'asset': self.asset_names[i],
                'index': self.index,
                'type': 'buy' if n_shares > 0 else 'sell',
                'price': trade_price,
                'shares': abs(n_shares),
                'value': abs(trade_value),
                'cost': transaction_cost
            })
    
    def _check_stop_loss(self):
        """각 자산에 대한 스톱로스 체크 및 적용"""
        for i in range(self.n_assets):
            if self.positions[i] == 0:
                continue
                
            current_price = self.current_prices[i]
            entry_price = self.entry_prices[i]
            
            # 가격 변화율 계산
            price_change = (current_price - entry_price) / entry_price
            
            # 스톱로스 조건 확인
            if ((self.positions[i] > 0 and price_change < -self.stop_loss_pct) or
                (self.positions[i] < 0 and price_change > self.stop_loss_pct)):
                
                # 포지션 청산
                trade_price = current_price * (1 - self.slippage * np.sign(self.positions[i]))
                trade_value = self.holdings[i] * trade_price
                transaction_cost = abs(trade_value) * self.trading_cost
                
                # 현금 업데이트
                self.cash += trade_value - transaction_cost
                
                # 거래 기록
                self.trades.append({
                    'asset': self.asset_names[i],
                    'index': self.index,
                    'type': 'stop_loss_sell' if self.positions[i] > 0 else 'stop_loss_buy',
                    'price': trade_price,
                    'shares': abs(self.holdings[i]),
                    'value': abs(trade_value),
                    'cost': transaction_cost
                })
                
                # 포지션 초기화
                self.holdings[i] = 0
                self.positions[i] = 0
                self.position_durations[i] = 0


# 사용 예시
if __name__ == "__main__":
    # 예시: 여러 자산의 데이터 준비
    import src.data.data_processor as dp
    
    # 여러 주식 데이터 다운로드 및 처리
    symbols = ['SPY', 'QQQ', 'GLD']
    asset_data = {}
    
    for symbol in symbols:
        asset_data[symbol] = dp.process_data(symbol, start_date='2020-01-01', end_date='2021-01-01')
    
    # 다중 자산 트레이딩 환경 생성
    env = MultiAssetTradingEnv(
        asset_data=asset_data,
        initial_capital=100000.0,
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=1.5,  # 최대 1.5배 레버리지 허용
        stop_loss_pct=0.02
    )
    
    # 간단한 테스트
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 랜덤 액션 (각 자산에 대한 -1 ~ 1 사이의 가중치)
        action = env.action_space.sample()
        
        # 환경 스텝 실행
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"최종 포트폴리오 가치: ${info['portfolio_value']:.2f}")
            print(f"총 보상: {total_reward:.2f}")
            print(f"거래 횟수: {len(info['trades'])}")
