import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Optional, Union, Tuple, Any

class EnhancedTradingEnv(gym.Env):
    """
    향상된 금융 트레이딩 환경
    
    기존 TradingEnv를 확장하여 더 풍부한 상태 표현, 유연한 행동 공간,
    사실적인 거래 시뮬레이션, 그리고 다양한 보상 함수를 제공합니다.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        trading_cost: float = 0.0005,
        slippage: float = 0.0001,
        risk_free_rate: float = 0.02,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.02,
        window_size: int = 20,
        reward_type: str = 'sharpe',
        action_type: str = 'discrete',
        feature_columns: Optional[List[str]] = None,
        macro_columns: Optional[List[str]] = None,
        news_columns: Optional[List[str]] = None,
        use_risk_adjustment: bool = True,
        use_market_indicators: bool = True,
        max_drawdown_limit: Optional[float] = None,
        transaction_limit: Optional[int] = None
    ):
        """
        Args:
            data: 주가 및 지표 데이터가 포함된 데이터프레임
            initial_capital: 초기 자본
            trading_cost: 거래 비용 (기본값: 0.05%)
            slippage: 슬리피지 (기본값: 0.01%)
            risk_free_rate: 무위험 이자율 (연간, 기본값: 2%)
            max_position_size: 최대 포지션 크기 (1.0 = 전체 자본)
            stop_loss_pct: 손절매 비율 (기본값: 2%)
            window_size: 관찰 윈도우 크기
            reward_type: 보상 함수 유형 ('simple', 'sharpe', 'sortino', 'calmar')
            action_type: 행동 공간 유형 ('discrete', 'continuous', 'hybrid')
            feature_columns: 사용할 특성 컬럼 (None이면 모든 표준화된 특성 사용)
            macro_columns: 사용할 거시경제 특성 컬럼
            news_columns: 사용할 뉴스 감성 특성 컬럼
            use_risk_adjustment: 위험 조정 보상 사용 여부
            use_market_indicators: 시장 상태 지표 사용 여부
            max_drawdown_limit: 최대 손실 제한 (None이면 제한 없음)
            transaction_limit: 최대 거래 횟수 제한 (None이면 제한 없음)
        """
        super().__init__()
        
        # 기본 환경 설정
        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate / 252  # 일일 무위험 이자율
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.reward_type = reward_type
        self.action_type = action_type
        self.use_risk_adjustment = use_risk_adjustment
        self.use_market_indicators = use_market_indicators
        self.max_drawdown_limit = max_drawdown_limit
        self.transaction_limit = transaction_limit
        
        # 특성 컬럼 선택
        self._set_feature_columns(feature_columns, macro_columns, news_columns)
        
        # 관찰 및 행동 공간 정의
        self._define_spaces()
        
        # 초기화
        self.reset()
        
    def _set_feature_columns(
        self, 
        feature_columns: Optional[List[str]], 
        macro_columns: Optional[List[str]], 
        news_columns: Optional[List[str]]
    ):
        """특성 컬럼 설정"""
        # 정규화된 특성 컬럼 (suffix가 _norm인 컬럼들)
        norm_columns = [col for col in self.data.columns if col.endswith('_norm')]
        
        # 기본 기술적 지표 (기존 환경과 동일)
        self.basic_features = [
            'RSI_norm', 'ForceIndex2_norm', '%K_norm', '%D_norm', 
            'MACD_norm', 'MACDSignal_norm', 'BBWidth_norm', 'ATR_norm',
            'VPT_norm', 'VPT_MA_norm', 'OBV_norm', 'ROC_norm'
        ]
        
        # 전체 특성에서 사용 가능한 컬럼 필터링
        available_basic = [col for col in self.basic_features if col in self.data.columns]
        
        # 특성 컬럼 설정
        if feature_columns is not None:
            # 사용자 지정 특성 사용
            available_features = [col for col in feature_columns if col in self.data.columns]
            if len(available_features) == 0:
                # 사용 가능한 특성이 없으면 기본 특성 사용
                self.feature_columns = available_basic
            else:
                self.feature_columns = available_features
        else:
            # 고급 지표가 있으면 사용
            advanced_features = [
                col for col in norm_columns 
                if col not in available_basic and not col.startswith('Macro_') and not col.startswith('News_')
            ]
            self.feature_columns = available_basic + advanced_features
        
        # 거시경제 특성 설정
        if macro_columns is not None:
            self.macro_columns = [col for col in macro_columns if col in self.data.columns]
        else:
            self.macro_columns = [col for col in norm_columns if col.startswith('Macro_')]
        
        # 뉴스 감성 특성 설정
        if news_columns is not None:
            self.news_columns = [col for col in news_columns if col in self.data.columns]
        else:
            self.news_columns = [col for col in norm_columns if col.startswith('News_')]
        
        # 모든 특성 컬럼 통합
        self.all_feature_columns = self.feature_columns + self.macro_columns + self.news_columns
        
        # 가격 및 볼륨 데이터
        self.price_columns = ['Open', 'High', 'Low', 'Close']
        self.volume_column = 'Volume'
        
    def _define_spaces(self):
        """관찰 및 행동 공간 정의"""
        # 관찰 공간 차원 계산
        feature_dim = len(self.all_feature_columns)
        position_info_dim = 3  # 포지션 상태, 보유 기간, 진입 가격 비율
        market_state_dim = 5 if self.use_market_indicators else 0  # 시장 상태 지표
        
        obs_dim = feature_dim + position_info_dim + market_state_dim
        
        # 관찰 공간 정의 (상태 표현)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 행동 공간 정의
        if self.action_type == 'discrete':
            # 이산 행동: 0=보유, 1=매수, 2=매도
            self.action_space = spaces.Discrete(3)
        elif self.action_type == 'continuous':
            # 연속 행동: 포지션 크기 (-1.0 ~ 1.0, 음수=공매도)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:  # 'hybrid'
            # 하이브리드 행동: [행동 유형, 포지션 크기]
            self.action_space = spaces.Tuple((
                spaces.Discrete(3),  # 행동 유형
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # 포지션 크기
            ))
    
    def reset(self):
        """환경을 초기 상태로 리셋"""
        # 인덱스 초기화 (window_size 이후부터 시작)
        self.index = self.window_size
        
        # 포트폴리오 상태 초기화
        self.position = 0  # 0=무포지션, 1=롱, -1=숏
        self.cash = self.initial_capital
        self.holdings = 0
        self.entry_price = 0
        self.last_price = float(self.data.loc[self.index, 'Close'])
        self.portfolio_values = [self.initial_capital]
        self.position_duration = 0
        
        # 거래 이력
        self.trades = []
        
        # 성과 지표
        self.daily_returns = []
        self.transaction_count = 0
        self.max_drawdown = 0
        
        # 현재 드로다운 추적
        self.peak_value = self.initial_capital
        self.current_drawdown = 0
        
        # 첫 번째 관찰 반환
        return self._get_observation()
    
    def _get_observation(self):
        """현재 상태 관찰 반환"""
        # 현재 시점의 특성 값 추출
        features = np.array([
            float(self.data.loc[self.index, col]) 
            for col in self.all_feature_columns
        ], dtype=np.float32)
        
        # 포지션 정보
        position_info = np.array([
            float(self.position),  # 현재 포지션 (1=롱, 0=무포지션, -1=숏)
            float(self.position_duration),  # 포지션 보유 기간
            float(self.entry_price / self.last_price - 1.0) if self.position != 0 else 0.0  # 진입가 대비 현재가 비율
        ], dtype=np.float32)
        
        # 시장 상태 지표 (선택적)
        if self.use_market_indicators:
            current_value = self.cash + (self.holdings * self.last_price)
            market_state = np.array([
                float(self.portfolio_values[-1] / self.initial_capital - 1.0),  # 총 수익률
                float(self.current_drawdown),  # 현재 드로다운
                float(sum(r > 0 for r in self.daily_returns[-20:]) / max(1, len(self.daily_returns[-20:]))),  # 승률
                float(self.transaction_count / max(1, self.index - self.window_size)),  # 거래 빈도
                float(np.std(self.daily_returns[-20:]) if len(self.daily_returns) >= 20 else 0.0)  # 변동성
            ], dtype=np.float32)
            
            # 관찰 벡터 결합
            observation = np.concatenate([features, position_info, market_state])
        else:
            # 시장 상태 없이 결합
            observation = np.concatenate([features, position_info])
        
        return observation
    
    def _calculate_reward(self, old_value, new_value, action):
        """현재 스텝의 보상 계산"""
        # 기본 수익률 계산
        returns = (new_value / old_value) - 1.0
        
        if self.reward_type == 'simple':
            # 단순 수익률
            reward = returns
        
        elif self.reward_type == 'sharpe':
            # Sharpe 비율 기반 보상
            if len(self.daily_returns) > 0:
                returns_std = np.std(self.daily_returns) + 1e-9
                sharpe = (returns - self.risk_free_rate) / returns_std
                reward = returns + (0.1 * sharpe if self.use_risk_adjustment else 0)
            else:
                reward = returns
        
        elif self.reward_type == 'sortino':
            # Sortino 비율 기반 보상 (하방 위험만 고려)
            if len(self.daily_returns) > 0:
                # 하방 수익률만 필터링
                downside_returns = [r for r in self.daily_returns if r < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns) + 1e-9
                    sortino = (returns - self.risk_free_rate) / downside_std
                    reward = returns + (0.1 * sortino if self.use_risk_adjustment else 0)
                else:
                    reward = returns
            else:
                reward = returns
        
        elif self.reward_type == 'calmar':
            # Calmar 비율 기반 보상 (최대 손실 대비 수익률)
            if self.max_drawdown > 0:
                avg_return = np.mean(self.daily_returns) if len(self.daily_returns) > 0 else returns
                calmar = avg_return / self.max_drawdown
                reward = returns + (0.1 * calmar if self.use_risk_adjustment else 0)
            else:
                reward = returns
                
        else:
            # 기본값: 단순 수익률
            reward = returns
        
        # 페널티 추가
        # 1. 포지션 유지 페널티 (장기 보유 억제)
        holding_penalty = -0.0001 * self.position_duration if self.position != 0 else 0
        
        # 2. 과도한 거래 페널티
        trading_penalty = -0.0002 if action != 0 else 0
        
        # 3. 드로다운 페널티 (최대 손실 제한에 가까울 때)
        drawdown_penalty = 0
        if self.max_drawdown_limit is not None and self.current_drawdown > 0:
            drawdown_ratio = self.current_drawdown / self.max_drawdown_limit
            if drawdown_ratio > 0.7:  # 최대 손실의 70% 이상
                drawdown_penalty = -0.001 * (drawdown_ratio - 0.7) / 0.3
        
        # 최종 보상 = 기본 보상 + 페널티들
        final_reward = reward + holding_penalty + trading_penalty + drawdown_penalty
        
        return final_reward
    
    def _process_action(self, action):
        """행동 처리 및 실행"""
        # 현재 가격 가져오기
        current_price = float(self.data.loc[self.index, 'Close'])
        
        # 행동 처리
        if self.action_type == 'discrete':
            # 이산 행동 처리
            action_type = action
            size = self.max_position_size
        elif self.action_type == 'continuous':
            # 연속 행동 처리
            if action > 0.05:  # 롱 포지션 (임계값 적용)
                action_type = 1
                size = float(action) * self.max_position_size
            elif action < -0.05:  # 숏 포지션 (임계값 적용)
                action_type = 2
                size = float(-action) * self.max_position_size
            else:  # 관망(-0.05 ~ 0.05)
                action_type = 0
                size = 0
        else:  # 'hybrid'
            # 하이브리드 행동 처리
            action_type, size_action = action
            size = float(size_action[0]) * self.max_position_size
        
        # 슬리피지 적용
        if action_type == 1:  # 매수
            exec_price = current_price * (1 + self.slippage)
        elif action_type == 2:  # 매도
            exec_price = current_price * (1 - self.slippage)
        else:  # 보유
            exec_price = current_price
        
        return action_type, size, exec_price
    
    def step(self, action):
        """환경에서 한 스텝 실행"""
        # 이전 포트폴리오 가치 계산
        current_price = float(self.data.loc[self.index, 'Close'])
        old_portfolio_value = self.cash + (self.holdings * current_price)
        
        # 스톱로스 체크
        stop_loss_triggered = False
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            if (self.position == 1 and price_change < -self.stop_loss_pct) or \
               (self.position == -1 and price_change > self.stop_loss_pct):
                # 스톱로스 발동 - 포지션 청산
                action = 2 if self.position == 1 else 1
                stop_loss_triggered = True
        
        # 최대 손실 제한 체크
        if self.max_drawdown_limit is not None and self.current_drawdown > self.max_drawdown_limit:
            # 최대 손실 초과 - 모든 포지션 청산
            if self.position == 1:
                action = 2  # 매도
            elif self.position == -1:
                action = 1  # 매수(청산)
        
        # 최대 거래 횟수 제한 체크
        if self.transaction_limit is not None and self.transaction_count >= self.transaction_limit:
            # 거래 횟수 초과 - 행동 제한 (청산만 허용)
            if (self.position == 1 and action != 2) or (self.position == -1 and action != 1):
                action = 0  # 보유만 허용
        
        # 행동 처리
        action_type, size, exec_price = self._process_action(action)
        
        # 거래 실행
        old_position = self.position
        
        if action_type == 1:  # 매수
            if self.position == 0:  # 무포지션 -> 롱
                self.position = 1
                self.holdings = size
                cost = exec_price * self.holdings
                self.cash -= cost * (1 + self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('buy', self.index, exec_price, self.holdings))
                self.position_duration = 0
                self.transaction_count += 1

            elif self.position == -1:  # 숏 -> 롱
                # 숏 포지션 청산
                profit = (self.entry_price - exec_price) * abs(self.holdings)
                self.cash += profit
                self.cash -= abs(self.holdings) * exec_price * self.trading_cost
                self.trades.append(('close_short', self.index, exec_price, abs(self.holdings)))
                self.transaction_count += 1

                if not stop_loss_triggered:
                    # 롱 포지션 진입
                    self.position = 1
                    self.holdings = size
                    cost = exec_price * self.holdings
                    self.cash -= cost * (1 + self.trading_cost)
                    self.entry_price = exec_price
                    self.trades.append(('buy', self.index, exec_price, self.holdings))
                    self.position_duration = 0
                    self.transaction_count += 1
                else:
                    # 스톱로스인 경우 포지션만 청산
                    self.position = 0
                    self.holdings = 0

        elif action_type == 2:  # 매도
            if self.position == 0:  # 무포지션 -> 숏
                self.position = -1
                self.holdings = -size
                proceed = exec_price * abs(self.holdings)
                self.cash += proceed * (1 - self.trading_cost)
                self.entry_price = exec_price
                self.trades.append(('short', self.index, exec_price, abs(self.holdings)))
                self.position_duration = 0
                self.transaction_count += 1

            elif self.position == 1:  # 롱 -> 숏
                # 롱 포지션 청산
                profit = (exec_price - self.entry_price) * self.holdings
                self.cash += profit
                self.cash += (self.holdings * exec_price) * (1 - self.trading_cost)
                self.trades.append(('close_long', self.index, exec_price, self.holdings))
                self.transaction_count += 1

                if not stop_loss_triggered:
                    # 숏 포지션 진입
                    self.position = -1
                    self.holdings = -size
                    proceed = exec_price * abs(self.holdings)
                    self.cash += proceed * (1 - self.trading_cost)
                    self.entry_price = exec_price
                    self.trades.append(('short', self.index, exec_price, abs(self.holdings)))
                    self.position_duration = 0
                    self.transaction_count += 1
                else:
                    # 스톱로스인 경우 포지션만 청산
                    self.position = 0
                    self.holdings = 0
        
        # 시간 진행
        self.index += 1
        if self.index >= len(self.data):
            # 데이터 끝에 도달
            return self._get_observation(), 0, True, {"message": "데이터 끝에 도달했습니다."}
        
        # 포지션 유지 기간 업데이트 (새 포지션에서는 증가하지 않음)
        if self.position != 0 and self.position == old_position:
            self.position_duration += 1
        
        # 새 포트폴리오 가치 계산
        self.last_price = float(self.data.loc[self.index, 'Close'])
        new_portfolio_value = self.cash + (self.holdings * self.last_price)
        self.portfolio_values.append(new_portfolio_value)
        
        # 일간 수익률 계산
        daily_return = (new_portfolio_value / old_portfolio_value) - 1.0
        self.daily_returns.append(daily_return)
        
        # 드로다운 계산
        if new_portfolio_value > self.peak_value:
            self.peak_value = new_portfolio_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = 1.0 - (new_portfolio_value / self.peak_value)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # 보상 계산
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action_type)
        
        # 에피소드 종료 여부 확인
        done = self.index >= len(self.data) - 1
        
        # 추가 정보
        info = {
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'return': daily_return,
            'trades': self.trades,
            'transaction_count': self.transaction_count,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown
        }
        
        if stop_loss_triggered:
            info['stop_loss'] = True
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """환경 시각화 (간단한 콘솔 출력)"""
        if mode != 'human':
            return
        
        current_price = self.data.loc[self.index, 'Close']
        portfolio_value = self.cash + (self.holdings * current_price)
        total_return = (portfolio_value / self.initial_capital - 1.0) * 100
        
        print(f"날짜: {self.data.index[self.index] if hasattr(self.data.index, '__getitem__') else self.index}")
        print(f"가격: ${current_price:.2f}")
        print(f"현금: ${self.cash:.2f}")
        print(f"보유량: {self.holdings:.4f}")
        print(f"포트폴리오 가치: ${portfolio_value:.2f}")
        print(f"수익률: {total_return:.2f}%")
        print(f"포지션: {'롱' if self.position == 1 else '숏' if self.position == -1 else '없음'}")
        print(f"거래 횟수: {self.transaction_count}")
        print(f"최대 손실: {self.max_drawdown*100:.2f}%")
        print("-" * 50)
    
    def get_performance_metrics(self):
        """성과 지표 계산 및 반환"""
        # 최종 포트폴리오 가치
        final_price = float(self.data.loc[min(self.index, len(self.data)-1), 'Close'])
        final_value = self.cash + (self.holdings * final_price)
        
        # 총 수익률
        total_return = (final_value / self.initial_capital) - 1.0
        
        # 연율화 수익률 (252 거래일 기준)
        n_days = len(self.daily_returns)
        annualized_return = ((1 + total_return) ** (252 / max(1, n_days))) - 1.0 if n_days > 0 else 0.0
        
        # 승률
        win_rate = sum(r > 0 for r in self.daily_returns) / max(1, len(self.daily_returns))
        
        # 변동성
        volatility = np.std(self.daily_returns) * np.sqrt(252) if len(self.daily_returns) > 0 else 0.0
        
        # Sharpe 비율
        sharpe = (annualized_return - self.risk_free_rate * 252) / (volatility + 1e-9)
        
        # Sortino 비율 (하방 위험만 고려)
        downside_returns = [r for r in self.daily_returns if r < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
        sortino = (annualized_return - self.risk_free_rate * 252) / downside_volatility
        
        # Calmar 비율 (최대 손실 대비 수익률)
        calmar = annualized_return / (self.max_drawdown + 1e-9)
        
        # 거래 통계
        avg_trade_duration = np.mean([t[1] for t in self.trades]) if len(self.trades) > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_trades': self.transaction_count,
            'avg_trade_duration': avg_trade_duration
        }
    
    def close(self):
        """환경 종료"""
        pass
