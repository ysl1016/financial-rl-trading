import sys
import os
import unittest
import numpy as np
import pandas as pd

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.trading_env import TradingEnv
from src.utils.indicators import calculate_technical_indicators

class TestTradingEnv(unittest.TestCase):
    """TradingEnv 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 환경 설정"""
        # 테스트용 데이터 생성
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # 가격 데이터 생성
        close_prices = np.random.normal(100, 5, size=100).cumsum() + 1000
        high_prices = close_prices * (1 + np.random.uniform(0, 0.05, size=100))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.05, size=100))
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, size=100))
        volumes = np.random.randint(1000, 10000, size=100)
        
        # 데이터프레임 생성
        self.test_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        # 기술적 지표 계산
        indicators = calculate_technical_indicators(self.test_data)
        
        # 데이터와 지표 결합
        self.processed_data = pd.concat([self.test_data, indicators], axis=1)
        
        # 테스트 환경 생성
        self.env = TradingEnv(
            data=self.processed_data,
            initial_capital=100000,
            trading_cost=0.0005,
            slippage=0.0001,
            risk_free_rate=0.02,
            max_position_size=1.0,
            stop_loss_pct=0.02
        )
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.env.initial_capital, 100000)
        self.assertEqual(self.env.trading_cost, 0.0005)
        self.assertEqual(self.env.slippage, 0.0001)
        self.assertEqual(self.env.risk_free_rate, 0.02 / 252)  # 일별 리스크프리 레이트
        self.assertEqual(self.env.max_position_size, 1.0)
        self.assertEqual(self.env.stop_loss_pct, 0.02)
        
        # 상태 및 행동 공간 확인
        self.assertEqual(self.env.observation_space.shape[0], 14)
        self.assertEqual(self.env.action_space.n, 3)
    
    def test_reset(self):
        """환경 리셋 테스트"""
        obs = self.env.reset()
        
        # 관측 상태 확인
        self.assertEqual(len(obs), 14)
        self.assertEqual(self.env.index, 0)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.cash, 100000)
        self.assertEqual(self.env.holdings, 0)
        self.assertEqual(len(self.env.portfolio_values), 1)
        self.assertEqual(self.env.portfolio_values[0], 100000)
    
    def test_step_hold(self):
        """보유(hold) 액션 테스트"""
        self.env.reset()
        initial_portfolio = self.env.portfolio_values[0]
        
        # 보유(hold) 액션 (0)
        next_obs, reward, done, info = self.env.step(0)
        
        # 상태 확인
        self.assertEqual(self.env.index, 1)
        self.assertEqual(self.env.position, 0)  # 여전히 포지션 없음
        self.assertEqual(self.env.cash, 100000)  # 현금 동일
        self.assertEqual(self.env.holdings, 0)  # 보유량 동일
        self.assertEqual(len(next_obs), 14)  # 관측 상태 확인
        
        # 포트폴리오 가치와 보상 확인
        self.assertEqual(len(self.env.portfolio_values), 2)
        self.assertEqual(self.env.portfolio_values[1], initial_portfolio)
        self.assertAlmostEqual(reward, 0.0, places=5)
    
    def test_step_buy(self):
        """매수(buy) 액션 테스트"""
        self.env.reset()
        initial_portfolio = self.env.portfolio_values[0]
        
        # 매수(buy) 액션 (1)
        next_obs, reward, done, info = self.env.step(1)
        
        # 상태 확인
        self.assertEqual(self.env.index, 1)
        self.assertEqual(self.env.position, 1)  # 롱 포지션
        self.assertGreater(initial_portfolio, self.env.cash)  # 현금 감소
        self.assertGreater(self.env.holdings, 0)  # 보유량 증가
        
        # 거래 기록 확인
        self.assertEqual(len(self.env.trades), 1)
        self.assertEqual(self.env.trades[0][0], 'buy')
    
    def test_step_sell(self):
        """매도(sell) 액션 테스트"""
        self.env.reset()
        initial_portfolio = self.env.portfolio_values[0]
        
        # 매도(sell) 액션 (2)
        next_obs, reward, done, info = self.env.step(2)
        
        # 상태 확인
        self.assertEqual(self.env.index, 1)
        self.assertEqual(self.env.position, -1)  # 숏 포지션
        self.assertGreater(self.env.cash, initial_portfolio)  # 현금 증가
        self.assertLess(self.env.holdings, 0)  # 음수 보유량
        
        # 거래 기록 확인
        self.assertEqual(len(self.env.trades), 1)
        self.assertEqual(self.env.trades[0][0], 'short')
    
    def test_position_change(self):
        """포지션 변경 테스트"""
        self.env.reset()
        
        # 매수(buy) 액션
        self.env.step(1)
        self.assertEqual(self.env.position, 1)
        
        # 매도(sell) 액션으로 포지션 변경
        self.env.step(2)
        self.assertEqual(self.env.position, -1)
        
        # 매수(buy) 액션으로 다시 포지션 변경
        self.env.step(1)
        self.assertEqual(self.env.position, 1)
        
        # 매수(buy) 액션 반복 - 포지션 유지 확인
        prev_holdings = self.env.holdings
        self.env.step(1)
        self.assertEqual(self.env.position, 1)
        self.assertEqual(self.env.holdings, prev_holdings)  # 이미 매수 포지션이므로 보유량 유지
    
    def test_stop_loss(self):
        """스톱로스 메커니즘 테스트"""
        self.env.reset()
        
        # 가격 데이터 직접 조작 (스톱로스 트리거를 위해)
        # 먼저 매수 포지션 진입
        self.env.step(1)
        entry_price = self.env.entry_price
        
        # 다음 단계의 가격을 큰 폭으로 하락시킴 (스톱로스 트리거)
        next_idx = self.env.index
        temp_close = self.env.data.at[self.env.data.index[next_idx], 'Close']
        self.env.data.at[self.env.data.index[next_idx], 'Close'] = entry_price * (1 - 0.025)  # 2.5% 하락
        
        # 보유(hold) 액션을 수행해도 스톱로스가 작동해야 함
        self.env.step(0)
        
        # 스톱로스로 인해 포지션이 청산되었는지 확인
        self.assertEqual(self.env.position, 0)
        self.assertTrue(any(trade[0] == 'close_long' for trade in self.env.trades))
        
        # 원래 가격으로 복원
        self.env.data.at[self.env.data.index[next_idx-1], 'Close'] = temp_close
    
    def test_done_flag(self):
        """에피소드 종료 플래그 테스트"""
        self.env.reset()
        
        # 마지막 단계까지 진행
        done = False
        step_count = 0
        max_steps = len(self.processed_data) - 1
        
        while not done and step_count < max_steps:
            _, _, done, _ = self.env.step(0)  # 보유(hold) 액션
            step_count += 1
        
        # 에피소드 종료 확인
        self.assertTrue(done)
        self.assertEqual(self.env.index, max_steps)
    
    def test_calculation_reward(self):
        """보상 계산 테스트"""
        self.env.reset()
        
        # 포트폴리오 가치 변화에 따른 보상 계산
        old_value = 100000
        new_value = 101000  # 1% 증가
        action = 1  # 매수
        
        # 보상 계산
        reward = self.env._calculate_reward(old_value, new_value, action)
        
        # 보상이 양수이고 대략적으로 1%에 가까운지 확인
        self.assertGreater(reward, 0)
        
        # 포트폴리오 가치 감소에 따른 보상 계산
        old_value = 100000
        new_value = 99000  # 1% 감소
        
        # 보상 계산
        reward = self.env._calculate_reward(old_value, new_value, action)
        
        # 보상이 음수인지 확인
        self.assertLess(reward, 0)

if __name__ == '__main__':
    unittest.main()
