"""
백테스팅을 위한 유틸리티 함수들을 제공합니다.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ..models.trading_env import TradingEnv


def split_data_by_date(data: pd.DataFrame, 
                      train_ratio: float = 0.6, 
                      val_ratio: float = 0.2, 
                      test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    날짜 기반으로 데이터를 훈련, 검증, 테스트 세트로 분할합니다.
    
    Args:
        data: OHLCV 데이터와 기술적 지표가 포함된 DataFrame
        train_ratio: 훈련 데이터 비율 (0.0-1.0)
        val_ratio: 검증 데이터 비율 (0.0-1.0)
        test_ratio: 테스트 데이터 비율 (0.0-1.0)
        
    Returns:
        훈련, 검증, 테스트 데이터셋 (DataFrame 튜플)
    """
    # 비율 합이 1.0인지 확인
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합은 1.0이어야 합니다."
    
    # 인덱스가 날짜 형식인지 확인
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data = data.set_index('Date')
        else:
            raise ValueError("데이터에 날짜 인덱스가 없습니다.")
    
    # 날짜 정렬
    data = data.sort_index()
    
    # 분할 인덱스 계산
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # 데이터 분할
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data


def split_data_by_window(data: pd.DataFrame, 
                        window_size: int = 252, 
                        stride: int = 126, 
                        min_samples: int = 1000) -> List[pd.DataFrame]:
    """
    롤링 윈도우 방식으로 데이터를 여러 기간으로 분할합니다.
    이는 워크아웃 샘플 테스트를 위해 유용합니다.
    
    Args:
        data: OHLCV 데이터와 기술적 지표가 포함된 DataFrame
        window_size: 각 윈도우의 기간 (일)
        stride: 윈도우 간 이동 간격 (일)
        min_samples: 윈도우에 필요한 최소 샘플 수
        
    Returns:
        DataFrame 목록 (각각 다른 시장 기간 포함)
    """
    windows = []
    n = len(data)
    
    # 충분한 데이터가 있는지 확인
    if n < min_samples:
        raise ValueError(f"데이터 포인트가 부족합니다. 최소 {min_samples}개가 필요하지만 {n}개가 있습니다.")
    
    # 인덱스가 날짜 형식인지 확인
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data = data.set_index('Date')
        else:
            raise ValueError("데이터에 날짜 인덱스가 없습니다.")
    
    # 데이터 정렬
    data = data.sort_index()
    
    # 윈도우 생성
    for i in range(0, n - window_size + 1, stride):
        if i + window_size <= n:
            window_data = data.iloc[i:i + window_size]
            if len(window_data) >= min_samples / 10:  # 최소한 데이터의 10%는 있어야 함
                windows.append(window_data)
    
    return windows


def generate_market_regimes(data: pd.DataFrame, 
                           lookback: int = 20, 
                           vol_threshold: float = 0.015) -> pd.DataFrame:
    """
    시장 레짐을 식별하고 레이블을 생성합니다.
    
    Args:
        data: OHLCV 데이터가 포함된 DataFrame
        lookback: 변동성 계산을 위한 룩백 기간
        vol_threshold: 고변동성 레짐을 정의하는 임계값
        
    Returns:
        시장 레짐 레이블이 추가된 DataFrame
    """
    # 데이터 복사
    df = data.copy()
    
    # 일일 수익률 계산
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()
    
    # 변동성 계산 (지수 이동 평균 이용)
    df['Volatility'] = df['Return'].ewm(span=lookback).std()
    
    # 추세 계산 (SMA 이용)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # 레짐 레이블링
    conditions = [
        (df['SMA20'] > df['SMA50']) & (df['Volatility'] <= vol_threshold),  # 저변동성 상승 추세
        (df['SMA20'] > df['SMA50']) & (df['Volatility'] > vol_threshold),   # 고변동성 상승 추세
        (df['SMA20'] <= df['SMA50']) & (df['Volatility'] <= vol_threshold), # 저변동성 하락 추세
        (df['SMA20'] <= df['SMA50']) & (df['Volatility'] > vol_threshold)   # 고변동성 하락 추세
    ]
    
    regime_labels = ['Bull_LowVol', 'Bull_HighVol', 'Bear_LowVol', 'Bear_HighVol']
    df['MarketRegime'] = np.select(conditions, regime_labels, default='Unknown')
    
    return df


def run_backtest(env: TradingEnv, 
                agent: Any, 
                episodes: int = 1, 
                render: bool = False, 
                verbose: bool = True) -> Dict[str, Any]:
    """
    훈련된 에이전트로 백테스트를 실행합니다.
    
    Args:
        env: 트레이딩 환경
        agent: 학습된 에이전트
        episodes: 백테스트 에피소드 수
        render: 결과 시각화 여부
        verbose: 상세 로깅 여부
        
    Returns:
        백테스트 결과를 포함하는 딕셔너리
    """
    results = {
        'portfolio_values': [],
        'returns': [],
        'trades': [],
        'positions': [],
        'actions': [],
        'rewards': [],
        'dates': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 에이전트 행동 선택
            action = agent.select_action(state)
            
            # 환경 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 결과 저장
            episode_reward += reward
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['positions'].append(env.position)
            
            if hasattr(env, 'data') and hasattr(env.data, 'index'):
                results['dates'].append(env.data.index[env.index - 1])
            
            if 'portfolio_value' in info:
                results['portfolio_values'].append(info['portfolio_value'])
            
            if 'trades' in info and len(info['trades']) > len(results['trades']):
                for trade in info['trades'][len(results['trades']):]:
                    results['trades'].append(trade)
            
            # 상태 업데이트
            state = next_state
            
            # 진행상황 출력
            if verbose and env.index % 100 == 0:
                print(f"Episode {episode+1}/{episodes}, Step {env.index}/{len(env.data)}, "
                      f"Portfolio: ${info['portfolio_value']:.2f}, Position: {env.position}")
        
        # 에피소드 결과 계산
        if len(results['portfolio_values']) > 1:
            returns = np.array(results['portfolio_values']) / results['portfolio_values'][0] - 1
            results['returns'] = returns.tolist()
        
        if verbose:
            print(f"Backtest Episode {episode+1}/{episodes} completed.")
            print(f"Final Portfolio Value: ${results['portfolio_values'][-1]:.2f}")
            print(f"Total Return: {results['returns'][-1] * 100:.2f}%")
            print(f"Number of Trades: {len(results['trades'])}")
            print("-" * 50)
    
    return results


def calculate_drawdowns(portfolio_values: List[float]) -> Tuple[float, int, int]:
    """
    포트폴리오 가치에서 최대 손실(drawdown)을 계산합니다.
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        
    Returns:
        최대 손실 비율, 시작 인덱스, 종료 인덱스
    """
    portfolio_values = np.array(portfolio_values)
    max_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - max_values) / max_values
    
    # 최대 손실 계산
    max_drawdown_idx = np.argmin(drawdowns)
    max_drawdown = drawdowns[max_drawdown_idx]
    
    # 최대 손실 기간의 시작점 찾기
    peak_idx = np.where(portfolio_values[:max_drawdown_idx] == max_values[max_drawdown_idx])[0]
    if len(peak_idx) > 0:
        peak_idx = peak_idx[-1]
    else:
        peak_idx = 0
    
    return max_drawdown, peak_idx, max_drawdown_idx


def simulate_market_shock(data: pd.DataFrame, 
                         shock_magnitude: float = -0.1, 
                         shock_duration: int = 10) -> pd.DataFrame:
    """
    시장 충격 이벤트를 시뮬레이션합니다.
    
    Args:
        data: OHLCV 데이터가 포함된 DataFrame
        shock_magnitude: 충격의 크기 (수익률 단위)
        shock_duration: 충격 지속 기간 (일)
        
    Returns:
        시장 충격이 적용된 DataFrame
    """
    # 데이터 복사
    df = data.copy()
    
    # 충격 적용 위치 (데이터의 중간 지점)
    shock_start = len(df) // 2
    
    # 기준 가격
    base_price = df['Close'].iloc[shock_start - 1]
    
    # 충격 적용
    for i in range(shock_duration):
        if shock_start + i < len(df):
            day_pct = (i + 1) / shock_duration  # 일별 충격 비율
            day_shock = shock_magnitude * (1 - day_pct)  # 시간에 따라 감소하는 충격
            
            # 기존 가격에서 충격 적용
            price_factor = (1 + day_shock)
            df.loc[df.index[shock_start + i], 'Open'] *= price_factor
            df.loc[df.index[shock_start + i], 'High'] *= price_factor
            df.loc[df.index[shock_start + i], 'Low'] *= price_factor
            df.loc[df.index[shock_start + i], 'Close'] *= price_factor
    
    # 충격 이후 수익률 재계산
    df['Return'] = df['Close'].pct_change()
    
    return df


def monte_carlo_simulation(env: TradingEnv, 
                          agent: Any, 
                          n_simulations: int = 100, 
                          confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    몬테카를로 시뮬레이션을 통해 모델의 강건성을 테스트합니다.
    
    Args:
        env: 트레이딩 환경
        agent: 학습된 에이전트
        n_simulations: 시뮬레이션 횟수
        confidence_level: 신뢰 구간 수준
        
    Returns:
        시뮬레이션 결과를 포함하는 딕셔너리
    """
    all_returns = []
    all_drawdowns = []
    all_sharpe_ratios = []
    
    for i in range(n_simulations):
        # 환경 초기화
        state = env.reset()
        done = False
        portfolio_values = [env.initial_capital]
        
        # 에피소드 실행
        while not done:
            # 노이즈를 추가한 행동 선택 (탐색 강화)
            action = agent.select_action(state)
            
            # 환경 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 포트폴리오 가치 추적
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
            
            # 상태 업데이트
            state = next_state
        
        # 결과 계산
        final_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        all_returns.append(final_return)
        
        # 최대 손실 계산
        max_drawdown, _, _ = calculate_drawdowns(portfolio_values)
        all_drawdowns.append(max_drawdown)
        
        # 일별 수익률 계산
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe 비율 계산 (연율화)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        all_sharpe_ratios.append(sharpe_ratio)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n_simulations} simulations")
    
    # 결과 통계 계산
    returns_array = np.array(all_returns)
    drawdowns_array = np.array(all_drawdowns)
    sharpe_array = np.array(all_sharpe_ratios)
    
    # 신뢰 구간 계산
    alpha = 1 - confidence_level
    returns_ci = np.percentile(returns_array, [alpha/2*100, (1-alpha/2)*100])
    drawdowns_ci = np.percentile(drawdowns_array, [alpha/2*100, (1-alpha/2)*100])
    sharpe_ci = np.percentile(sharpe_array, [alpha/2*100, (1-alpha/2)*100])
    
    # 결과 반환
    return {
        'returns': {
            'mean': np.mean(returns_array),
            'std': np.std(returns_array),
            'min': np.min(returns_array),
            'max': np.max(returns_array),
            'median': np.median(returns_array),
            'ci_lower': returns_ci[0],
            'ci_upper': returns_ci[1]
        },
        'max_drawdowns': {
            'mean': np.mean(drawdowns_array),
            'std': np.std(drawdowns_array),
            'min': np.min(drawdowns_array),
            'max': np.max(drawdowns_array),
            'median': np.median(drawdowns_array),
            'ci_lower': drawdowns_ci[0],
            'ci_upper': drawdowns_ci[1]
        },
        'sharpe_ratios': {
            'mean': np.mean(sharpe_array),
            'std': np.std(sharpe_array),
            'min': np.min(sharpe_array),
            'max': np.max(sharpe_array),
            'median': np.median(sharpe_array),
            'ci_lower': sharpe_ci[0],
            'ci_upper': sharpe_ci[1]
        },
        'all_returns': all_returns,
        'all_drawdowns': all_drawdowns,
        'all_sharpe_ratios': all_sharpe_ratios
    }
