#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
백테스팅 유틸리티 모듈

이 모듈은 금융 트레이딩 전략의 백테스팅을 위한 다양한 유틸리티 함수를 제공합니다.
데이터 분할, 시장 레짐 감지, 몬테카를로 시뮬레이션, 스트레스 테스트 등의 기능을 포함합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import random
import math

from .evaluation import (
    calculate_returns, calculate_sharpe_ratio, calculate_sortino_ratio,
    calculate_max_drawdown, calculate_performance_summary, detect_market_regimes
)


def split_data_by_date(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    날짜별 데이터 분할

    Args:
        data: 분할할 데이터프레임 (날짜 인덱스 필요)
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        shuffle: 셔플 여부 (주의: 시계열 데이터는 일반적으로 셔플하지 않음)
    
    Returns:
        train_data, val_data, test_data: 분할된 데이터프레임
    """
    # 비율 합이 1인지 확인
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "비율의 합은 1이어야 합니다."
    
    # 데이터 정렬
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    
    # 셔플 (필요한 경우)
    if shuffle:
        data = data.sample(frac=1.0, random_state=42)
    
    # 인덱스 계산
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    # 데이터 분할
    train_data = data.iloc[:train_idx].copy()
    val_data = data.iloc[train_idx:val_idx].copy()
    test_data = data.iloc[val_idx:].copy()
    
    return train_data, val_data, test_data


def split_data_by_windows(
    data: pd.DataFrame,
    window_size: int,
    stride: int = 1,
    normalize: bool = True
) -> List[pd.DataFrame]:
    """
    윈도우 기반 데이터 분할

    Args:
        data: 분할할 데이터프레임
        window_size: 윈도우 크기
        stride: 윈도우 이동 단위
        normalize: 윈도우별 정규화 여부
    
    Returns:
        windows: 윈도우 데이터프레임 리스트
    """
    windows = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data.iloc[i:i+window_size].copy()
        
        if normalize:
            # 수치형 열만 정규화
            numeric_cols = window.select_dtypes(include=np.number).columns
            
            if not numeric_cols.empty:
                window[numeric_cols] = (window[numeric_cols] - window[numeric_cols].mean()) / window[numeric_cols].std()
        
        windows.append(window)
    
    return windows


def detect_market_regimes_hmm(
    returns: np.ndarray,
    n_regimes: int = 4,
    lookback: int = 252
) -> np.ndarray:
    """
    HMM 기반 시장 레짐 감지

    Args:
        returns: 수익률 시계열
        n_regimes: 레짐 수
        lookback: 관측 윈도우 크기
    
    Returns:
        레짐 레이블
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        print("hmmlearn 패키지가 설치되어 있지 않습니다.")
        return np.full(len(returns), np.nan)
    
    # 관측 데이터 구성
    if len(returns) <= lookback:
        lookback = len(returns) // 2
    
    # 특성 추출
    features = []
    
    # 롤링 통계
    for i in range(lookback, len(returns) + 1):
        window_returns = returns[i - lookback:i]
        
        # 특성 계산
        mean = np.mean(window_returns)
        std = np.std(window_returns)
        skew = 0 if len(window_returns) < 3 else (((window_returns - mean) ** 3).mean() / (std ** 3))
        kurt = 0 if len(window_returns) < 4 else (((window_returns - mean) ** 4).mean() / (std ** 4)) - 3
        
        # 추가 특성
        abs_returns = np.abs(window_returns)
        mean_abs = np.mean(abs_returns)
        max_dd = calculate_max_drawdown(np.cumprod(1 + window_returns))
        
        features.append([mean, std, skew, kurt, mean_abs, max_dd])
    
    features = np.array(features)
    
    # 특성 정규화
    for j in range(features.shape[1]):
        if np.std(features[:, j]) > 0:
            features[:, j] = (features[:, j] - np.mean(features[:, j])) / np.std(features[:, j])
    
    # HMM 모델링
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    try:
        model.fit(features)
        regimes = model.predict(features)
    except Exception as e:
        print(f"HMM 모델링 오류: {e}")
        regimes = np.zeros(len(features))
    
    # 레짐 레이블 패딩
    full_regimes = np.full(len(returns), np.nan)
    full_regimes[lookback:] = regimes
    
    # 초기 부분 채우기
    if lookback > 0:
        full_regimes[:lookback] = full_regimes[lookback]
    
    return full_regimes


def detect_market_regimes_kmeans(
    returns: np.ndarray,
    n_regimes: int = 4,
    lookback: int = 252
) -> np.ndarray:
    """
    K-means 기반 시장 레짐 감지

    Args:
        returns: 수익률 시계열
        n_regimes: 레짐 수
        lookback: 관측 윈도우 크기
    
    Returns:
        레짐 레이블
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn 패키지가 설치되어 있지 않습니다.")
        return np.full(len(returns), np.nan)
    
    # 관측 데이터 구성
    if len(returns) <= lookback:
        lookback = len(returns) // 2
    
    # 특성 추출
    features = []
    
    # 롤링 통계
    for i in range(lookback, len(returns) + 1):
        window_returns = returns[i - lookback:i]
        
        # 특성 계산
        mean = np.mean(window_returns)
        std = np.std(window_returns)
        
        # 승률 계산
        win_rate = np.mean(window_returns > 0)
        
        # 상승/하락 연속성
        pos_streak = max([len(list(g)) for k, g in itertools.groupby(window_returns > 0) if k], default=0)
        neg_streak = max([len(list(g)) for k, g in itertools.groupby(window_returns < 0) if k], default=0)
        
        features.append([mean, std, win_rate, pos_streak, neg_streak])
    
    features = np.array(features)
    
    # 특성 정규화
    for j in range(features.shape[1]):
        if np.std(features[:, j]) > 0:
            features[:, j] = (features[:, j] - np.mean(features[:, j])) / np.std(features[:, j])
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(features)
    
    # 레짐 레이블 패딩
    full_regimes = np.full(len(returns), np.nan)
    full_regimes[lookback:] = regimes
    
    # 초기 부분 채우기
    if lookback > 0:
        full_regimes[:lookback] = full_regimes[lookback]
    
    return full_regimes


def detect_market_regimes_volatility(
    returns: np.ndarray,
    n_regimes: int = 4,
    lookback: int = 252
) -> np.ndarray:
    """
    변동성 기반 시장 레짐 감지

    Args:
        returns: 수익률 시계열
        n_regimes: 레짐 수
        lookback: 관측 윈도우 크기
    
    Returns:
        레짐 레이블
    """
    # 관측 데이터 구성
    if len(returns) <= lookback:
        lookback = len(returns) // 2
    
    # 롤링 변동성 계산
    rolling_vol = np.full(len(returns), np.nan)
    
    for i in range(lookback, len(returns) + 1):
        window_returns = returns[i - lookback:i]
        rolling_vol[i - 1] = np.std(window_returns)
    
    # 변동성 백분위수 계산
    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    percentiles = np.linspace(0, 100, n_regimes + 1)[1:]
    thresholds = np.percentile(valid_vol, percentiles)
    
    # 레짐 할당
    regimes = np.full(len(returns), np.nan)
    
    for i in range(len(rolling_vol)):
        if not np.isnan(rolling_vol[i]):
            regime = 0
            for j, threshold in enumerate(thresholds):
                if rolling_vol[i] >= threshold:
                    regime = j + 1
            regimes[i] = regime
    
    # NaN 값 보간
    for i in range(len(regimes)):
        if np.isnan(regimes[i]) and i > 0 and not np.isnan(regimes[i-1]):
            regimes[i] = regimes[i-1]
    
    # 초기 부분 채우기
    first_valid = np.where(~np.isnan(regimes))[0][0] if np.any(~np.isnan(regimes)) else 0
    if first_valid > 0:
        regimes[:first_valid] = regimes[first_valid]
    
    return regimes


def detect_market_regimes_composite(
    data: pd.DataFrame,
    n_regimes: int = 4,
    lookback: int = 252,
    method: str = 'combined'
) -> np.ndarray:
    """
    복합 시장 레짐 감지

    Args:
        data: OHLCV 데이터프레임
        n_regimes: 레짐 수
        lookback: 관측 윈도우 크기
        method: 감지 방법 ('combined', 'hmm', 'kmeans', 'volatility')
    
    Returns:
        레짐 레이블
    """
    # 가격 및 수익률 추출
    if 'Close' not in data.columns:
        raise ValueError("데이터프레임에 'Close' 열이 없습니다.")
    
    prices = data['Close'].values
    returns = calculate_returns(prices)
    
    # 방법별 감지
    if method == 'hmm':
        return detect_market_regimes_hmm(returns, n_regimes, lookback)
    elif method == 'kmeans':
        return detect_market_regimes_kmeans(returns, n_regimes, lookback)
    elif method == 'volatility':
        return detect_market_regimes_volatility(returns, n_regimes, lookback)
    elif method == 'combined':
        # 여러 방법의 결과 통합
        try:
            hmm_regimes = detect_market_regimes_hmm(returns, n_regimes, lookback)
            kmeans_regimes = detect_market_regimes_kmeans(returns, n_regimes, lookback)
            vol_regimes = detect_market_regimes_volatility(returns, n_regimes, lookback)
            
            # 통합 방법: 다수결 (NaN 제외)
            combined_regimes = np.full(len(returns), np.nan)
            
            for i in range(len(returns)):
                votes = [r for r in [hmm_regimes[i], kmeans_regimes[i], vol_regimes[i]] if not np.isnan(r)]
                if votes:
                    # 가장 흔한 레짐 선택
                    unique_votes, counts = np.unique(votes, return_counts=True)
                    combined_regimes[i] = unique_votes[np.argmax(counts)]
            
            # NaN 값 보간
            for i in range(len(combined_regimes)):
                if np.isnan(combined_regimes[i]) and i > 0 and not np.isnan(combined_regimes[i-1]):
                    combined_regimes[i] = combined_regimes[i-1]
            
            # 초기 부분 채우기
            first_valid = np.where(~np.isnan(combined_regimes))[0][0] if np.any(~np.isnan(combined_regimes)) else 0
            if first_valid > 0:
                combined_regimes[:first_valid] = combined_regimes[first_valid]
            
            return combined_regimes
        
        except Exception as e:
            print(f"복합 레짐 감지 오류: {e}")
            # 오류 발생 시 변동성 기반 방법 사용
            return detect_market_regimes_volatility(returns, n_regimes, lookback)
    else:
        raise ValueError(f"지원되지 않는 방법: {method}")


def run_backtest(
    data: pd.DataFrame,
    strategy_func: Callable,
    initial_capital: float = 100000.0,
    trading_cost: float = 0.0005,
    slippage: float = 0.0001,
    risk_free_rate: float = 0.02,
    max_position_size: float = 1.0,
    stop_loss_pct: float = 0.02,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    백테스트 실행

    Args:
        data: OHLCV 데이터프레임
        strategy_func: 전략 함수 (데이터를 입력받아 행동 반환)
        initial_capital: 초기 자본
        trading_cost: 거래 비용
        slippage: 슬리피지
        risk_free_rate: 무위험 이자율
        max_position_size: 최대 포지션 크기
        stop_loss_pct: 손절매 비율
        verbose: 상세 출력 여부
    
    Returns:
        백테스트 결과 딕셔너리
    """
    # 초기화
    capital = initial_capital
    position = 0  # 0=보유 없음, 1=롱, -1=숏
    holdings = 0
    entry_price = 0
    
    portfolio_values = [capital]
    positions = [position]
    trades = []
    
    # 백테스트 실행
    for i in range(1, len(data)):
        # 현재 및 이전 데이터
        current_data = data.iloc[:i+1]
        current_price = current_data.iloc[-1]['Close']
        
        # 전략 함수 호출
        action = strategy_func(current_data)
        
        # 스톱로스 확인
        if position != 0:
            price_change = (current_price - entry_price) / entry_price
            if (position == 1 and price_change < -stop_loss_pct) or \
               (position == -1 and price_change > stop_loss_pct):
                if verbose:
                    print(f"스톱로스 발동: {data.index[i]}, 가격: {current_price:.2f}, 변화율: {price_change:.2%}")
                
                action = 2 if position == 1 else 1  # 포지션 종료
        
        # 슬리피지 적용
        if action == 1:  # 매수
            exec_price = current_price * (1 + slippage)
        elif action == 2:  # 매도
            exec_price = current_price * (1 - slippage)
        else:
            exec_price = current_price
        
        # 행동 실행
        if action == 1:  # 매수
            if position == 0:
                position = 1
                holdings = max_position_size
                cost = exec_price * holdings
                capital -= cost * (1 + trading_cost)
                entry_price = exec_price
                
                trades.append({
                    'date': data.index[i],
                    'type': 'buy',
                    'price': exec_price,
                    'size': holdings,
                    'cost': cost,
                    'capital': capital
                })
                
                if verbose:
                    print(f"매수: {data.index[i]}, 가격: {exec_price:.2f}, 크기: {holdings}, 비용: {cost:.2f}")
            
            elif position == -1:
                # 숏 포지션 종료
                profit = (entry_price - exec_price) * abs(holdings)
                capital += profit
                capital -= abs(holdings) * exec_price * trading_cost
                
                trades.append({
                    'date': data.index[i],
                    'type': 'close_short',
                    'price': exec_price,
                    'size': abs(holdings),
                    'profit': profit,
                    'capital': capital
                })
                
                if verbose:
                    print(f"숏 종료: {data.index[i]}, 가격: {exec_price:.2f}, 수익: {profit:.2f}")
                
                # 롱 포지션 진입
                position = 1
                holdings = max_position_size
                cost = exec_price * holdings
                capital -= cost * (1 + trading_cost)
                entry_price = exec_price
                
                trades.append({
                    'date': data.index[i],
                    'type': 'buy',
                    'price': exec_price,
                    'size': holdings,
                    'cost': cost,
                    'capital': capital
                })
                
                if verbose:
                    print(f"매수: {data.index[i]}, 가격: {exec_price:.2f}, 크기: {holdings}, 비용: {cost:.2f}")
        
        elif action == 2:  # 매도
            if position == 0:
                position = -1
                holdings = -max_position_size
                proceed = exec_price * abs(holdings)
                capital += proceed * (1 - trading_cost)
                entry_price = exec_price
                
                trades.append({
                    'date': data.index[i],
                    'type': 'short',
                    'price': exec_price,
                    'size': abs(holdings),
                    'proceed': proceed,
                    'capital': capital
                })
                
                if verbose:
                    print(f"숏: {data.index[i]}, 가격: {exec_price:.2f}, 크기: {abs(holdings)}, 매출: {proceed:.2f}")
            
            elif position == 1:
                # 롱 포지션 종료
                profit = (exec_price - entry_price) * holdings
                capital += profit
                capital -= holdings * exec_price * trading_cost
                
                trades.append({
                    'date': data.index[i],
                    'type': 'close_long',
                    'price': exec_price,
                    'size': holdings,
                    'profit': profit,
                    'capital': capital
                })
                
                if verbose:
                    print(f"롱 종료: {data.index[i]}, 가격: {exec_price:.2f}, 수익: {profit:.2f}")
                
                # 숏 포지션 진입
                position = -1
                holdings = -max_position_size
                proceed = exec_price * abs(holdings)
                capital += proceed * (1 - trading_cost)
                entry_price = exec_price
                
                trades.append({
                    'date': data.index[i],
                    'type': 'short',
                    'price': exec_price,
                    'size': abs(holdings),
                    'proceed': proceed,
                    'capital': capital
                })
                
                if verbose:
                    print(f"숏: {data.index[i]}, 가격: {exec_price:.2f}, 크기: {abs(holdings)}, 매출: {proceed:.2f}")
        
        # 포트폴리오 가치 업데이트
        if position == 0:
            portfolio_value = capital
        elif position == 1:
            portfolio_value = capital + (holdings * current_price)
        else:  # position == -1
            portfolio_value = capital - (holdings * current_price)
        
        portfolio_values.append(portfolio_value)
        positions.append(position)
    
    # 수익률 계산
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # 성과 지표 계산
    metrics = calculate_performance_summary(
        returns=returns,
        prices=np.array(portfolio_values),
        risk_free_rate=risk_free_rate
    )
    
    # 결과 반환
    return {
        'portfolio_values': portfolio_values,
        'positions': positions,
        'trades': trades,
        'metrics': metrics,
        'data': data
    }


def generate_monte_carlo_scenarios(
    returns: np.ndarray,
    n_scenarios: int = 100,
    horizon: int = 252,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    몬테카를로 시나리오 생성

    Args:
        returns: 수익률 시계열
        n_scenarios: 시나리오 수
        horizon: 시나리오 기간
        seed: 무작위 시드
    
    Returns:
        시나리오 배열 [n_scenarios, horizon]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 수익률 통계 추정
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # 정규 분포에서 무작위 수익률 샘플링
    scenarios = np.random.normal(mean_return, std_return, size=(n_scenarios, horizon))
    
    return scenarios


def generate_monte_carlo_bootstrap(
    returns: np.ndarray,
    n_scenarios: int = 100,
    horizon: int = 252,
    block_size: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    블록 부트스트랩 기반 몬테카를로 시나리오 생성

    Args:
        returns: 수익률 시계열
        n_scenarios: 시나리오 수
        horizon: 시나리오 기간
        block_size: 블록 크기
        seed: 무작위 시드
    
    Returns:
        시나리오 배열 [n_scenarios, horizon]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 블록 수 계산
    n_blocks = int(np.ceil(horizon / block_size))
    total_samples = n_blocks * block_size
    
    # 시나리오 생성
    scenarios = np.zeros((n_scenarios, horizon))
    
    for i in range(n_scenarios):
        # 블록 인덱스 샘플링
        block_indices = np.random.randint(0, len(returns) - block_size + 1, size=n_blocks)
        
        # 블록 결합
        sampled_returns = np.concatenate([returns[idx:idx+block_size] for idx in block_indices])
        
        # 필요한 기간만큼 자르기
        scenarios[i, :] = sampled_returns[:horizon]
    
    return scenarios


def generate_monte_carlo_copula(
    returns: np.ndarray,
    n_scenarios: int = 100,
    horizon: int = 252,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    코퓰라 기반 몬테카를로 시나리오 생성

    Args:
        returns: 수익률 시계열
        n_scenarios: 시나리오 수
        horizon: 시나리오 기간
        seed: 무작위 시드
    
    Returns:
        시나리오 배열 [n_scenarios, horizon]
    """
    try:
        from scipy import stats
    except ImportError:
        print("scipy 패키지가 설치되어 있지 않습니다.")
        return generate_monte_carlo_scenarios(returns, n_scenarios, horizon, seed)
    
    if seed is not None:
        np.random.seed(seed)
    
    # 경험적 분포 추정
    ecdf = stats.ecdf(returns)
    
    # 수익률을 균등 분포로 변환
    uniform_data = ecdf(returns)
    
    # 코퓰라 시나리오 생성
    scenarios = np.zeros((n_scenarios, horizon))
    
    for i in range(n_scenarios):
        # 균등 분포에서 샘플링
        u = np.random.uniform(size=horizon)
        
        # 경험적 분포의 역함수를 사용하여 수익률로 변환
        scenarios[i, :] = np.percentile(returns, u * 100)
    
    return scenarios


def run_monte_carlo(
    portfolio_values: np.ndarray,
    returns: np.ndarray,
    n_scenarios: int = 100,
    horizon: int = 252,
    method: str = 'normal',
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    몬테카를로 시뮬레이션 실행

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        returns: 수익률 시계열
        n_scenarios: 시나리오 수
        horizon: 시나리오 기간
        method: 시나리오 생성 방법 ('normal', 'bootstrap', 'copula')
        seed: 무작위 시드
    
    Returns:
        시뮬레이션 결과 딕셔너리
    """
    # 시나리오 생성
    if method == 'bootstrap':
        scenarios = generate_monte_carlo_bootstrap(returns, n_scenarios, horizon, seed=seed)
    elif method == 'copula':
        scenarios = generate_monte_carlo_copula(returns, n_scenarios, horizon, seed=seed)
    else:  # 'normal'
        scenarios = generate_monte_carlo_scenarios(returns, n_scenarios, horizon, seed=seed)
    
    # 시나리오별 포트폴리오 가치 계산
    initial_value = portfolio_values[-1]
    scenario_values = np.zeros((n_scenarios, horizon + 1))
    scenario_values[:, 0] = initial_value
    
    for i in range(n_scenarios):
        for j in range(horizon):
            scenario_values[i, j+1] = scenario_values[i, j] * (1 + scenarios[i, j])
    
    # 시나리오별 성과 지표 계산
    scenario_metrics = []
    
    for i in range(n_scenarios):
        scenario_returns = np.diff(scenario_values[i]) / scenario_values[i][:-1]
        scenario_value = scenario_values[i]
        
        metrics = {
            'final_value': scenario_value[-1],
            'total_return': (scenario_value[-1] / scenario_value[0]) - 1,
            'sharpe_ratio': calculate_sharpe_ratio(scenario_returns),
            'sortino_ratio': calculate_sortino_ratio(scenario_returns),
            'max_drawdown': calculate_max_drawdown(scenario_value),
            'volatility': np.std(scenario_returns) * np.sqrt(252),
            'scenario_id': i
        }
        
        scenario_metrics.append(metrics)
    
    # 성과 요약
    final_values = np.array([m['final_value'] for m in scenario_metrics])
    total_returns = np.array([m['total_return'] for m in scenario_metrics])
    max_drawdowns = np.array([m['max_drawdown'] for m in scenario_metrics])
    
    summary = {
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'percentile_5_final_value': np.percentile(final_values, 5),
        'percentile_95_final_value': np.percentile(final_values, 95),
        'mean_return': np.mean(total_returns),
        'median_return': np.median(total_returns),
        'std_return': np.std(total_returns),
        'percentile_5_return': np.percentile(total_returns, 5),
        'percentile_95_return': np.percentile(total_returns, 95),
        'mean_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        'worst_max_drawdown': np.max(max_drawdowns),
        'success_rate': np.mean(total_returns > 0)
    }
    
    # 결과 반환
    return {
        'scenario_values': scenario_values,
        'scenario_metrics': scenario_metrics,
        'summary': summary
    }


def generate_stress_scenarios(
    data: pd.DataFrame,
    n_scenarios: int = 5,
    stress_factors: Optional[List[float]] = None,
    scenario_types: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    스트레스 테스트 시나리오 생성

    Args:
        data: OHLCV 데이터프레임
        n_scenarios: 시나리오 수
        stress_factors: 스트레스 계수 리스트 (없으면 자동 생성)
        scenario_types: 시나리오 유형 리스트 ('vol', 'crash', 'trend')
        seed: 무작위 시드
    
    Returns:
        시나리오 데이터프레임 리스트
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 스트레스 계수 설정
    if stress_factors is None:
        stress_factors = [1.0 + 0.2 * i for i in range(n_scenarios)]
    
    # 시나리오 유형 설정
    if scenario_types is None:
        scenario_types = ['vol'] * n_scenarios
        if n_scenarios > 1:
            scenario_types[1] = 'crash'
        if n_scenarios > 2:
            scenario_types[2] = 'trend'
    
    # 시나리오 생성
    scenarios = []
    
    for i in range(min(n_scenarios, len(stress_factors), len(scenario_types))):
        factor = stress_factors[i]
        scenario_type = scenario_types[i]
        
        scenario = data.copy()
        
        if scenario_type == 'vol':
            # 변동성 증폭 시나리오
            mean_price = scenario['Close'].mean()
            scenario['Close'] = mean_price + factor * (scenario['Close'] - mean_price)
            scenario['High'] = mean_price + factor * (scenario['High'] - mean_price)
            scenario['Low'] = mean_price + factor * (scenario['Low'] - mean_price)
        
        elif scenario_type == 'crash':
            # 시장 급락 시나리오
            crash_idx = len(scenario) // 3  # 1/3 지점에서 급락 시작
            crash_duration = len(scenario) // 10  # 10% 기간 동안 급락
            
            crash_factor = 1.0 - (factor - 1.0)  # 예: factor=1.2 -> crash_factor=0.8
            
            # 급락 전/후 가격
            pre_crash = scenario.iloc[:crash_idx].copy()
            crash_period = scenario.iloc[crash_idx:crash_idx+crash_duration].copy()
            post_crash = scenario.iloc[crash_idx+crash_duration:].copy()
            
            # 급락 기간 가격 조정
            base_price = crash_period.iloc[0]['Close']
            for j in range(len(crash_period)):
                crash_pct = (j + 1) / len(crash_period) * (1 - crash_factor)
                crash_period.iloc[j, crash_period.columns.get_indexer(['Close', 'High', 'Low'])] *= (1 - crash_pct)
            
            # 급락 후 가격 조정
            post_crash.iloc[:, post_crash.columns.get_indexer(['Close', 'High', 'Low'])] *= crash_factor
            
            # 시나리오 재결합
            scenario = pd.concat([pre_crash, crash_period, post_crash])
        
        elif scenario_type == 'trend':
            # 추세 전환 시나리오
            trend_factor = factor
            
            # 현재 추세 감지
            returns = scenario['Close'].pct_change().dropna()
            current_trend = np.mean(returns)
            
            # 추세 반전
            trend_diff = -current_trend * 2 * trend_factor
            
            # 선형 추세 추가
            trend = np.linspace(0, trend_diff, len(scenario))
            scenario['Close'] = scenario['Close'] * (1 + trend)
            scenario['High'] = scenario['High'] * (1 + trend)
            scenario['Low'] = scenario['Low'] * (1 + trend)
        
        else:
            # 기본 변동성 증폭
            mean_price = scenario['Close'].mean()
            scenario['Close'] = mean_price + factor * (scenario['Close'] - mean_price)
            scenario['High'] = mean_price + factor * (scenario['High'] - mean_price)
            scenario['Low'] = mean_price + factor * (scenario['Low'] - mean_price)
        
        # 시나리오 메타데이터 추가
        scenario.attrs['scenario_id'] = i
        scenario.attrs['scenario_type'] = scenario_type
        scenario.attrs['stress_factor'] = factor
        
        scenarios.append(scenario)
    
    return scenarios


def run_stress_test(
    data: pd.DataFrame,
    strategy_func: Callable,
    n_scenarios: int = 5,
    stress_factors: Optional[List[float]] = None,
    scenario_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
    initial_capital: float = 100000.0,
    trading_cost: float = 0.0005,
    slippage: float = 0.0001,
    risk_free_rate: float = 0.02,
    max_position_size: float = 1.0,
    stop_loss_pct: float = 0.02,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    스트레스 테스트 실행

    Args:
        data: OHLCV 데이터프레임
        strategy_func: 전략 함수 (데이터를 입력받아 행동 반환)
        n_scenarios: 시나리오 수
        stress_factors: 스트레스 계수 리스트 (없으면 자동 생성)
        scenario_types: 시나리오 유형 리스트 ('vol', 'crash', 'trend')
        seed: 무작위 시드
        initial_capital: 초기 자본
        trading_cost: 거래 비용
        slippage: 슬리피지
        risk_free_rate: 무위험 이자율
        max_position_size: 최대 포지션 크기
        stop_loss_pct: 손절매 비율
        verbose: 상세 출력 여부
    
    Returns:
        스트레스 테스트 결과 딕셔너리
    """
    # 스트레스 시나리오 생성
    scenarios = generate_stress_scenarios(
        data, n_scenarios, stress_factors, scenario_types, seed
    )
    
    # 각 시나리오에 대해 백테스트 실행
    results = []
    
    for i, scenario in enumerate(scenarios):
        if verbose:
            scenario_type = scenario.attrs.get('scenario_type', 'unknown')
            stress_factor = scenario.attrs.get('stress_factor', 1.0)
            print(f"\n시나리오 {i} ({scenario_type}, 계수: {stress_factor:.2f}) 백테스트 실행 중...")
        
        # 백테스트 실행
        result = run_backtest(
            scenario, strategy_func,
            initial_capital=initial_capital,
            trading_cost=trading_cost,
            slippage=slippage,
            risk_free_rate=risk_free_rate,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            verbose=verbose
        )
        
        # 시나리오 메타데이터 추가
        result['scenario_id'] = i
        result['scenario_type'] = scenario.attrs.get('scenario_type', 'unknown')
        result['stress_factor'] = scenario.attrs.get('stress_factor', 1.0)
        
        results.append(result)
    
    # 결과 요약
    scenario_metrics = [result['metrics'] for result in results]
    
    total_returns = np.array([metrics['total_return'] for metrics in scenario_metrics])
    sharpe_ratios = np.array([metrics['sharpe_ratio'] for metrics in scenario_metrics])
    max_drawdowns = np.array([metrics['max_drawdown'] for metrics in scenario_metrics])
    
    summary = {
        'mean_return': np.mean(total_returns),
        'median_return': np.median(total_returns),
        'std_return': np.std(total_returns),
        'min_return': np.min(total_returns),
        'max_return': np.max(total_returns),
        'mean_sharpe': np.mean(sharpe_ratios),
        'median_sharpe': np.median(sharpe_ratios),
        'mean_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        'worst_max_drawdown': np.max(max_drawdowns),
        'success_rate': np.mean(total_returns > 0)
    }
    
    # 결과 반환
    return {
        'scenario_results': results,
        'summary': summary
    }


def calculate_risk_contribution(
    returns: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    리스크 기여도 계산

    Args:
        returns: 자산별 수익률 [시간, 자산 수]
        weights: 자산 비중 [자산 수]
    
    Returns:
        리스크 기여도 [자산 수]
    """
    # 공분산 행렬 계산
    cov_matrix = np.cov(returns.T)
    
    # 포트폴리오 변동성 계산
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 한계 기여도 계산
    marginal_contrib = np.dot(cov_matrix, weights)
    
    # 리스크 기여도 계산
    risk_contrib = weights * marginal_contrib / portfolio_vol
    
    return risk_contrib


def optimize_risk_parity(
    returns: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> np.ndarray:
    """
    리스크 패리티 최적화

    Args:
        returns: 자산별 수익률 [시간, 자산 수]
        max_iter: 최대 반복 횟수
        tol: 수렴 허용 오차
    
    Returns:
        최적 비중 [자산 수]
    """
    n = returns.shape[1]
    
    # 초기 균등 비중
    x = np.ones(n) / n
    
    # 공분산 행렬 계산
    cov_matrix = np.cov(returns.T)
    
    # 반복 최적화
    for i in range(max_iter):
        # 리스크 기여도 계산
        rc = calculate_risk_contribution(returns, x)
        
        # 목표 리스크 기여도 (균등)
        target_rc = np.ones(n) / n
        
        # 차이 계산
        diff = rc - target_rc
        
        # 수렴 확인
        if np.max(np.abs(diff)) < tol:
            break
        
        # 그래디언트 계산
        grad = 2 * np.dot(cov_matrix, x)
        
        # 업데이트 방향
        direction = grad * diff
        
        # 스텝 크기
        step_size = 0.01
        
        # 비중 업데이트
        x -= step_size * direction
        
        # 비중 정규화
        x = np.maximum(x, 0)
        x = x / np.sum(x)
    
    return x
