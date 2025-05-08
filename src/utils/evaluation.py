#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
금융 평가 지표 모듈

이 모듈은 금융 트레이딩 전략의 다양한 성과 지표를 계산하는 함수들을 제공합니다.
수익률 계산, 위험 조정 지표(Sharpe, Sortino, Calmar 비율 등), 위험 지표(최대 손실, VaR, CVaR 등),
거래 효율성 지표(승률, 손익비, 기대 수익) 등을 포함합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from scipy import stats


def calculate_returns(
    prices: np.ndarray,
    log_returns: bool = False
) -> np.ndarray:
    """
    수익률 계산

    Args:
        prices: 가격 시계열 데이터
        log_returns: 로그 수익률 계산 여부 (기본값: False)
    
    Returns:
        수익률 시계열
    """
    if log_returns:
        returns = np.log(prices[1:] / prices[:-1])
    else:
        returns = (prices[1:] / prices[:-1]) - 1
    
    return returns


def annualize_returns(
    returns: np.ndarray,
    periods_per_year: int
) -> float:
    """
    수익률 연율화

    Args:
        returns: 수익률 시계열
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        연율화된 수익률
    """
    return np.mean(returns) * periods_per_year


def annualize_volatility(
    returns: np.ndarray,
    periods_per_year: int
) -> float:
    """
    변동성 연율화

    Args:
        returns: 수익률 시계열
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        연율화된 변동성
    """
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe 비율 계산

    Args:
        returns: 수익률 시계열
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        Sharpe 비율
    """
    if len(returns) < 2:
        return 0.0
    
    ann_returns = annualize_returns(returns, periods_per_year)
    ann_volatility = annualize_volatility(returns, periods_per_year)
    
    # 변동성이 0에 가까우면 매우 큰 값 반환 방지
    if ann_volatility < 1e-6:
        return 0.0
    
    return (ann_returns - risk_free_rate) / ann_volatility


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_acceptable_return: float = 0.0
) -> float:
    """
    Sortino 비율 계산

    Args:
        returns: 수익률 시계열
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
        min_acceptable_return: 최소 허용 수익률
    
    Returns:
        Sortino 비율
    """
    if len(returns) < 2:
        return 0.0
    
    ann_returns = annualize_returns(returns, periods_per_year)
    
    # 하방 편차 계산
    downside_returns = returns[returns < min_acceptable_return]
    
    if len(downside_returns) == 0:
        return np.inf if ann_returns > risk_free_rate else 0.0
    
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year)
    
    if downside_deviation < 1e-6:
        return 0.0
    
    return (ann_returns - risk_free_rate) / downside_deviation


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    최대 손실(drawdown) 계산

    Args:
        prices: 가격 시계열 데이터
    
    Returns:
        최대 손실률
    """
    # 누적 최댓값 계산
    peak = np.maximum.accumulate(prices)
    
    # 손실률 계산
    drawdown = (peak - prices) / peak
    
    # 최대 손실률 반환
    return np.max(drawdown)


def calculate_calmar_ratio(
    returns: np.ndarray,
    prices: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calmar 비율 계산

    Args:
        returns: 수익률 시계열
        prices: 가격 시계열 데이터
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        Calmar 비율
    """
    if len(returns) < 2:
        return 0.0
    
    ann_returns = annualize_returns(returns, periods_per_year)
    max_dd = calculate_max_drawdown(prices)
    
    if max_dd < 1e-6:
        return np.inf if ann_returns > risk_free_rate else 0.0
    
    return (ann_returns - risk_free_rate) / max_dd


def calculate_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Omega 비율 계산

    Args:
        returns: 수익률 시계열
        threshold: 임계 수익률
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        Omega 비율
    """
    if len(returns) < 2:
        return 0.0
    
    # 임계 수익률로 조정된 수익률
    adjusted_returns = returns - (threshold / periods_per_year)
    
    # 상방/하방 부분 계산
    upper = adjusted_returns[adjusted_returns > 0].sum()
    lower = -adjusted_returns[adjusted_returns < 0].sum()
    
    if lower < 1e-6:
        return np.inf if upper > 0 else 0.0
    
    return upper / lower


def calculate_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Value at Risk (VaR) 계산

    Args:
        returns: 수익률 시계열
        confidence_level: 신뢰 수준
    
    Returns:
        VaR 값
    """
    if len(returns) < 2:
        return 0.0
    
    return -np.percentile(returns, 100 * (1 - confidence_level))


def calculate_conditional_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall 계산

    Args:
        returns: 수익률 시계열
        confidence_level: 신뢰 수준
    
    Returns:
        CVaR 값
    """
    if len(returns) < 2:
        return 0.0
    
    var = calculate_value_at_risk(returns, confidence_level)
    return -np.mean(returns[returns <= -var])


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    승률 계산

    Args:
        returns: 수익률 시계열
    
    Returns:
        승률
    """
    if len(returns) == 0:
        return 0.0
    
    wins = np.sum(returns > 0)
    return wins / len(returns)


def calculate_profit_loss_ratio(returns: np.ndarray) -> float:
    """
    손익비 계산

    Args:
        returns: 수익률 시계열
    
    Returns:
        손익비
    """
    if len(returns) == 0:
        return 0.0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    
    avg_win = np.mean(wins)
    avg_loss = np.abs(np.mean(losses))
    
    if avg_loss < 1e-6:
        return np.inf
    
    return avg_win / avg_loss


def calculate_expectancy(returns: np.ndarray) -> float:
    """
    기대 수익 계산

    Args:
        returns: 수익률 시계열
    
    Returns:
        기대 수익
    """
    if len(returns) == 0:
        return 0.0
    
    win_rate = calculate_win_rate(returns)
    profit_loss_ratio = calculate_profit_loss_ratio(returns)
    
    return (win_rate * profit_loss_ratio) - (1 - win_rate)


def calculate_average_drawdown(prices: np.ndarray) -> float:
    """
    평균 손실(drawdown) 계산

    Args:
        prices: 가격 시계열 데이터
    
    Returns:
        평균 손실률
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    return np.mean(drawdown)


def calculate_average_drawdown_duration(prices: np.ndarray) -> float:
    """
    평균 손실 지속 기간 계산

    Args:
        prices: 가격 시계열 데이터
    
    Returns:
        평균 손실 지속 기간
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    # 손실 상태 식별
    in_drawdown = drawdown > 0
    
    # 손실 에피소드 식별
    drawdown_episodes = np.diff(in_drawdown.astype(int))
    start_indices = np.where(drawdown_episodes == 1)[0] + 1
    end_indices = np.where(drawdown_episodes == -1)[0] + 1
    
    # 시작과 끝 조정
    if in_drawdown[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if in_drawdown[-1]:
        end_indices = np.append(end_indices, len(prices))
    
    # 지속 기간 계산
    if len(start_indices) == 0 or len(end_indices) == 0:
        return 0.0
    
    durations = end_indices - start_indices
    
    return np.mean(durations)


def calculate_ulcer_index(prices: np.ndarray) -> float:
    """
    Ulcer Index 계산 (연속적인 손실의 제곱합의 제곱근)

    Args:
        prices: 가격 시계열 데이터
    
    Returns:
        Ulcer Index
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    return np.sqrt(np.mean(np.square(drawdown)))


def calculate_sterling_ratio(
    returns: np.ndarray,
    prices: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    n_drawdowns: int = 5
) -> float:
    """
    Sterling 비율 계산

    Args:
        returns: 수익률 시계열
        prices: 가격 시계열 데이터
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
        n_drawdowns: 고려할 손실 에피소드 수
    
    Returns:
        Sterling 비율
    """
    if len(returns) < 2:
        return 0.0
    
    ann_returns = annualize_returns(returns, periods_per_year)
    
    # 손실 계산
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    # 손실 에피소드 식별
    in_drawdown = drawdown > 0
    
    # 손실 상태 변화 식별
    drawdown_episodes = np.diff(in_drawdown.astype(int))
    start_indices = np.where(drawdown_episodes == 1)[0] + 1
    end_indices = np.where(drawdown_episodes == -1)[0] + 1
    
    # 시작과 끝 조정
    if in_drawdown[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if in_drawdown[-1]:
        end_indices = np.append(end_indices, len(prices))
    
    if len(start_indices) == 0 or len(end_indices) == 0:
        return np.inf if ann_returns > risk_free_rate else 0.0
    
    # 각 에피소드의 최대 손실 계산
    max_drawdowns = []
    for start, end in zip(start_indices, end_indices):
        episode_dd = drawdown[start:end]
        if len(episode_dd) > 0:
            max_drawdowns.append(np.max(episode_dd))
    
    # 상위 n개 손실 선택
    max_drawdowns.sort(reverse=True)
    top_drawdowns = max_drawdowns[:min(n_drawdowns, len(max_drawdowns))]
    
    if not top_drawdowns:
        return np.inf if ann_returns > risk_free_rate else 0.0
    
    avg_dd = np.mean(top_drawdowns)
    
    if avg_dd < 1e-6:
        return np.inf if ann_returns > risk_free_rate else 0.0
    
    return (ann_returns - risk_free_rate) / avg_dd


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Information Ratio 계산

    Args:
        returns: 전략 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        Information Ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # 초과 수익률 계산
    if len(returns) != len(benchmark_returns):
        raise ValueError("returns와 benchmark_returns의 길이가 일치해야 합니다.")
    
    excess_returns = returns - benchmark_returns
    
    # 연율화된 초과 수익률
    ann_excess_returns = np.mean(excess_returns) * periods_per_year
    
    # 초과 수익률의 연율화된 변동성 (추적 오차)
    tracking_error = np.std(excess_returns) * np.sqrt(periods_per_year)
    
    if tracking_error < 1e-6:
        return 0.0
    
    return ann_excess_returns / tracking_error


def calculate_tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
    """
    Tail Ratio 계산 (우측 꼬리와 좌측 꼬리의 비율)

    Args:
        returns: 수익률 시계열
        percentile: 꼬리를 정의하는 백분위수
    
    Returns:
        Tail Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # 우측 꼬리 (상위 5%)과 좌측 꼬리 (하위 5%)
    upper_percentile = np.percentile(returns, 100 - percentile)
    lower_percentile = np.percentile(returns, percentile)
    
    upper_tail = returns[returns >= upper_percentile]
    lower_tail = returns[returns <= lower_percentile]
    
    if len(upper_tail) == 0 or len(lower_tail) == 0:
        return 0.0
    
    upper_abs_mean = np.mean(np.abs(upper_tail))
    lower_abs_mean = np.mean(np.abs(lower_tail))
    
    if lower_abs_mean < 1e-6:
        return np.inf
    
    return upper_abs_mean / lower_abs_mean


def calculate_skewness(returns: np.ndarray) -> float:
    """
    수익률 분포의 왜도 계산

    Args:
        returns: 수익률 시계열
    
    Returns:
        왜도 값
    """
    if len(returns) < 2:
        return 0.0
    
    return stats.skew(returns)


def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    수익률 분포의 첨도 계산

    Args:
        returns: 수익률 시계열
    
    Returns:
        첨도 값
    """
    if len(returns) < 2:
        return 0.0
    
    return stats.kurtosis(returns)


def calculate_downside_frequency(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    하락 빈도 계산

    Args:
        returns: 수익률 시계열
        threshold: 임계 수익률
    
    Returns:
        하락 빈도
    """
    if len(returns) == 0:
        return 0.0
    
    return np.mean(returns < threshold)


def calculate_downside_correlation(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    하락 상관관계 계산

    Args:
        returns: 전략 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열
        threshold: 임계 수익률
    
    Returns:
        하락 상관관계
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # 벤치마크 하락 기간 식별
    down_mask = benchmark_returns < threshold
    
    if np.sum(down_mask) < 2:
        return 0.0
    
    # 하락 기간의 상관관계 계산
    return np.corrcoef(returns[down_mask], benchmark_returns[down_mask])[0, 1]


def calculate_rolling_sharpe(
    returns: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    롤링 Sharpe 비율 계산

    Args:
        returns: 수익률 시계열
        window: 롤링 윈도우 크기
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        롤링 Sharpe 비율 시계열
    """
    rolling_sharpe = np.full(len(returns), np.nan)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        rolling_sharpe[i - 1] = calculate_sharpe_ratio(window_returns, risk_free_rate, periods_per_year)
    
    return rolling_sharpe


def calculate_rolling_sortino(
    returns: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_acceptable_return: float = 0.0
) -> np.ndarray:
    """
    롤링 Sortino 비율 계산

    Args:
        returns: 수익률 시계열
        window: 롤링 윈도우 크기
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
        min_acceptable_return: 최소 허용 수익률
    
    Returns:
        롤링 Sortino 비율 시계열
    """
    rolling_sortino = np.full(len(returns), np.nan)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        rolling_sortino[i - 1] = calculate_sortino_ratio(
            window_returns, risk_free_rate, periods_per_year, min_acceptable_return
        )
    
    return rolling_sortino


def calculate_drawdowns(prices: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    모든 손실(drawdown) 에피소드 계산

    Args:
        prices: 가격 시계열 데이터
    
    Returns:
        drawdown: 전체 손실률 시계열
        drawdown_details: 손실 에피소드 세부 정보 리스트
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    
    # 손실 에피소드 식별
    in_drawdown = drawdown > 0
    
    # 손실 상태 변화 식별
    drawdown_episodes = np.diff(np.concatenate([[0], in_drawdown.astype(int), [0]]))
    start_indices = np.where(drawdown_episodes == 1)[0]
    end_indices = np.where(drawdown_episodes == -1)[0] - 1
    
    drawdown_details = []
    
    for start, end in zip(start_indices, end_indices):
        if start <= end:  # 유효한 에피소드인지 확인
            episode_dd = drawdown[start:end+1]
            max_dd_idx = start + np.argmax(episode_dd)
            
            drawdown_details.append({
                'start_idx': int(start),
                'end_idx': int(end),
                'max_dd_idx': int(max_dd_idx),
                'start_price': float(prices[start]),
                'end_price': float(prices[end]),
                'max_dd_price': float(prices[max_dd_idx]),
                'peak_price': float(peak[max_dd_idx]),
                'recovery_time': int(end - max_dd_idx),
                'duration': int(end - start),
                'max_drawdown': float(np.max(episode_dd)),
                'drawdown_at_end': float(drawdown[end])
            })
    
    return drawdown, drawdown_details


def calculate_trade_statistics(
    trades: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    거래 통계 계산

    Args:
        trades: 거래 기록 리스트
    
    Returns:
        거래 통계 딕셔너리
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'expectancy': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'avg_trade_length': 0.0,
            'profitable_trades': 0,
            'unprofitable_trades': 0
        }
    
    # 수익 추출
    profits = [trade['profit'] for trade in trades if 'profit' in trade]
    
    if not profits:
        return {
            'total_trades': len(trades),
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'expectancy': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'avg_trade_length': 0.0,
            'profitable_trades': 0,
            'unprofitable_trades': 0
        }
    
    # 승/패 거래 구분
    winning_trades = [profit for profit in profits if profit > 0]
    losing_trades = [profit for profit in profits if profit <= 0]
    
    # 통계 계산
    total_trades = len(profits)
    profitable_trades = len(winning_trades)
    unprofitable_trades = len(losing_trades)
    
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
    
    avg_profit = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0
    
    profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else np.inf
    
    expectancy = (win_rate * avg_profit) - ((1 - win_rate) * abs(avg_loss))
    
    largest_profit = np.max(winning_trades) if winning_trades else 0.0
    largest_loss = np.min(losing_trades) if losing_trades else 0.0
    
    # 거래 길이 계산
    trade_lengths = [trade['duration'] for trade in trades if 'duration' in trade]
    avg_trade_length = np.mean(trade_lengths) if trade_lengths else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'expectancy': expectancy,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'largest_profit': largest_profit,
        'largest_loss': largest_loss,
        'avg_trade_length': avg_trade_length,
        'profitable_trades': profitable_trades,
        'unprofitable_trades': unprofitable_trades
    }


def calculate_regime_statistics(
    returns: np.ndarray,
    regime_labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    시장 레짐별 성과 통계 계산

    Args:
        returns: 수익률 시계열
        regime_labels: 레짐 레이블 시계열
    
    Returns:
        레짐별 통계 딕셔너리
    """
    if len(returns) != len(regime_labels):
        raise ValueError("returns와 regime_labels의 길이가 일치해야 합니다.")
    
    # 유니크 레짐 식별
    unique_regimes = np.unique(regime_labels)
    
    statistics = {}
    
    for regime in unique_regimes:
        # 현재 레짐 마스크
        mask = regime_labels == regime
        
        # 현재 레짐 수익률
        regime_returns = returns[mask]
        
        if len(regime_returns) == 0:
            continue
        
        # 기본 통계
        statistics[str(regime)] = {
            'count': len(regime_returns),
            'mean_return': np.mean(regime_returns),
            'std_return': np.std(regime_returns),
            'median_return': np.median(regime_returns),
            'win_rate': calculate_win_rate(regime_returns),
            'skewness': calculate_skewness(regime_returns),
            'kurtosis': calculate_kurtosis(regime_returns),
            'min_return': np.min(regime_returns),
            'max_return': np.max(regime_returns)
        }
    
    return statistics


def calculate_performance_summary(
    returns: np.ndarray,
    prices: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    성과 요약 통계 계산

    Args:
        returns: 수익률 시계열
        prices: 가격 시계열 데이터
        benchmark_returns: 벤치마크 수익률 시계열 (없으면 None)
        risk_free_rate: 무위험 이자율 (연율화된 값)
        periods_per_year: 연간 주기 수 (일별=252, 주별=52, 월별=12)
    
    Returns:
        성과 요약 통계 딕셔너리
    """
    if len(returns) == 0:
        return {}
    
    summary = {}
    
    # 수익률 통계
    summary['total_return'] = (prices[-1] / prices[0]) - 1
    summary['annualized_return'] = annualize_returns(returns, periods_per_year)
    summary['annualized_volatility'] = annualize_volatility(returns, periods_per_year)
    
    # 위험 조정 지표
    summary['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    summary['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    summary['calmar_ratio'] = calculate_calmar_ratio(returns, prices, risk_free_rate, periods_per_year)
    summary['omega_ratio'] = calculate_omega_ratio(returns, 0.0, periods_per_year)
    
    # 위험 지표
    summary['max_drawdown'] = calculate_max_drawdown(prices)
    summary['var_95'] = calculate_value_at_risk(returns, 0.95)
    summary['cvar_95'] = calculate_conditional_value_at_risk(returns, 0.95)
    summary['avg_drawdown'] = calculate_average_drawdown(prices)
    summary['ulcer_index'] = calculate_ulcer_index(prices)
    
    # 거래 효율성 지표
    summary['win_rate'] = calculate_win_rate(returns)
    summary['profit_loss_ratio'] = calculate_profit_loss_ratio(returns)
    summary['expectancy'] = calculate_expectancy(returns)
    
    # 분포 통계
    summary['skewness'] = calculate_skewness(returns)
    summary['kurtosis'] = calculate_kurtosis(returns)
    summary['tail_ratio'] = calculate_tail_ratio(returns)
    
    # 벤치마크 비교 지표
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        summary['information_ratio'] = calculate_information_ratio(
            returns, benchmark_returns, periods_per_year
        )
        summary['downside_correlation'] = calculate_downside_correlation(
            returns, benchmark_returns
        )
        
        # 벤치마크 대비 베타
        if np.std(benchmark_returns) > 0:
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            summary['beta'] = beta
            
            # 알파 계산 (CAPM)
            benchmark_ann_return = annualize_returns(benchmark_returns, periods_per_year)
            expected_return = risk_free_rate + beta * (benchmark_ann_return - risk_free_rate)
            summary['alpha'] = summary['annualized_return'] - expected_return
            
            # 트레이킹 에러
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(periods_per_year)
            summary['tracking_error'] = tracking_error
            
            # 상/하방 캡처 비율
            up_market = benchmark_returns > 0
            down_market = benchmark_returns < 0
            
            if np.sum(up_market) > 0 and np.sum(benchmark_returns[up_market]) != 0:
                up_capture = np.sum(returns[up_market]) / np.sum(benchmark_returns[up_market])
                summary['up_capture_ratio'] = up_capture
            
            if np.sum(down_market) > 0 and np.sum(benchmark_returns[down_market]) != 0:
                down_capture = np.sum(returns[down_market]) / np.sum(benchmark_returns[down_market])
                summary['down_capture_ratio'] = down_capture
    
    return summary


def detect_market_regimes(
    returns: np.ndarray,
    n_regimes: int = 4,
    window: int = 252,
    method: str = 'hmm'
) -> np.ndarray:
    """
    시장 레짐 감지

    Args:
        returns: 수익률 시계열
        n_regimes: 레짐 수
        window: 롤링 윈도우 크기
        method: 감지 방법 ('hmm', 'kmeans', 'volatility')
    
    Returns:
        레짐 레이블 시계열
    """
    if method == 'hmm':
        try:
            from hmmlearn import hmm
            
            # 특성 생성 (수익률, 변동성)
            features = np.column_stack([
                returns,
                np.abs(returns),  # 변동성 프록시
                np.roll(returns, 1),  # 지연된 수익률
                np.roll(np.abs(returns), 1)  # 지연된 변동성
            ])
            
            # 첫 행의 NaN 제거
            features = features[1:]
            
            # HMM 모델링
            model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            
            model.fit(features)
            
            # 레짐 예측
            regimes = model.predict(features)
            
            # NaN 처리를 위한 패딩
            regimes_full = np.full(len(returns), np.nan)
            regimes_full[1:] = regimes
            
            return regimes_full
        
        except ImportError:
            print("hmmlearn 패키지가 설치되어 있지 않습니다. 'volatility' 방법을 사용합니다.")
            method = 'volatility'
    
    if method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            
            # 롤링 통계
            rolling_mean = np.full(len(returns), np.nan)
            rolling_std = np.full(len(returns), np.nan)
            
            for i in range(window, len(returns) + 1):
                window_returns = returns[i - window:i]
                rolling_mean[i - 1] = np.mean(window_returns)
                rolling_std[i - 1] = np.std(window_returns)
            
            # 특성 생성
            features = np.column_stack([rolling_mean, rolling_std])
            
            # NaN 제거
            mask = ~np.isnan(features).any(axis=1)
            valid_features = features[mask]
            
            # KMeans 클러스터링
            model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = model.fit_predict(valid_features)
            
            # 결과 라벨링
            regimes = np.full(len(returns), np.nan)
            regimes[mask] = labels
            
            # NaN을 앞쪽 값으로 채우기
            for i in range(len(regimes)):
                if np.isnan(regimes[i]) and i > 0:
                    regimes[i] = regimes[i-1]
            
            return regimes
        
        except ImportError:
            print("scikit-learn 패키지가 설치되어 있지 않습니다. 'volatility' 방법을 사용합니다.")
            method = 'volatility'
    
    if method == 'volatility':
        # 롤링 변동성
        rolling_vol = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
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
        
        # NaN을 앞쪽 값으로 채우기
        for i in range(len(regimes)):
            if np.isnan(regimes[i]) and i > 0:
                regimes[i] = regimes[i-1]
        
        return regimes
    
    raise ValueError(f"지원되지 않는 감지 방법: {method}")
