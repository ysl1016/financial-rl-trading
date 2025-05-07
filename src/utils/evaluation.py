"""
트레이딩 모델 성능 평가를 위한 다양한 금융 지표 및 평가 함수를 제공합니다.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy import stats
import warnings


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    포트폴리오 가치로부터 수익률을 계산합니다.
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        
    Returns:
        일별 수익률 배열
    """
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def calculate_cumulative_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    포트폴리오 가치로부터 누적 수익률을 계산합니다.
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        
    Returns:
        누적 수익률 배열
    """
    initial_value = portfolio_values[0]
    cumulative_returns = np.array(portfolio_values) / initial_value - 1
    return cumulative_returns


def calculate_annualized_return(portfolio_values: List[float], 
                               trading_days: int = 252) -> float:
    """
    포트폴리오 가치로부터 연간화된 수익률을 계산합니다.
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        trading_days: 연간 거래일 수
        
    Returns:
        연간화된 수익률
    """
    total_days = len(portfolio_values) - 1
    if total_days < 1:
        return 0.0
    
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    years = total_days / trading_days
    
    # 복리 수익률 공식 사용
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    return annualized_return


def calculate_volatility(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    일별 수익률로부터 연간화된 변동성을 계산합니다.
    
    Args:
        returns: 일별 수익률 배열
        trading_days: 연간 거래일 수
        
    Returns:
        연간화된 변동성
    """
    daily_volatility = np.std(returns)
    annualized_volatility = daily_volatility * np.sqrt(trading_days)
    return annualized_volatility


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = 0.0, 
                          trading_days: int = 252) -> float:
    """
    Sharpe 비율을 계산합니다.
    
    Args:
        returns: 일별 수익률 배열
        risk_free_rate: 연간 무위험 수익률
        trading_days: 연간 거래일 수
        
    Returns:
        Sharpe 비율
    """
    # 일별 무위험 수익률로 변환
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    excess_returns = returns - daily_risk_free
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days)
    
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray, 
                           risk_free_rate: float = 0.0, 
                           trading_days: int = 252) -> float:
    """
    Sortino 비율을 계산합니다 (하방 위험만 고려).
    
    Args:
        returns: 일별 수익률 배열
        risk_free_rate: 연간 무위험 수익률
        trading_days: 연간 거래일 수
        
    Returns:
        Sortino 비율
    """
    # 일별 무위험 수익률로 변환
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    excess_returns = returns - daily_risk_free
    
    # 하방 수익률만 필터링
    downside_returns = excess_returns[excess_returns < 0]
    
    # 하방 수익률이 없는 경우 처리
    if len(downside_returns) == 0:
        warnings.warn("하방 수익률이 없습니다. Sortino 비율을 계산할 수 없습니다.")
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(trading_days)
    
    return sortino


def calculate_max_drawdown(portfolio_values: List[float]) -> Tuple[float, int, int]:
    """
    최대 손실(drawdown)을 계산합니다.
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        
    Returns:
        최대 손실 비율, 시작 인덱스, 종료 인덱스
    """
    # 최대 누적 값 계산
    portfolio_values = np.array(portfolio_values)
    peak_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak_values) / peak_values
    
    # 최대 손실 계산
    max_drawdown_idx = np.argmin(drawdowns)
    max_drawdown = drawdowns[max_drawdown_idx]
    
    # 최대 손실 기간의 시작점 찾기
    peak_idx = np.where(portfolio_values[:max_drawdown_idx] == peak_values[max_drawdown_idx])[0]
    if len(peak_idx) > 0:
        peak_idx = peak_idx[-1]
    else:
        peak_idx = 0
    
    return max_drawdown, int(peak_idx), int(max_drawdown_idx)


def calculate_calmar_ratio(portfolio_values: List[float], 
                          trading_days: int = 252) -> float:
    """
    Calmar 비율을 계산합니다 (연간 수익률 / 최대 손실).
    
    Args:
        portfolio_values: 포트폴리오 가치 시계열
        trading_days: 연간 거래일 수
        
    Returns:
        Calmar 비율
    """
    # 연간화된 수익률 계산
    annual_return = calculate_annualized_return(portfolio_values, trading_days)
    
    # 최대 손실 계산
    max_drawdown, _, _ = calculate_max_drawdown(portfolio_values)
    
    # 최대 손실이 0인 경우 처리
    if max_drawdown == 0:
        warnings.warn("최대 손실이 0입니다. Calmar 비율을 계산할 수 없습니다.")
        return np.inf
    
    # Calmar 비율 계산
    calmar_ratio = annual_return / abs(max_drawdown)
    
    return calmar_ratio


def calculate_omega_ratio(returns: np.ndarray, 
                         threshold: float = 0.0, 
                         trading_days: int = 252) -> float:
    """
    Omega 비율을 계산합니다 (상방 부분 모멘트 / 하방 부분 모멘트).
    
    Args:
        returns: 일별 수익률 배열
        threshold: 목표 수익률 임계값 (일별)
        trading_days: 연간 거래일 수
        
    Returns:
        Omega 비율
    """
    # 임계값 조정
    if threshold > 1:
        # 연간 임계값인 경우 일별로 변환
        threshold = (1 + threshold) ** (1 / trading_days) - 1
    
    # 임계값 기준 상방/하방 수익률 분리
    excess_returns = returns - threshold
    upside_returns = excess_returns[excess_returns >= 0]
    downside_returns = excess_returns[excess_returns < 0]
    
    # 하방 수익률이 없는 경우 처리
    if len(downside_returns) == 0:
        warnings.warn("하방 수익률이 없습니다. Omega 비율을 계산할 수 없습니다.")
        return np.inf
    
    # 상방 및 하방 부분 모멘트 계산
    upside_mean = np.sum(upside_returns)
    downside_mean = np.sum(np.abs(downside_returns))
    
    # Omega 비율 계산
    omega_ratio = upside_mean / downside_mean
    
    return omega_ratio


def calculate_var(returns: np.ndarray, 
                 confidence_level: float = 0.95) -> float:
    """
    Value at Risk (VaR)를 계산합니다.
    
    Args:
        returns: 일별 수익률 배열
        confidence_level: 신뢰수준 (0-1)
        
    Returns:
        VaR 값 (양수로 표현)
    """
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return abs(var)


def calculate_cvar(returns: np.ndarray, 
                  confidence_level: float = 0.95) -> float:
    """
    Conditional Value at Risk (CVaR)를 계산합니다.
    
    Args:
        returns: 일별 수익률 배열
        confidence_level: 신뢰수준 (0-1)
        
    Returns:
        CVaR 값 (양수로 표현)
    """
    var = np.percentile(returns, 100 * (1 - confidence_level))
    cvar = np.mean(returns[returns <= var])
    return abs(cvar)


def calculate_win_rate(trades: List[Tuple]) -> float:
    """
    거래의 승률을 계산합니다.
    
    Args:
        trades: 거래 데이터 (type, index, price) 튜플 목록
        
    Returns:
        승리 거래 비율
    """
    if not trades:
        return 0.0
    
    profits = []
    position = 0
    entry_price = 0
    
    for trade_type, _, price in trades:
        if trade_type == 'buy':
            position = 1
            entry_price = price
        elif trade_type == 'short':
            position = -1
            entry_price = price
        elif trade_type == 'close_long' and position == 1:
            profits.append(price - entry_price)
            position = 0
        elif trade_type == 'close_short' and position == -1:
            profits.append(entry_price - price)
            position = 0
    
    if not profits:
        return 0.0
    
    win_count = sum(1 for p in profits if p > 0)
    return win_count / len(profits)


def calculate_profit_loss_ratio(trades: List[Tuple]) -> float:
    """
    거래의 수익-손실 비율을 계산합니다.
    
    Args:
        trades: 거래 데이터 (type, index, price) 튜플 목록
        
    Returns:
        평균 수익 / 평균 손실
    """
    if not trades:
        return 0.0
    
    profits = []
    position = 0
    entry_price = 0
    
    for trade_type, _, price in trades:
        if trade_type == 'buy':
            position = 1
            entry_price = price
        elif trade_type == 'short':
            position = -1
            entry_price = price
        elif trade_type == 'close_long' and position == 1:
            profits.append(price - entry_price)
            position = 0
        elif trade_type == 'close_short' and position == -1:
            profits.append(entry_price - price)
            position = 0
    
    if not profits:
        return 0.0
    
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    
    if not losses:
        return np.inf
    if not wins:
        return 0.0
    
    avg_win = np.mean(wins)
    avg_loss = np.mean(np.abs(losses))
    
    return avg_win / avg_loss


def calculate_expectancy(trades: List[Tuple]) -> float:
    """
    거래의 기대 수익을 계산합니다.
    
    Args:
        trades: 거래 데이터 (type, index, price) 튜플 목록
        
    Returns:
        기대 수익 (win_rate * avg_win - (1 - win_rate) * avg_loss)
    """
    if not trades:
        return 0.0
    
    profits = []
    position = 0
    entry_price = 0
    
    for trade_type, _, price in trades:
        if trade_type == 'buy':
            position = 1
            entry_price = price
        elif trade_type == 'short':
            position = -1
            entry_price = price
        elif trade_type == 'close_long' and position == 1:
            profits.append(price - entry_price)
            position = 0
        elif trade_type == 'close_short' and position == -1:
            profits.append(entry_price - price)
            position = 0
    
    if not profits:
        return 0.0
    
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    
    win_rate = len(wins) / len(profits) if profits else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(np.abs(losses)) if losses else 0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    return expectancy


def calculate_avg_trade_duration(trades: List[Tuple], 
                                positions: List[int]) -> float:
    """
    평균 거래 지속 시간을 계산합니다.
    
    Args:
        trades: 거래 데이터 (type, index, price) 튜플 목록
        positions: 각 시점의 포지션
        
    Returns:
        평균 거래 지속 시간 (시장 기간)
    """
    if not trades:
        return 0.0
    
    durations = []
    current_position = 0
    entry_index = 0
    
    for trade_type, index, _ in trades:
        if trade_type in ['buy', 'short'] and current_position == 0:
            current_position = 1 if trade_type == 'buy' else -1
            entry_index = index
        elif trade_type in ['close_long', 'close_short'] and current_position != 0:
            durations.append(index - entry_index)
            current_position = 0
    
    if not durations:
        return 0.0
    
    return np.mean(durations)


def calculate_comprehensive_metrics(backtest_results: Dict[str, Any], 
                                   risk_free_rate: float = 0.02,
                                   trading_days: int = 252) -> Dict[str, float]:
    """
    백테스트 결과에서 종합적인 성과 지표를 계산합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        risk_free_rate: 연간 무위험 수익률
        trading_days: 연간 거래일 수
        
    Returns:
        계산된 모든 성과 지표가 포함된 딕셔너리
    """
    metrics = {}
    
    # 기본 데이터 추출
    portfolio_values = backtest_results.get('portfolio_values', [])
    trades = backtest_results.get('trades', [])
    positions = backtest_results.get('positions', [])
    
    if not portfolio_values or len(portfolio_values) < 2:
        warnings.warn("포트폴리오 가치 데이터가 부족합니다. 지표를 계산할 수 없습니다.")
        return metrics
    
    # 수익률 계산
    returns = calculate_returns(portfolio_values)
    
    # 기본 성과 지표
    metrics['total_return'] = portfolio_values[-1] / portfolio_values[0] - 1
    metrics['annualized_return'] = calculate_annualized_return(portfolio_values, trading_days)
    metrics['volatility'] = calculate_volatility(returns, trading_days)
    
    # 위험 조정 지표
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, trading_days)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, trading_days)
    
    # 손실 및 리스크 지표
    max_dd, peak_idx, valley_idx = calculate_max_drawdown(portfolio_values)
    metrics['max_drawdown'] = max_dd
    metrics['max_drawdown_duration'] = valley_idx - peak_idx
    metrics['calmar_ratio'] = calculate_calmar_ratio(portfolio_values, trading_days)
    
    # 고급 지표
    metrics['omega_ratio'] = calculate_omega_ratio(returns, risk_free_rate / trading_days, trading_days)
    metrics['var_95'] = calculate_var(returns, 0.95)
    metrics['cvar_95'] = calculate_cvar(returns, 0.95)
    
    # 거래 관련 지표
    if trades:
        metrics['trade_count'] = len(trades) // 2  # 매수-매도 쌍으로 계산
        metrics['win_rate'] = calculate_win_rate(trades)
        metrics['profit_loss_ratio'] = calculate_profit_loss_ratio(trades)
        metrics['expectancy'] = calculate_expectancy(trades)
        
        if positions:
            metrics['avg_trade_duration'] = calculate_avg_trade_duration(trades, positions)
    
    # 수익성 지표
    metrics['profit_factor'] = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.any(returns < 0) else np.inf
    metrics['avg_return'] = np.mean(returns)
    metrics['max_return'] = np.max(returns)
    metrics['min_return'] = np.min(returns)
    
    return metrics


def compare_strategies(strategy_results: Dict[str, Dict[str, Any]], 
                      benchmark_results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    여러 전략의 성과를 비교합니다.
    
    Args:
        strategy_results: 각 전략별 백테스트 결과의 딕셔너리
        benchmark_results: 벤치마크 결과 (선택 사항)
        
    Returns:
        전략 비교 DataFrame
    """
    # 모든 전략 성과 지표 계산
    strategy_metrics = {}
    
    for name, results in strategy_results.items():
        metrics = calculate_comprehensive_metrics(results)
        strategy_metrics[name] = metrics
    
    # 벤치마크가 있는 경우 추가
    if benchmark_results:
        benchmark_metrics = calculate_comprehensive_metrics(benchmark_results)
        strategy_metrics['Benchmark'] = benchmark_metrics
    
    # DataFrame 생성
    metrics_df = pd.DataFrame(strategy_metrics)
    
    # 정보 비율 계산 (벤치마크 대비 초과 수익)
    if benchmark_results and 'portfolio_values' in benchmark_results:
        benchmark_returns = calculate_returns(benchmark_results['portfolio_values'])
        
        for name, results in strategy_results.items():
            if 'portfolio_values' in results and len(results['portfolio_values']) > 1:
                strategy_returns = calculate_returns(results['portfolio_values'])
                
                # 기간 맞추기 (더 짧은 쪽에 맞춤)
                min_length = min(len(strategy_returns), len(benchmark_returns))
                tracking_error = np.std(strategy_returns[:min_length] - benchmark_returns[:min_length])
                
                if tracking_error > 0:
                    excess_return = (results['portfolio_values'][-1] / results['portfolio_values'][0]) - \
                                   (benchmark_results['portfolio_values'][-1] / benchmark_results['portfolio_values'][0])
                    info_ratio = excess_return / tracking_error
                    metrics_df.loc['information_ratio', name] = info_ratio
    
    return metrics_df


def statistical_significance_test(returns1: np.ndarray, 
                                 returns2: np.ndarray,
                                 test_type: str = 't-test') -> Dict[str, float]:
    """
    두 수익률 시리즈간의 통계적 유의성을 테스트합니다.
    
    Args:
        returns1: 첫번째 전략의 수익률 배열
        returns2: 두번째 전략의 수익률 배열
        test_type: 테스트 유형 ('t-test' 또는 'wilcoxon')
        
    Returns:
        테스트 결과를 포함하는 딕셔너리
    """
    # 길이 맞추기
    min_length = min(len(returns1), len(returns2))
    returns1 = returns1[:min_length]
    returns2 = returns2[:min_length]
    
    result = {}
    
    if test_type == 't-test':
        # 독립 표본 t-검정
        t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)
        result['test_type'] = 't-test'
        result['t_statistic'] = t_stat
        result['p_value'] = p_value
        result['significant'] = p_value < 0.05
    
    elif test_type == 'wilcoxon':
        # Wilcoxon 부호 순위 검정 (비모수적)
        w_stat, p_value = stats.wilcoxon(returns1, returns2)
        result['test_type'] = 'wilcoxon'
        result['w_statistic'] = w_stat
        result['p_value'] = p_value
        result['significant'] = p_value < 0.05
    
    else:
        raise ValueError(f"지원되지 않는 테스트 유형: {test_type}. 't-test' 또는 'wilcoxon'을 사용하세요.")
    
    return result
