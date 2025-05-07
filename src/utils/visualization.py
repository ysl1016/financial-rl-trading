"""
백테스팅 결과 시각화를 위한 다양한 차트 및 시각화 함수를 제공합니다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from matplotlib.gridspec import GridSpec
from datetime import datetime


def plot_portfolio_performance(backtest_results: Dict[str, Any], 
                              benchmark_values: Optional[List[float]] = None,
                              benchmark_name: str = 'Benchmark',
                              title: str = 'Portfolio Performance',
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    포트폴리오 성과를 시각화합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        benchmark_values: 벤치마크 포트폴리오 가치 (선택 사항)
        benchmark_name: 벤치마크 이름
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    portfolio_values = backtest_results.get('portfolio_values', [])
    dates = backtest_results.get('dates', None)
    
    if not portfolio_values:
        raise ValueError("포트폴리오 가치 데이터가 없습니다.")
    
    # 날짜 데이터가 없는 경우 인덱스 생성
    if dates is None or len(dates) != len(portfolio_values):
        dates = list(range(len(portfolio_values)))
    
    # 포트폴리오 가치 정규화
    normalized_portfolio = np.array(portfolio_values) / portfolio_values[0]
    
    # 벤치마크 정규화 (있는 경우)
    if benchmark_values and len(benchmark_values) > 0:
        min_length = min(len(normalized_portfolio), len(benchmark_values))
        normalized_benchmark = np.array(benchmark_values[:min_length]) / benchmark_values[0]
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # 포트폴리오 가치 차트
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, normalized_portfolio, label='Strategy', color='#1f77b4', linewidth=2)
    
    if benchmark_values and len(benchmark_values) > 0:
        ax1.plot(dates[:min_length], normalized_benchmark, label=benchmark_name, color='#ff7f0e', linewidth=2, linestyle='--')
    
    # 매수/매도 표시 (있는 경우)
    if 'trades' in backtest_results and backtest_results['trades']:
        buy_indices = []
        buy_values = []
        sell_indices = []
        sell_values = []
        
        for trade_type, idx, price in backtest_results['trades']:
            if trade_type == 'buy':
                buy_indices.append(idx)
                buy_values.append(normalized_portfolio[idx])
            elif trade_type == 'short':
                sell_indices.append(idx)
                sell_values.append(normalized_portfolio[idx])
        
        if buy_indices:
            ax1.scatter(
                [dates[i] for i in buy_indices],
                buy_values,
                color='green',
                label='Buy',
                marker='^',
                alpha=0.7,
                s=100
            )
        
        if sell_indices:
            ax1.scatter(
                [dates[i] for i in sell_indices],
                sell_values,
                color='red',
                label='Sell',
                marker='v',
                alpha=0.7,
                s=100
            )
    
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Normalized Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # 수익률 차트
    ax2 = fig.add_subplot(gs[1])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    ax2.bar(dates[1:], returns, alpha=0.5, color=np.where(returns >= 0, 'green', 'red'))
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 드로다운 차트
    ax3 = fig.add_subplot(gs[2])
    max_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - max_values) / max_values
    ax3.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
    ax3.set_ylabel('Drawdown', fontsize=12)
    ax3.set_xlabel('Date' if isinstance(dates[0], datetime) else 'Trading Days', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 날짜 형식 포맷 (날짜 객체인 경우)
    if isinstance(dates[0], datetime):
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            if len(dates) > 50:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_return_distribution(backtest_results: Dict[str, Any],
                            benchmark_values: Optional[List[float]] = None,
                            benchmark_name: str = 'Benchmark',
                            title: str = 'Return Distribution',
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    수익률 분포를 시각화합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        benchmark_values: 벤치마크 포트폴리오 가치 (선택 사항)
        benchmark_name: 벤치마크 이름
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    portfolio_values = backtest_results.get('portfolio_values', [])
    
    if not portfolio_values or len(portfolio_values) < 2:
        raise ValueError("포트폴리오 가치 데이터가 부족합니다.")
    
    # 수익률 계산
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # 벤치마크 수익률 계산 (있는 경우)
    benchmark_returns = None
    if benchmark_values and len(benchmark_values) > 1:
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1], hspace=0.25, wspace=0.1)
    
    # 히스토그램 플롯
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(returns, bins=50, kde=True, color='#1f77b4', alpha=0.6, label='Strategy', ax=ax1)
    
    if benchmark_returns is not None:
        min_length = min(len(returns), len(benchmark_returns))
        sns.histplot(benchmark_returns[:min_length], bins=50, kde=True, color='#ff7f0e', alpha=0.4, label=benchmark_name, ax=ax1)
    
    ax1.axvline(x=np.mean(returns), color='#1f77b4', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
    
    if benchmark_returns is not None:
        ax1.axvline(x=np.mean(benchmark_returns), color='#ff7f0e', linestyle='--', label=f'{benchmark_name} Mean: {np.mean(benchmark_returns):.4f}')
    
    ax1.set_title(f'{title} - Histogram', fontsize=14)
    ax1.set_xlabel('Daily Returns', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 박스 플롯
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [returns]
    box_labels = ['Strategy']
    
    if benchmark_returns is not None:
        box_data.append(benchmark_returns[:min_length])
        box_labels.append(benchmark_name)
    
    ax2.boxplot(box_data, labels=box_labels, vert=True, patch_artist=True,
               boxprops=dict(facecolor='#1f77b4', alpha=0.6))
    ax2.set_title('Box Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # QQ 플롯
    ax3 = fig.add_subplot(gs[1, 0])
    from scipy import stats
    
    # 정규 QQ 플롯
    qq = stats.probplot(returns, dist="norm", plot=ax3)
    ax3.set_title('QQ Plot (vs. Normal Distribution)', fontsize=14)
    ax3.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax3.set_ylabel('Sample Quantiles', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 통계 데이터
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # 통계 계산
    mean = np.mean(returns)
    std = np.std(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    var_95 = np.percentile(returns, 5)
    sharpe = mean / std * np.sqrt(252)  # 연간화된 Sharpe Ratio
    
    stats_text = (
        f"Mean: {mean:.4f}\n"
        f"Std Dev: {std:.4f}\n"
        f"Annualized Vol: {std * np.sqrt(252):.4f}\n"
        f"Skewness: {skew:.4f}\n"
        f"Kurtosis: {kurt:.4f}\n"
        f"VaR (95%): {abs(var_95):.4f}\n"
        f"Sharpe Ratio: {sharpe:.4f}\n"
    )
    
    ax4.text(0, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    return fig


def plot_rolling_metrics(backtest_results: Dict[str, Any],
                        window: int = 60,
                        title: str = 'Rolling Performance Metrics',
                        figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
    """
    롤링 성과 지표를 시각화합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        window: 롤링 윈도우 크기
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    portfolio_values = backtest_results.get('portfolio_values', [])
    dates = backtest_results.get('dates', None)
    
    if not portfolio_values or len(portfolio_values) < window:
        raise ValueError(f"포트폴리오 가치 데이터가 부족합니다. 최소 {window}개가 필요합니다.")
    
    # 날짜 데이터가 없는 경우 인덱스 생성
    if dates is None or len(dates) != len(portfolio_values):
        dates = list(range(len(portfolio_values)))
    
    # 일별 수익률 계산
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # 롤링 지표 계산
    rolling_returns = []
    rolling_volatility = []
    rolling_sharpe = []
    rolling_sortino = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        
        # 롤링 수익률 (연간화)
        period_return = (portfolio_values[i] / portfolio_values[i - window]) - 1
        annual_return = (1 + period_return) ** (252 / window) - 1
        rolling_returns.append(annual_return)
        
        # 롤링 변동성 (연간화)
        vol = np.std(window_returns) * np.sqrt(252)
        rolling_volatility.append(vol)
        
        # 롤링 Sharpe
        if vol > 0:
            sharpe = annual_return / vol
            rolling_sharpe.append(sharpe)
        else:
            rolling_sharpe.append(0)
        
        # 롤링 Sortino
        downside_returns = window_returns[window_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = annual_return / (np.std(downside_returns) * np.sqrt(252))
            rolling_sortino.append(sortino)
        else:
            rolling_sortino.append(0)
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
    
    # 롤링 수익률 차트
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates[window:], rolling_returns, color='#1f77b4', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'{title} - {window}-Day Rolling Annualized Return', fontsize=14)
    ax1.set_ylabel('Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 롤링 변동성 차트
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates[window:], rolling_volatility, color='#ff7f0e', linewidth=2)
    ax2.set_title(f'{window}-Day Rolling Annualized Volatility', fontsize=14)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 롤링 Sharpe 차트
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates[window:], rolling_sharpe, color='#2ca02c', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14)
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 롤링 Sortino 차트
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(dates[window:], rolling_sortino, color='#d62728', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title(f'{window}-Day Rolling Sortino Ratio', fontsize=14)
    ax4.set_ylabel('Sortino Ratio', fontsize=12)
    ax4.set_xlabel('Date' if isinstance(dates[0], datetime) else 'Trading Days', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 날짜 형식 포맷 (날짜 객체인 경우)
    if isinstance(dates[0], datetime):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            if len(dates) > 50:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_market_regimes(backtest_results: Dict[str, Any],
                       market_regimes: List[str],
                       title: str = 'Performance Across Market Regimes',
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    시장 레짐별 성과를 시각화합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        market_regimes: 시장 레짐 레이블 목록
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    portfolio_values = backtest_results.get('portfolio_values', [])
    dates = backtest_results.get('dates', None)
    
    if not portfolio_values:
        raise ValueError("포트폴리오 가치 데이터가 없습니다.")
    
    if not market_regimes or len(market_regimes) != len(portfolio_values) - 1:
        raise ValueError("시장 레짐 데이터가 잘못되었습니다.")
    
    # 날짜 데이터가 없는 경우 인덱스 생성
    if dates is None or len(dates) != len(portfolio_values):
        dates = list(range(len(portfolio_values)))
    
    # 일별 수익률 계산
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # 각 레짐별 수익률 분리
    unique_regimes = sorted(set(market_regimes))
    regime_returns = {regime: [] for regime in unique_regimes}
    
    for i, regime in enumerate(market_regimes):
        regime_returns[regime].append(returns[i])
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # 포트폴리오 가치 차트 (레짐별 색상)
    ax1 = fig.add_subplot(gs[0])
    
    # 시작점은 항상 1
    cumulative_returns = [1.0]
    
    # 누적 수익률 계산
    for r in returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))
    
    # 각 레짐별 구간 시각화
    for regime in unique_regimes:
        regime_indices = [i for i, r in enumerate(market_regimes) if r == regime]
        
        if regime_indices:
            # 연속된 구간 찾기
            segments = []
            current_segment = [regime_indices[0]]
            
            for i in range(1, len(regime_indices)):
                if regime_indices[i] == regime_indices[i-1] + 1:
                    current_segment.append(regime_indices[i])
                else:
                    segments.append(current_segment)
                    current_segment = [regime_indices[i]]
            
            segments.append(current_segment)
            
            # 각 구간 플롯
            for segment in segments:
                start, end = segment[0], segment[-1]
                ax1.plot(dates[start:end+2], cumulative_returns[start:end+2],
                        label=regime if len(segments) == 1 else None)
    
    # 레짐 전환점 표시
    regime_changes = []
    for i in range(1, len(market_regimes)):
        if market_regimes[i] != market_regimes[i-1]:
            regime_changes.append(i)
    
    for change in regime_changes:
        ax1.axvline(x=dates[change], color='black', linestyle='--', alpha=0.5)
    
    ax1.set_title(f'{title} - Cumulative Returns', fontsize=14)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Market Regime', fontsize=10)
    
    # 레짐별 평균 수익률 막대 차트
    ax2 = fig.add_subplot(gs[1])
    regime_mean_returns = {regime: np.mean(returns) if returns else 0 for regime, returns in regime_returns.items()}
    
    colors = plt.cm.tab10(range(len(unique_regimes)))
    bars = ax2.bar(unique_regimes, [regime_mean_returns[regime] for regime in unique_regimes],
                  color=colors)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)
    
    ax2.set_title('Average Daily Return by Market Regime', fontsize=14)
    ax2.set_ylabel('Average Return', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 레짐별 변동성 막대 차트
    ax3 = fig.add_subplot(gs[2])
    regime_volatility = {regime: np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
                        for regime, returns in regime_returns.items()}
    
    bars = ax3.bar(unique_regimes, [regime_volatility[regime] for regime in unique_regimes],
                  color=colors)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom',
                fontsize=10)
    
    ax3.set_title('Annualized Volatility by Market Regime', fontsize=14)
    ax3.set_ylabel('Volatility', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_trade_analysis(backtest_results: Dict[str, Any],
                       title: str = 'Trade Analysis',
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    거래 분석 결과를 시각화합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    trades = backtest_results.get('trades', [])
    
    if not trades:
        raise ValueError("거래 데이터가 없습니다.")
    
    # 거래 데이터 처리
    profits = []
    trade_types = []
    durations = []
    
    position = 0
    entry_price = 0
    entry_idx = 0
    
    for trade_type, idx, price in trades:
        if trade_type == 'buy':
            position = 1
            entry_price = price
            entry_idx = idx
        elif trade_type == 'short':
            position = -1
            entry_price = price
            entry_idx = idx
        elif trade_type == 'close_long' and position == 1:
            profit = price - entry_price
            profits.append(profit)
            trade_types.append('long')
            durations.append(idx - entry_idx)
            position = 0
        elif trade_type == 'close_short' and position == -1:
            profit = entry_price - price
            profits.append(profit)
            trade_types.append('short')
            durations.append(idx - entry_idx)
            position = 0
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # 이익/손실 분포 히스토그램
    ax1 = fig.add_subplot(gs[0, 0])
    
    if profits:
        sns.histplot(profits, bins=20, kde=True, color='#1f77b4', ax=ax1)
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.axvline(x=np.mean(profits), color='green', linestyle='-', 
                   label=f'Mean: {np.mean(profits):.4f}')
    
    ax1.set_title(f'{title} - Profit/Loss Distribution', fontsize=14)
    ax1.set_xlabel('Profit/Loss', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 롱/숏 거래별 수익 비교
    ax2 = fig.add_subplot(gs[0, 1])
    
    if profits and trade_types:
        long_profits = [p for p, t in zip(profits, trade_types) if t == 'long']
        short_profits = [p for p, t in zip(profits, trade_types) if t == 'short']
        
        data = []
        labels = []
        
        if long_profits:
            data.append(long_profits)
            labels.append('Long')
        
        if short_profits:
            data.append(short_profits)
            labels.append('Short')
        
        ax2.boxplot(data, labels=labels, patch_artist=True, 
                   boxprops=dict(facecolor='#1f77b4', alpha=0.6))
        
        # 평균 표시
        for i, d in enumerate(data):
            if d:
                ax2.scatter([i+1], [np.mean(d)], color='red', s=50, zorder=3,
                           label='Mean' if i == 0 else "")
    
    ax2.set_title('Profit/Loss by Trade Direction', fontsize=14)
    ax2.set_ylabel('Profit/Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 거래 지속 시간 히스토그램
    ax3 = fig.add_subplot(gs[1, 0])
    
    if durations:
        sns.histplot(durations, bins=20, kde=True, color='#ff7f0e', ax=ax3)
        ax3.axvline(x=np.mean(durations), color='red', linestyle='-', 
                   label=f'Mean: {np.mean(durations):.1f}')
    
    ax3.set_title('Trade Duration Distribution', fontsize=14)
    ax3.set_xlabel('Duration (Days)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 수익과 지속 시간의 관계
    ax4 = fig.add_subplot(gs[1, 1])
    
    if profits and durations:
        ax4.scatter(durations, profits, alpha=0.7, c=np.array(profits) > 0, 
                   cmap='RdYlGn', vmin=-max(abs(min(profits)), abs(max(profits))),
                   vmax=max(abs(min(profits)), abs(max(profits))))
        
        # 추세선
        try:
            z = np.polyfit(durations, profits, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(durations), max(durations), 100)
            ax4.plot(x_range, p(x_range), "r--", alpha=0.8,
                     label=f"Trend: y={z[0]:.6f}x+{z[1]:.4f}")
            
            # 상관관계 계산
            corr = np.corrcoef(durations, profits)[0, 1]
            ax4.text(0.05, 0.95, f"Correlation: {corr:.4f}", transform=ax4.transAxes,
                    verticalalignment='top', fontsize=10)
        except:
            pass
    
    ax4.set_title('Profit/Loss vs. Duration', fontsize=14)
    ax4.set_xlabel('Duration (Days)', fontsize=12)
    ax4.set_ylabel('Profit/Loss', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_monte_carlo_simulation(simulation_results: Dict[str, Any],
                               title: str = 'Monte Carlo Simulation Results',
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    몬테카를로 시뮬레이션 결과를 시각화합니다.
    
    Args:
        simulation_results: 시뮬레이션 결과 딕셔너리
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    all_returns = simulation_results.get('all_returns', [])
    all_drawdowns = simulation_results.get('all_drawdowns', [])
    all_sharpe_ratios = simulation_results.get('all_sharpe_ratios', [])
    
    if not all_returns or not all_drawdowns or not all_sharpe_ratios:
        raise ValueError("시뮬레이션 결과 데이터가 부족합니다.")
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # 수익률 분포 히스토그램
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(all_returns, bins=30, kde=True, color='#1f77b4', ax=ax1)
    
    # 신뢰 구간 표시
    returns_ci = simulation_results['returns']
    ax1.axvline(x=returns_ci['mean'], color='red', linestyle='-', 
               label=f"Mean: {returns_ci['mean']:.4f}")
    ax1.axvline(x=returns_ci['ci_lower'], color='green', linestyle='--', 
               label=f"95% CI Lower: {returns_ci['ci_lower']:.4f}")
    ax1.axvline(x=returns_ci['ci_upper'], color='green', linestyle='--', 
               label=f"95% CI Upper: {returns_ci['ci_upper']:.4f}")
    
    ax1.set_title(f'{title} - Return Distribution', fontsize=14)
    ax1.set_xlabel('Total Return', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 최대 손실 분포 히스토그램
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(all_drawdowns, bins=30, kde=True, color='#d62728', ax=ax2)
    
    # 신뢰 구간 표시
    drawdowns_ci = simulation_results['max_drawdowns']
    ax2.axvline(x=drawdowns_ci['mean'], color='red', linestyle='-', 
               label=f"Mean: {drawdowns_ci['mean']:.4f}")
    ax2.axvline(x=drawdowns_ci['ci_lower'], color='green', linestyle='--', 
               label=f"95% CI Lower: {drawdowns_ci['ci_lower']:.4f}")
    ax2.axvline(x=drawdowns_ci['ci_upper'], color='green', linestyle='--', 
               label=f"95% CI Upper: {drawdowns_ci['ci_upper']:.4f}")
    
    ax2.set_title('Maximum Drawdown Distribution', fontsize=14)
    ax2.set_xlabel('Max Drawdown', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Sharpe 비율 분포 히스토그램
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(all_sharpe_ratios, bins=30, kde=True, color='#2ca02c', ax=ax3)
    
    # 신뢰 구간 표시
    sharpe_ci = simulation_results['sharpe_ratios']
    ax3.axvline(x=sharpe_ci['mean'], color='red', linestyle='-', 
               label=f"Mean: {sharpe_ci['mean']:.4f}")
    ax3.axvline(x=sharpe_ci['ci_lower'], color='green', linestyle='--', 
               label=f"95% CI Lower: {sharpe_ci['ci_lower']:.4f}")
    ax3.axvline(x=sharpe_ci['ci_upper'], color='green', linestyle='--', 
               label=f"95% CI Upper: {sharpe_ci['ci_upper']:.4f}")
    
    ax3.set_title('Sharpe Ratio Distribution', fontsize=14)
    ax3.set_xlabel('Sharpe Ratio', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # 성과 지표 요약
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # 통계 데이터 텍스트
    stats_text = (
        f"Total Return:\n"
        f"  Mean: {returns_ci['mean']:.4f}\n"
        f"  Median: {returns_ci['median']:.4f}\n"
        f"  Std Dev: {returns_ci['std']:.4f}\n"
        f"  95% CI: [{returns_ci['ci_lower']:.4f}, {returns_ci['ci_upper']:.4f}]\n\n"
        f"Maximum Drawdown:\n"
        f"  Mean: {drawdowns_ci['mean']:.4f}\n"
        f"  Median: {drawdowns_ci['median']:.4f}\n"
        f"  Std Dev: {drawdowns_ci['std']:.4f}\n"
        f"  95% CI: [{drawdowns_ci['ci_lower']:.4f}, {drawdowns_ci['ci_upper']:.4f}]\n\n"
        f"Sharpe Ratio:\n"
        f"  Mean: {sharpe_ci['mean']:.4f}\n"
        f"  Median: {sharpe_ci['median']:.4f}\n"
        f"  Std Dev: {sharpe_ci['std']:.4f}\n"
        f"  95% CI: [{sharpe_ci['ci_lower']:.4f}, {sharpe_ci['ci_upper']:.4f}]"
    )
    
    ax4.text(0, 1.0, stats_text, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    return fig


def plot_comparison_chart(strategy_results: Dict[str, Dict[str, Any]],
                         benchmark_results: Optional[Dict[str, Any]] = None,
                         benchmark_name: str = 'Benchmark',
                         title: str = 'Strategy Comparison',
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    여러 전략의 성과를 비교합니다.
    
    Args:
        strategy_results: 각 전략별 백테스트 결과의 딕셔너리
        benchmark_results: 벤치마크 결과 (선택 사항)
        benchmark_name: 벤치마크 이름
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    if not strategy_results:
        raise ValueError("전략 결과 데이터가 없습니다.")
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # 누적 수익률 비교 차트
    ax1 = fig.add_subplot(gs[0, :])
    
    # 각 전략의 누적 수익률 계산 및 플롯
    for strategy_name, results in strategy_results.items():
        if 'portfolio_values' in results and len(results['portfolio_values']) > 0:
            values = results['portfolio_values']
            normalized = np.array(values) / values[0]
            dates = results.get('dates', list(range(len(values))))
            ax1.plot(dates, normalized, label=strategy_name, linewidth=2)
    
    # 벤치마크 추가 (있는 경우)
    if benchmark_results and 'portfolio_values' in benchmark_results:
        values = benchmark_results['portfolio_values']
        normalized = np.array(values) / values[0]
        dates = benchmark_results.get('dates', list(range(len(values))))
        ax1.plot(dates, normalized, label=benchmark_name, linewidth=2, linestyle='--', color='black')
    
    ax1.set_title(f'{title} - Cumulative Returns', fontsize=14)
    ax1.set_ylabel('Normalized Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 주요 성과 지표 비교 막대 차트
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    # 지표 계산
    from .evaluation import calculate_comprehensive_metrics
    
    metrics_data = {}
    for strategy_name, results in strategy_results.items():
        metrics_data[strategy_name] = calculate_comprehensive_metrics(results)
    
    if benchmark_results:
        metrics_data[benchmark_name] = calculate_comprehensive_metrics(benchmark_results)
    
    # 지표별 차트
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row, col = (1, 0) if i < 2 else (1, 1)
        ax = fig.add_subplot(gs[row, col])
        
        values = []
        labels = []
        
        for strategy_name, metrics_dict in metrics_data.items():
            if metric in metrics_dict:
                values.append(metrics_dict[metric])
                labels.append(strategy_name)
        
        # 값이 있는 경우만 플롯
        if values:
            # 최대 손실은 음수로 표시
            if metric == 'max_drawdown':
                values = [-v for v in values]
            
            # 색상 설정
            colors = plt.cm.tab10(range(len(values)))
            if benchmark_name in labels:
                idx = labels.index(benchmark_name)
                colors[idx] = [0, 0, 0, 1]  # 검은색으로 벤치마크 표시
            
            # 막대 차트 그리기
            bars = ax.bar(range(len(values)), values, color=colors)
            
            # 축 설정
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(metric_name, fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom',
                        fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_decision_heatmap(features: np.ndarray, 
                         actions: np.ndarray, 
                         feature_names: List[str],
                         title: str = 'Decision Heatmap',
                         figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    특성과 행동 간의 관계를 히트맵으로 시각화합니다.
    
    Args:
        features: 특성 데이터 배열
        actions: 행동 데이터 배열
        feature_names: 특성 이름 목록
        title: 그래프 제목
        figsize: 그림 크기
        
    Returns:
        matplotlib 그림 객체
    """
    if len(features) != len(actions):
        raise ValueError("특성과 행동 데이터의 길이가 일치하지 않습니다.")
    
    if features.shape[1] != len(feature_names):
        raise ValueError("특성 데이터의 열 수와 특성 이름 수가 일치하지 않습니다.")
    
    # 고유 행동 값 추출
    unique_actions = np.unique(actions)
    
    # 그림 생성
    fig, axes = plt.subplots(1, len(unique_actions), figsize=figsize)
    
    if len(unique_actions) == 1:
        axes = [axes]
    
    # 각 행동별 상관관계 계산 및 시각화
    for i, action_value in enumerate(unique_actions):
        action_mask = actions == action_value
        
        if np.sum(action_mask) > 1:
            # 해당 행동의 특성 값만 선택
            action_features = features[action_mask]
            
            # 상관관계 계산
            corr_matrix = np.corrcoef(action_features, rowvar=False)
            
            # 히트맵 시각화
            im = axes[i].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # 축 레이블 설정
            axes[i].set_xticks(np.arange(len(feature_names)))
            axes[i].set_yticks(np.arange(len(feature_names)))
            axes[i].set_xticklabels(feature_names, rotation=90)
            axes[i].set_yticklabels(feature_names)
            
            # 제목 설정
            action_labels = ['Hold', 'Buy', 'Sell']
            action_label = action_labels[action_value] if action_value < len(action_labels) else f'Action {action_value}'
            axes[i].set_title(f'{action_label} (n={np.sum(action_mask)})', fontsize=14)
            
            # 상관 계수 값 표시
            for j in range(len(feature_names)):
                for k in range(len(feature_names)):
                    if abs(corr_matrix[j, k]) > 0.3:  # 중요한 상관 관계만 표시
                        text_color = 'white' if abs(corr_matrix[j, k]) > 0.5 else 'black'
                        axes[i].text(k, j, f'{corr_matrix[j, k]:.2f}',
                                  ha='center', va='center', color=text_color, fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f'Not enough data for Action {action_value}',
                      ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # 컬러바 추가
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.01)
    cbar.set_label('Correlation', fontsize=12)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def create_performance_dashboard(backtest_results: Dict[str, Any],
                               benchmark_results: Optional[Dict[str, Any]] = None,
                               benchmark_name: str = 'Benchmark',
                               title: str = 'Trading Strategy Performance Dashboard',
                               save_path: Optional[str] = None) -> None:
    """
    트레이딩 전략 성과 대시보드를 생성합니다.
    
    Args:
        backtest_results: 백테스트 결과 딕셔너리
        benchmark_results: 벤치마크 결과 (선택 사항)
        benchmark_name: 벤치마크 이름
        title: 대시보드 제목
        save_path: 저장 경로 (선택 사항)
    """
    # 기본 성과 지표 계산
    from .evaluation import calculate_comprehensive_metrics
    
    metrics = calculate_comprehensive_metrics(backtest_results)
    
    # 포트폴리오 성과 차트
    fig1 = plot_portfolio_performance(backtest_results, 
                                     benchmark_values=benchmark_results['portfolio_values'] if benchmark_results else None,
                                     benchmark_name=benchmark_name,
                                     title=title)
    
    # 수익률 분포 차트
    fig2 = plot_return_distribution(backtest_results,
                                   benchmark_values=benchmark_results['portfolio_values'] if benchmark_results else None,
                                   benchmark_name=benchmark_name)
    
    # 롤링 지표 차트
    fig3 = plot_rolling_metrics(backtest_results, window=60)
    
    # 거래 분석 차트
    if 'trades' in backtest_results and backtest_results['trades']:
        fig4 = plot_trade_analysis(backtest_results)
    
    # 저장 경로가 제공된 경우 차트 저장
    if save_path:
        fig1.savefig(f'{save_path}_portfolio.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{save_path}_returns.png', dpi=300, bbox_inches='tight')
        fig3.savefig(f'{save_path}_metrics.png', dpi=300, bbox_inches='tight')
        
        if 'trades' in backtest_results and backtest_results['trades']:
            fig4.savefig(f'{save_path}_trades.png', dpi=300, bbox_inches='tight')
    
    # 주요 지표 출력
    print(f"==== {title} - 성과 요약 ====")
    print(f"총 수익률: {metrics['total_return']:.4f} ({metrics['annualized_return']:.4f} 연간화)")
    print(f"변동성 (연간화): {metrics['volatility']:.4f}")
    print(f"Sharpe 비율: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino 비율: {metrics['sortino_ratio']:.4f}")
    print(f"최대 손실: {metrics['max_drawdown']:.4f}")
    print(f"최대 손실 기간: {metrics['max_drawdown_duration']} 일")
    print(f"Calmar 비율: {metrics['calmar_ratio']:.4f}")
    
    if 'trade_count' in metrics:
        print(f"거래 횟수: {metrics['trade_count']}")
        print(f"승률: {metrics['win_rate']:.4f}")
        print(f"수익-손실 비율: {metrics['profit_loss_ratio']:.4f}")
        print(f"기대 수익: {metrics['expectancy']:.4f}")
    
    print("=========================")
    
    # 차트 표시
    plt.show()
