#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
금융 시각화 유틸리티 모듈

이 모듈은 금융 트레이딩 전략 분석 및 백테스팅을 위한 다양한 시각화 도구를 제공합니다.
포트폴리오 성과, 수익률 분포, 위험 지표, 거래 패턴, 시장 레짐, 벤치마크 비교 등의
시각화 함수를 포함합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from typing import List, Dict, Tuple, Union, Optional, Any


def set_style():
    """시각화 스타일 설정"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def plot_portfolio_value(
    portfolio_values: np.ndarray,
    benchmark_values: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    title: str = 'Portfolio Value Over Time',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    포트폴리오 가치 시각화

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        benchmark_values: 벤치마크 가치 시계열 (옵션)
        dates: 날짜 배열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    if dates is not None:
        x = dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(portfolio_values))
    
    # 포트폴리오 가치 플롯
    plt.plot(x, portfolio_values, label='Portfolio', linewidth=2)
    
    # 벤치마크 가치 플롯 (있는 경우)
    if benchmark_values is not None:
        # 길이 조정
        min_length = min(len(portfolio_values), len(benchmark_values))
        b_values = benchmark_values[:min_length]
        
        # 벤치마크 스케일링 (초기 값을 포트폴리오 초기값과 맞춤)
        scaled_benchmark = b_values * (portfolio_values[0] / b_values[0])
        
        plt.plot(x[:min_length], scaled_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_returns(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    title: str = 'Returns Over Time',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    수익률 시각화

    Args:
        returns: 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열 (옵션)
        dates: 날짜 배열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    if dates is not None:
        x = dates[1:]  # 수익률은 첫 번째 데이터 포인트가 없음
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(returns))
    
    # 수익률 플롯
    plt.plot(x, returns, label='Strategy Returns', alpha=0.7)
    
    # 벤치마크 수익률 플롯 (있는 경우)
    if benchmark_returns is not None:
        # 길이 조정
        min_length = min(len(returns), len(benchmark_returns))
        plt.plot(x[:min_length], benchmark_returns[:min_length], label='Benchmark Returns', alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # y축을 백분율로 표시
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_cumulative_returns(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    title: str = 'Cumulative Returns',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    누적 수익률 시각화

    Args:
        returns: 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열 (옵션)
        dates: 날짜 배열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    # 누적 수익률 계산
    cum_returns = np.cumprod(1 + returns) - 1
    
    if dates is not None:
        x = dates[1:]  # 수익률은 첫 번째 데이터 포인트가 없음
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(cum_returns))
    
    # 누적 수익률 플롯
    plt.plot(x, cum_returns, label='Strategy', linewidth=2)
    
    # 벤치마크 누적 수익률 플롯 (있는 경우)
    if benchmark_returns is not None:
        # 길이 조정
        min_length = min(len(returns), len(benchmark_returns))
        b_returns = benchmark_returns[:min_length]
        
        # 누적 벤치마크 수익률 계산
        cum_benchmark = np.cumprod(1 + b_returns) - 1
        
        plt.plot(x[:min_length], cum_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # y축을 백분율로 표시
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_return_distribution(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    title: str = 'Return Distribution',
    figsize: Tuple[int, int] = (14, 7),
    bins: int = 50,
    save_path: Optional[str] = None
):
    """
    수익률 분포 시각화

    Args:
        returns: 수익률 시계열
        benchmark_returns: 벤치마크 수익률 시계열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        bins: 히스토그램 빈 수
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    # 전략 수익률 히스토그램
    sns.histplot(returns, kde=True, bins=bins, alpha=0.6, label='Strategy')
    
    # 벤치마크 수익률 히스토그램 (있는 경우)
    if benchmark_returns is not None:
        sns.histplot(benchmark_returns, kde=True, bins=bins, alpha=0.4, label='Benchmark')
    
    plt.axvline(x=0, color='red', linestyle='--')
    plt.axvline(x=np.mean(returns), color='blue', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
    
    if benchmark_returns is not None:
        plt.axvline(x=np.mean(benchmark_returns), color='green', linestyle='--', 
                    label=f'Benchmark Mean: {np.mean(benchmark_returns):.2%}')
    
    plt.title(title)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # x축을 백분율로 표시
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_drawdown(
    portfolio_values: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = 'Portfolio Drawdown',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    포트폴리오 손실(drawdown) 시각화

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        dates: 날짜 배열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    # 누적 최댓값 계산
    peak = np.maximum.accumulate(portfolio_values)
    
    # 손실률 계산
    drawdown = (peak - portfolio_values) / peak
    
    if dates is not None:
        x = dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(drawdown))
    
    # 손실 플롯
    plt.fill_between(x, 0, drawdown, alpha=0.3, color='red')
    plt.plot(x, drawdown, color='red', label='Drawdown')
    
    # 최대 손실 표시
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    if dates is not None and max_dd_idx < len(dates):
        max_date = dates[max_dd_idx]
        plt.scatter(max_date, max_dd, color='darkred', s=100, zorder=5)
        plt.annotate(f'Max Drawdown: {max_dd:.2%}', 
                     xy=(max_date, max_dd), 
                     xytext=(30, 20), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    else:
        plt.scatter(max_dd_idx, max_dd, color='darkred', s=100, zorder=5)
        plt.annotate(f'Max Drawdown: {max_dd:.2%}', 
                     xy=(max_dd_idx, max_dd), 
                     xytext=(30, 20), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # y축을 백분율로 표시
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_rolling_metrics(
    returns: np.ndarray,
    window: int = 252,
    metrics: List[str] = ['sharpe', 'volatility', 'return'],
    dates: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    롤링 성과 지표 시각화

    Args:
        returns: 수익률 시계열
        window: 롤링 윈도우 크기
        metrics: 계산할 지표 목록
        dates: 날짜 배열 (옵션)
        risk_free_rate: 무위험 이자율
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    # 유효 지표 목록
    valid_metrics = ['sharpe', 'volatility', 'return', 'sortino', 'calmar', 'information']
    
    # 지표 필터링
    metrics = [m for m in metrics if m in valid_metrics]
    
    if not metrics:
        print("유효한 지표가 없습니다.")
        return
    
    # 롤링 지표 계산
    rolling_metrics = {}
    
    for metric in metrics:
        rolling_metrics[metric] = np.full(len(returns), np.nan)
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        
        if 'return' in metrics:
            rolling_metrics['return'][i - 1] = np.mean(window_returns) * 252  # 연율화
        
        if 'volatility' in metrics:
            rolling_metrics['volatility'][i - 1] = np.std(window_returns) * np.sqrt(252)  # 연율화
        
        if 'sharpe' in metrics:
            mean_return = np.mean(window_returns) * 252
            volatility = np.std(window_returns) * np.sqrt(252)
            if volatility > 0:
                rolling_metrics['sharpe'][i - 1] = (mean_return - risk_free_rate) / volatility
        
        if 'sortino' in metrics:
            mean_return = np.mean(window_returns) * 252
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(252)
                if downside_deviation > 0:
                    rolling_metrics['sortino'][i - 1] = (mean_return - risk_free_rate) / downside_deviation
        
        if 'calmar' in metrics:
            # 윈도우 내 누적 가치 계산
            window_values = np.cumprod(1 + window_returns)
            peak = np.maximum.accumulate(window_values)
            drawdown = (peak - window_values) / peak
            max_dd = np.max(drawdown)
            
            mean_return = np.mean(window_returns) * 252
            if max_dd > 0:
                rolling_metrics['calmar'][i - 1] = (mean_return - risk_free_rate) / max_dd
    
    # 그래프 생성
    n_metrics = len(metrics)
    fig = plt.figure(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(n_metrics, 1, i + 1)
        
        if dates is not None:
            x = dates[window:]
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            x = np.arange(window, len(returns) + 1)
        
        ax.plot(x, rolling_metrics[metric][window-1:], label=f'{metric.capitalize()}')
        
        # 특정 지표에 대한 기준선 추가
        if metric == 'sharpe':
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good')
            ax.axhline(y=2, color='blue', linestyle='--', alpha=0.5, label='Very Good')
            ax.axhline(y=3, color='purple', linestyle='--', alpha=0.5, label='Excellent')
        elif metric == 'return':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f'Rolling {metric.capitalize()} ({window} days)')
        
        # y축 포맷 조정
        if metric in ['return', 'volatility']:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        ax.grid(True)
        ax.legend()
        
        # x축 레이블은 마지막 서브플롯에만 표시
        if i == n_metrics - 1:
            ax.set_xlabel('Date' if dates is not None else 'Time')
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_positions(
    positions: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = 'Positions Over Time',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
):
    """
    포지션 시각화

    Args:
        positions: 포지션 시계열 (0=보유 없음, 1=롱, -1=숏)
        dates: 날짜 배열 (옵션)
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    plt.figure(figsize=figsize)
    
    if dates is not None:
        x = dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(positions))
    
    # 롱 포지션 배경 표시
    long_mask = positions > 0
    plt.fill_between(x, 0, 1, where=long_mask, alpha=0.2, color='green', label='Long')
    
    # 숏 포지션 배경 표시
    short_mask = positions < 0
    plt.fill_between(x, 0, -1, where=short_mask, alpha=0.2, color='red', label='Short')
    
    # 포지션 플롯
    plt.plot(x, positions, color='blue', label='Position')
    
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time')
    plt.ylabel('Position')
    plt.yticks([-1, 0, 1], ['Short', 'No Position', 'Long'])
    plt.legend()
    plt.grid(True)
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_trade_analysis(
    trades: List[Dict],
    title: str = 'Trade Analysis',
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
):
    """
    거래 분석 시각화

    Args:
        trades: 거래 기록 리스트
        title: 그래프 제목
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    if not trades:
        print("거래 기록이 없습니다.")
        return
    
    # 거래 타입 분석
    trade_types = {}
    profits = []
    durations = []
    
    for trade in trades:
        trade_type = trade[0]
        trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
        
        # 수익을 계산할 수 있는 경우 (폐색 거래)
        if trade_type in ['close_long', 'close_short'] and len(trade) > 3:
            profits.append(trade[3])  # 수익
        
        # 거래 기간을 계산할 수 있는 경우
        if len(trade) > 4:
            durations.append(trade[4])  # 거래 기간
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 거래 타입 분포
    ax1 = fig.add_subplot(2, 2, 1)
    types = list(trade_types.keys())
    counts = list(trade_types.values())
    
    ax1.bar(types, counts)
    ax1.set_title('Trade Type Distribution')
    ax1.set_xlabel('Trade Type')
    ax1.set_ylabel('Count')
    
    # 2. 수익 분포
    if profits:
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(profits, bins=20, alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.axvline(x=np.mean(profits), color='green', linestyle='--', 
                    label=f'Mean: {np.mean(profits):.2f}')
        ax2.set_title('Profit Distribution')
        ax2.set_xlabel('Profit')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # 3. 거래 기간 분포
    if durations:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(durations, bins=20, alpha=0.7)
        ax3.axvline(x=np.mean(durations), color='green', linestyle='--',
                    label=f'Mean: {np.mean(durations):.2f} days')
        ax3.set_title('Trade Duration Distribution')
        ax3.set_xlabel('Duration (days)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # 4. 수익 vs 기간 산점도
    if profits and durations and len(profits) == len(durations):
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(durations, profits, alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--')
        
        # 추세선 추가
        if len(profits) > 1:
            z = np.polyfit(durations, profits, 1)
            p = np.poly1d(z)
            ax4.plot(sorted(durations), p(sorted(durations)), "r--", 
                     label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
        
        ax4.set_title('Profit vs Duration')
        ax4.set_xlabel('Duration (days)')
        ax4.set_ylabel('Profit')
        ax4.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_market_regimes(
    portfolio_values: np.ndarray,
    regime_labels: np.ndarray,
    dates: Optional[np.ndarray] = None,
    benchmark_values: Optional[np.ndarray] = None,
    regime_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    시장 레짐과 포트폴리오 성과 시각화

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        regime_labels: 시장 레짐 레이블
        dates: 날짜 배열 (옵션)
        benchmark_values: 벤치마크 가치 시계열 (옵션)
        regime_names: 레짐 번호와 이름 매핑 딕셔너리 (옵션)
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    # 레짐 색상
    colors = ['lightblue', 'lightgreen', 'lightsalmon', 'plum', 'lightgray', 'wheat', 'lightcyan', 'lightpink']
    
    # 유니크 레짐
    unique_regimes = np.unique(regime_labels)
    unique_regimes = [r for r in unique_regimes if not np.isnan(r)]
    
    # 레짐 이름 (없으면 자동 생성)
    if regime_names is None:
        regime_names = {int(r): f'Regime {int(r)}' for r in unique_regimes}
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # 날짜 또는 시간 축 설정
    if dates is not None:
        x = dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(portfolio_values))
    
    # 포트폴리오 및 벤치마크 플롯
    ax1.plot(x, portfolio_values, label='Portfolio', linewidth=2)
    
    if benchmark_values is not None:
        # 길이 조정
        min_length = min(len(portfolio_values), len(benchmark_values))
        b_values = benchmark_values[:min_length]
        
        # 벤치마크 스케일링
        scaled_benchmark = b_values * (portfolio_values[0] / b_values[0])
        
        ax1.plot(x[:min_length], scaled_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    
    # 레짐 배경색 추가
    for r in unique_regimes:
        mask = regime_labels == r
        r_int = int(r)
        color = colors[r_int % len(colors)]
        
        for i in range(len(mask) - 1):
            if mask[i]:
                if dates is not None:
                    start = dates[i]
                    if i + 1 < len(dates):
                        end = dates[i + 1]
                    else:
                        end = dates[i]
                else:
                    start = i
                    end = i + 1
                
                ax1.axvspan(start, end, alpha=0.3, color=color)
                ax2.axvspan(start, end, alpha=0.3, color=color)
    
    # 레짐 레이블 플롯
    ax2.plot(x, regime_labels, 'o-', markersize=3)
    
    # 축 설정
    ax1.set_title('Portfolio Value with Market Regimes')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Market Regimes')
    ax2.set_xlabel('Date' if dates is not None else 'Time')
    ax2.set_ylabel('Regime')
    ax2.set_yticks([int(r) for r in unique_regimes])
    ax2.set_yticklabels([regime_names.get(int(r), f'Regime {int(r)}') for r in unique_regimes])
    ax2.grid(True)
    
    # 레짐 범례 추가
    legend_elements = [Patch(facecolor=colors[int(r) % len(colors)], alpha=0.3, 
                             label=regime_names.get(int(r), f'Regime {int(r)}'))
                       for r in unique_regimes]
    ax2.legend(handles=legend_elements, loc='best')
    
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_regime_performance(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    regime_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    레짐별 성과 비교 시각화

    Args:
        returns: 수익률 시계열
        regime_labels: 시장 레짐 레이블
        benchmark_returns: 벤치마크 수익률 시계열 (옵션)
        regime_names: 레짐 번호와 이름 매핑 딕셔너리 (옵션)
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    # 유니크 레짐
    unique_regimes = np.unique(regime_labels)
    unique_regimes = [r for r in unique_regimes if not np.isnan(r)]
    
    # 레짐 이름 (없으면 자동 생성)
    if regime_names is None:
        regime_names = {int(r): f'Regime {int(r)}' for r in unique_regimes}
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    
    # 1. 레짐별 평균 수익률
    ax1 = fig.add_subplot(2, 2, 1)
    
    regime_returns = []
    regime_names_list = []
    
    for r in unique_regimes:
        mask = regime_labels == r
        r_returns = returns[mask]
        
        if len(r_returns) > 0:
            regime_returns.append(np.mean(r_returns))
            regime_names_list.append(regime_names.get(int(r), f'Regime {int(r)}'))
    
    # 벤치마크가 있는 경우 평균 수익률도 계산
    if benchmark_returns is not None:
        benchmark_regime_returns = []
        
        for r in unique_regimes:
            mask = regime_labels == r
            mask = mask[:len(benchmark_returns)]  # 길이 조정
            b_returns = benchmark_returns[mask]
            
            if len(b_returns) > 0:
                benchmark_regime_returns.append(np.mean(b_returns))
    
    # 수익률 플롯
    x = np.arange(len(regime_names_list))
    width = 0.35
    
    ax1.bar(x - width/2 if benchmark_returns is not None else x, regime_returns, width, label='Strategy')
    
    if benchmark_returns is not None and len(benchmark_regime_returns) == len(regime_returns):
        ax1.bar(x + width/2, benchmark_regime_returns, width, label='Benchmark')
    
    ax1.set_title('Average Return by Regime')
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Average Return')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_names_list)
    ax1.legend()
    ax1.grid(True)
    
    # y축을 백분율로 표시
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 2. 레짐별 변동성
    ax2 = fig.add_subplot(2, 2, 2)
    
    regime_vols = []
    
    for r in unique_regimes:
        mask = regime_labels == r
        r_returns = returns[mask]
        
        if len(r_returns) > 0:
            regime_vols.append(np.std(r_returns))
    
    # 벤치마크가 있는 경우 변동성도 계산
    if benchmark_returns is not None:
        benchmark_regime_vols = []
        
        for r in unique_regimes:
            mask = regime_labels == r
            mask = mask[:len(benchmark_returns)]  # 길이 조정
            b_returns = benchmark_returns[mask]
            
            if len(b_returns) > 0:
                benchmark_regime_vols.append(np.std(b_returns))
    
    # 변동성 플롯
    ax2.bar(x - width/2 if benchmark_returns is not None else x, regime_vols, width, label='Strategy')
    
    if benchmark_returns is not None and len(benchmark_regime_vols) == len(regime_vols):
        ax2.bar(x + width/2, benchmark_regime_vols, width, label='Benchmark')
    
    ax2.set_title('Volatility by Regime')
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Volatility')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_names_list)
    ax2.legend()
    ax2.grid(True)
    
    # y축을 백분율로 표시
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 3. 레짐별 Sharpe 비율
    ax3 = fig.add_subplot(2, 2, 3)
    
    regime_sharpes = []
    
    for r in unique_regimes:
        mask = regime_labels == r
        r_returns = returns[mask]
        
        if len(r_returns) > 0:
            mean_return = np.mean(r_returns)
            std_return = np.std(r_returns)
            
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)  # 연율화
                regime_sharpes.append(sharpe)
            else:
                regime_sharpes.append(0)
    
    # 벤치마크가 있는 경우 Sharpe 비율도 계산
    if benchmark_returns is not None:
        benchmark_regime_sharpes = []
        
        for r in unique_regimes:
            mask = regime_labels == r
            mask = mask[:len(benchmark_returns)]  # 길이 조정
            b_returns = benchmark_returns[mask]
            
            if len(b_returns) > 0:
                mean_return = np.mean(b_returns)
                std_return = np.std(b_returns)
                
                if std_return > 0:
                    sharpe = mean_return / std_return * np.sqrt(252)  # 연율화
                    benchmark_regime_sharpes.append(sharpe)
                else:
                    benchmark_regime_sharpes.append(0)
    
    # Sharpe 비율 플롯
    ax3.bar(x - width/2 if benchmark_returns is not None else x, regime_sharpes, width, label='Strategy')
    
    if benchmark_returns is not None and len(benchmark_regime_sharpes) == len(regime_sharpes):
        ax3.bar(x + width/2, benchmark_regime_sharpes, width, label='Benchmark')
    
    ax3.set_title('Sharpe Ratio by Regime')
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regime_names_list)
    ax3.legend()
    ax3.grid(True)
    
    # 4. 레짐별 승률
    ax4 = fig.add_subplot(2, 2, 4)
    
    regime_win_rates = []
    
    for r in unique_regimes:
        mask = regime_labels == r
        r_returns = returns[mask]
        
        if len(r_returns) > 0:
            wins = np.sum(r_returns > 0)
            win_rate = wins / len(r_returns)
            regime_win_rates.append(win_rate)
        else:
            regime_win_rates.append(0)
    
    # 벤치마크가 있는 경우 승률도 계산
    if benchmark_returns is not None:
        benchmark_regime_win_rates = []
        
        for r in unique_regimes:
            mask = regime_labels == r
            mask = mask[:len(benchmark_returns)]  # 길이 조정
            b_returns = benchmark_returns[mask]
            
            if len(b_returns) > 0:
                wins = np.sum(b_returns > 0)
                win_rate = wins / len(b_returns)
                benchmark_regime_win_rates.append(win_rate)
            else:
                benchmark_regime_win_rates.append(0)
    
    # 승률 플롯
    ax4.bar(x - width/2 if benchmark_returns is not None else x, regime_win_rates, width, label='Strategy')
    
    if benchmark_returns is not None and len(benchmark_regime_win_rates) == len(regime_win_rates):
        ax4.bar(x + width/2, benchmark_regime_win_rates, width, label='Benchmark')
    
    ax4.set_title('Win Rate by Regime')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Win Rate')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regime_names_list)
    ax4.legend()
    ax4.grid(True)
    
    # y축을 백분율로 표시
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_monte_carlo_results(
    mc_results: List[Dict],
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
):
    """
    몬테카를로 시뮬레이션 결과 시각화

    Args:
        mc_results: 몬테카를로 시뮬레이션 결과 리스트
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    if not mc_results:
        print("시뮬레이션 결과가 없습니다.")
        return
    
    # 모든 시뮬레이션의 포트폴리오 가치 추출
    all_portfolio_values = [result['portfolio_values'] for result in mc_results]
    
    # 시뮬레이션 길이 표준화 (가장 짧은 시뮬레이션에 맞춤)
    min_length = min(len(values) for values in all_portfolio_values)
    all_portfolio_values = [values[:min_length] for values in all_portfolio_values]
    
    # 배열로 변환
    all_portfolio_values = np.array(all_portfolio_values)
    
    # 평균 및 백분위수 계산
    mean_values = np.mean(all_portfolio_values, axis=0)
    median_values = np.median(all_portfolio_values, axis=0)
    percentile_5 = np.percentile(all_portfolio_values, 5, axis=0)
    percentile_25 = np.percentile(all_portfolio_values, 25, axis=0)
    percentile_75 = np.percentile(all_portfolio_values, 75, axis=0)
    percentile_95 = np.percentile(all_portfolio_values, 95, axis=0)
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    
    # 1. 포트폴리오 가치 시뮬레이션
    ax1 = fig.add_subplot(2, 2, 1)
    
    # 개별 시뮬레이션 (투명도를 낮게 설정)
    for i in range(min(30, len(all_portfolio_values))):  # 가시성을 위해 최대 30개만 표시
        ax1.plot(all_portfolio_values[i], alpha=0.1, color='gray')
    
    # 평균 및 백분위수
    ax1.plot(mean_values, label='Mean', color='blue', linewidth=2)
    ax1.plot(median_values, label='Median', color='green', linewidth=2)
    ax1.fill_between(range(min_length), percentile_5, percentile_95, alpha=0.2, color='blue', label='5-95 Percentile')
    ax1.fill_between(range(min_length), percentile_25, percentile_75, alpha=0.3, color='blue', label='25-75 Percentile')
    
    ax1.set_title('Monte Carlo Simulation: Portfolio Value')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 최종 포트폴리오 가치 분포
    ax2 = fig.add_subplot(2, 2, 2)
    
    final_values = all_portfolio_values[:, -1]
    sns.histplot(final_values, kde=True, ax=ax2)
    
    ax2.axvline(x=mean_values[-1], color='r', linestyle='--', label=f'Mean: ${mean_values[-1]:.2f}')
    ax2.axvline(x=median_values[-1], color='g', linestyle='--', label=f'Median: ${median_values[-1]:.2f}')
    
    ax2.set_title('Distribution of Final Portfolio Values')
    ax2.set_xlabel('Portfolio Value ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 수익률 통계
    ax3 = fig.add_subplot(2, 2, 3)
    
    returns = np.diff(all_portfolio_values, axis=1) / all_portfolio_values[:, :-1]
    mean_returns = np.mean(returns, axis=1)
    
    sns.histplot(mean_returns, kde=True, ax=ax3)
    ax3.axvline(x=np.mean(mean_returns), color='r', linestyle='--', label=f'Mean: {np.mean(mean_returns):.4f}')
    
    ax3.set_title('Distribution of Average Daily Returns')
    ax3.set_xlabel('Average Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True)
    
    # x축을 백분율로 표시
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    
    # 4. 최대 손실 분포
    ax4 = fig.add_subplot(2, 2, 4)
    
    max_drawdowns = []
    for values in all_portfolio_values:
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)
        max_drawdowns.append(max_drawdown)
    
    sns.histplot(max_drawdowns, kde=True, ax=ax4)
    ax4.axvline(x=np.mean(max_drawdowns), color='r', linestyle='--', label=f'Mean: {np.mean(max_drawdowns):.4f}')
    
    ax4.set_title('Distribution of Maximum Drawdowns')
    ax4.set_xlabel('Maximum Drawdown')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True)
    
    # x축을 백분율로 표시
    ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stress_test_results(
    stress_results: List[Dict],
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
):
    """
    스트레스 테스트 결과 시각화

    Args:
        stress_results: 스트레스 테스트 결과 리스트
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    if not stress_results:
        print("스트레스 테스트 결과가 없습니다.")
        return
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    
    # 1. 시나리오별 포트폴리오 가치
    ax1 = fig.add_subplot(2, 2, 1)
    
    for result in stress_results:
        label = f"Scenario {result['scenario']} (Vol Factor: {result['vol_factor']:.2f})"
        ax1.plot(result['portfolio_values'], label=label)
    
    ax1.set_title('Stress Test: Portfolio Value Under Different Scenarios')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # 스트레스 테스트 요약 데이터 생성
    scenario_data = []
    
    for result in stress_results:
        portfolio_values = result['portfolio_values']
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # 누적 최댓값 계산
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_dd = np.max(drawdown)
        
        scenario_data.append({
            'scenario': result['scenario'],
            'vol_factor': result['vol_factor'],
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': max_dd,
            'volatility': np.std(returns) * np.sqrt(252)
        })
    
    # 변동성 계수에 따른 성과 지표
    vol_factors = [data['vol_factor'] for data in scenario_data]
    
    # 2. 총 수익률 vs 변동성 계수
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(vol_factors, [data['total_return'] for data in scenario_data], 'o-')
    ax2.set_title('Total Return vs Volatility Factor')
    ax2.set_xlabel('Volatility Factor')
    ax2.set_ylabel('Total Return')
    ax2.grid(True)
    
    # y축을 백분율로 표시
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # 3. Sharpe 비율 vs 변동성 계수
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(vol_factors, [data['sharpe_ratio'] for data in scenario_data], 'o-')
    ax3.set_title('Sharpe Ratio vs Volatility Factor')
    ax3.set_xlabel('Volatility Factor')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.grid(True)
    
    # 4. 최대 손실 vs 변동성 계수
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(vol_factors, [data['max_drawdown'] for data in scenario_data], 'o-')
    ax4.set_title('Maximum Drawdown vs Volatility Factor')
    ax4.set_xlabel('Volatility Factor')
    ax4.set_ylabel('Maximum Drawdown')
    ax4.grid(True)
    
    # y축을 백분율로 표시
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_dashboard(
    portfolio_values: np.ndarray,
    returns: np.ndarray,
    metrics: Dict[str, float],
    benchmark_values: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 16),
    save_path: Optional[str] = None
):
    """
    성과 대시보드 시각화

    Args:
        portfolio_values: 포트폴리오 가치 시계열
        returns: 수익률 시계열
        metrics: 성과 지표 딕셔너리
        benchmark_values: 벤치마크 가치 시계열 (옵션)
        benchmark_returns: 벤치마크 수익률 시계열 (옵션)
        dates: 날짜 배열 (옵션)
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    # 누적 수익률 계산
    cum_returns = np.cumprod(1 + returns) - 1
    
    if benchmark_returns is not None:
        # 길이 조정
        min_length = min(len(returns), len(benchmark_returns))
        b_returns = benchmark_returns[:min_length]
        cum_benchmark = np.cumprod(1 + b_returns) - 1
    
    # 손실 계산
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    
    # 롤링 지표 계산
    window = min(252, len(returns) // 4)  # 최소 4개의 윈도우
    
    if window > 10:
        rolling_sharpe = np.full(len(returns), np.nan)
        rolling_volatility = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            rolling_sharpe[i - 1] = metrics.get('sharpe_ratio', 0) if 'sharpe_ratio' in metrics else 0
            rolling_volatility[i - 1] = np.std(window_returns) * np.sqrt(252)
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Performance Dashboard', fontsize=16)
    
    # 1. 포트폴리오 가치
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    
    if dates is not None:
        x = dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        x = np.arange(len(portfolio_values))
    
    ax1.plot(x, portfolio_values, label='Portfolio', linewidth=2)
    
    if benchmark_values is not None:
        # 벤치마크 스케일링
        b_values = benchmark_values[:min(len(portfolio_values), len(benchmark_values))]
        scaled_benchmark = b_values * (portfolio_values[0] / b_values[0])
        
        ax1.plot(x[:len(scaled_benchmark)], scaled_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio Value')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 누적 수익률
    ax2 = plt.subplot2grid((4, 2), (1, 0))
    
    if dates is not None:
        x_returns = dates[1:]  # 수익률은 첫 번째 데이터 포인트가 없음
    else:
        x_returns = np.arange(len(cum_returns))
    
    ax2.plot(x_returns, cum_returns, label='Strategy', linewidth=2)
    
    if benchmark_returns is not None:
        ax2.plot(x_returns[:len(cum_benchmark)], cum_benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Return')
    ax2.legend()
    ax2.grid(True)
    
    # y축을 백분율로 표시
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # 3. 손실
    ax3 = plt.subplot2grid((4, 2), (1, 1))
    
    ax3.fill_between(x, 0, drawdown, alpha=0.3, color='red')
    ax3.plot(x, drawdown, color='red', label='Drawdown')
    
    # 최대 손실 표시
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    ax3.scatter(x[max_dd_idx] if max_dd_idx < len(x) else x[-1], max_dd, color='darkred', s=100, zorder=5)
    ax3.annotate(f'Max: {max_dd:.1%}', 
                 xy=(x[max_dd_idx] if max_dd_idx < len(x) else x[-1], max_dd), 
                 xytext=(30, 20), 
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax3.set_title('Drawdown')
    ax3.set_ylabel('Drawdown')
    ax3.grid(True)
    
    # y축을 백분율로 표시
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # 4. 월별 수익률 히트맵 (날짜가 있는 경우)
    if dates is not None and isinstance(dates[0], pd.Timestamp):
        ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
        
        # 일별 수익률을 월별로 재구성
        df_returns = pd.Series(returns, index=dates[1:])
        monthly_returns = df_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 월별 수익률을 연도와 월로 피벗
        returns_pivot = monthly_returns.unstack(level=0) if hasattr(monthly_returns, 'unstack') else None
        
        if returns_pivot is not None:
            # 히트맵 생성
            sns.heatmap(returns_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('Monthly Returns (%)')
            ax4.set_ylabel('Month')
            ax4.set_xlabel('Year')
        else:
            # 충분한 데이터가 없는 경우 메시지 표시
            ax4.text(0.5, 0.5, 'Insufficient data for monthly returns heatmap', 
                    horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns')
            ax4.axis('off')
    else:
        # 날짜가 없는 경우 롤링 지표
        ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
        
        if window > 10:
            ax4.plot(x_returns[window-1:], rolling_sharpe[window-1:], label='Rolling Sharpe')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title(f'Rolling Sharpe Ratio ({window} days)')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.legend()
            ax4.grid(True)
        else:
            # 충분한 데이터가 없는 경우 메시지 표시
            ax4.text(0.5, 0.5, 'Insufficient data for rolling metrics', 
                    horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Rolling Metrics')
            ax4.axis('off')
    
    # 5. 성과 지표 테이블
    ax5 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    
    # 주요 지표 선택
    key_metrics = [
        ('Total Return', metrics.get('total_return', 0)),
        ('Annualized Return', metrics.get('annualized_return', 0)),
        ('Annualized Volatility', metrics.get('annualized_volatility', 0)),
        ('Sharpe Ratio', metrics.get('sharpe_ratio', 0)),
        ('Sortino Ratio', metrics.get('sortino_ratio', 0)),
        ('Maximum Drawdown', metrics.get('max_drawdown', 0)),
        ('Win Rate', metrics.get('win_rate', 0)),
        ('Profit/Loss Ratio', metrics.get('profit_loss_ratio', 0))
    ]
    
    # 테이블 데이터 형식 지정
    cell_text = []
    for name, value in key_metrics:
        if name in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'Win Rate']:
            formatted_value = f"{value:.2%}"
        elif name in ['Sharpe Ratio', 'Sortino Ratio']:
            formatted_value = f"{value:.2f}"
        elif name == 'Profit/Loss Ratio':
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        cell_text.append([name, formatted_value])
    
    # 테이블 생성
    ax5.axis('tight')
    ax5.axis('off')
    table = ax5.table(cellText=cell_text, colLabels=['Metric', 'Value'], 
                      cellLoc='center', loc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 헤더 행
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        elif j == 0:  # 지표 열
            cell.set_text_props(ha='left')
            cell.PAD = 0.3
    
    ax5.set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if dates is not None:
        fig.autofmt_xdate()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_factor_exposure(
    returns: np.ndarray,
    factor_returns: Dict[str, np.ndarray],
    dates: Optional[np.ndarray] = None,
    window: int = 252,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    팩터 노출도 시각화

    Args:
        returns: 전략 수익률 시계열
        factor_returns: 팩터 수익률 시계열 딕셔너리
        dates: 날짜 배열 (옵션)
        window: 롤링 회귀 윈도우 크기
        figsize: 그림 크기
        save_path: 저장 경로 (옵션)
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels 패키지가 설치되어 있지 않습니다.")
        return
    
    # 팩터 개수
    n_factors = len(factor_returns)
    
    if n_factors == 0:
        print("팩터 데이터가 없습니다.")
        return
    
    # 데이터 길이 확인
    factor_lengths = [len(factor_rets) for factor_rets in factor_returns.values()]
    if min(factor_lengths) < window:
        print(f"일부 팩터의 데이터가 충분하지 않습니다. 최소 윈도우 크기: {window}")
        return
    
    # 롤링 회귀 계수 계산
    coefficients = {}
    t_stats = {}
    r_squared = np.zeros(len(returns) - window + 1)
    
    for factor_name in factor_returns.keys():
        coefficients[factor_name] = np.zeros(len(returns) - window + 1)
        t_stats[factor_name] = np.zeros(len(returns) - window + 1)
    
    for i in range(window, len(returns) + 1):
        y = returns[i - window:i]
        
        # 팩터 수익률 데이터 준비
        X = np.ones((window, n_factors + 1))  # 상수항 포함
        
        for j, (factor_name, factor_rets) in enumerate(factor_returns.items()):
            X[:, j + 1] = factor_rets[i - window:i]
        
        # 회귀 분석
        model = sm.OLS(y, X)
        results = model.fit()
        
        # 회귀 계수 저장
        for j, factor_name in enumerate(factor_returns.keys()):
            coefficients[factor_name][i - window] = results.params[j + 1]  # 상수항 제외
            t_stats[factor_name][i - window] = results.tvalues[j + 1]
        
        # 결정 계수 저장
        r_squared[i - window] = results.rsquared
    
    # 시간축 설정
    if dates is not None:
        x = dates[window:]
    else:
        x = np.arange(window, len(returns) + 1)
    
    # 그래프 생성
    fig = plt.figure(figsize=figsize)
    
    # 1. 롤링 팩터 노출도
    ax1 = fig.add_subplot(2, 1, 1)
    
    for factor_name, coefs in coefficients.items():
        ax1.plot(x, coefs, label=factor_name)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'Rolling Factor Exposures ({window} days)')
    ax1.set_xlabel('Date' if dates is not None else 'Time')
    ax1.set_ylabel('Coefficient')
    ax1.legend()
    ax1.grid(True)
    
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # 2. 롤링 결정 계수
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax2.plot(x, r_squared, label='R-squared')
    ax2.set_title(f'Rolling R-squared ({window} days)')
    ax2.set_xlabel('Date' if dates is not None else 'Time')
    ax2.set_ylabel('R-squared')
    ax2.legend()
    ax2.grid(True)
    
    if dates is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    
    if dates is not None:
        fig.autofmt_xdate()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
