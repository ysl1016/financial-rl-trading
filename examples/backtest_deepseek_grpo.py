#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 GRPO 에이전트 백테스팅 스크립트

이 스크립트는 DeepSeek-R1 기반 GRPO 에이전트의 성능을 다양한 시장 조건에서
테스트하고 평가하기 위한 백테스팅 환경을 제공합니다. 시장 레짐 분석, 벤치마크 비교,
몬테카를로 시뮬레이션, 스트레스 테스트 등 다양한 평가를 지원합니다.
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent
from src.utils.evaluation import (
    calculate_returns, calculate_sharpe_ratio, calculate_sortino_ratio,
    calculate_max_drawdown, calculate_calmar_ratio, calculate_omega_ratio,
    calculate_win_rate, calculate_profit_loss_ratio, calculate_expectancy,
    calculate_performance_summary, detect_market_regimes, calculate_regime_statistics,
    calculate_trade_statistics, calculate_drawdowns
)

# 로그 파일 설정
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="DeepSeek-R1 GRPO 에이전트 백테스팅")
    
    # 모델 및 데이터 관련 인수
    parser.add_argument('--model_path', type=str, required=True, help='불러올 모델 경로')
    parser.add_argument('--symbols', nargs='+', default=['SPY'], help='테스트할 심볼 목록')
    parser.add_argument('--start_date', type=str, default='2022-01-01', help='백테스트 시작 날짜')
    parser.add_argument('--end_date', type=str, default=None, help='백테스트 종료 날짜')
    parser.add_argument('--benchmark', type=str, default='SPY', help='벤치마크 심볼')
    
    # 백테스트 환경 관련 인수
    parser.add_argument('--initial_capital', type=float, default=100000.0, help='초기 자본금')
    parser.add_argument('--trading_cost', type=float, default=0.0005, help='거래 비용')
    parser.add_argument('--slippage', type=float, default=0.0001, help='슬리피지')
    parser.add_argument('--risk_free_rate', type=float, default=0.02, help='무위험 이자율')
    parser.add_argument('--max_position_size', type=float, default=1.0, help='최대 포지션 크기')
    parser.add_argument('--stop_loss_pct', type=float, default=0.02, help='손절매 비율')
    
    # 시뮬레이션 관련 인수
    parser.add_argument('--mc_simulations', type=int, default=100, help='몬테카를로 시뮬레이션 횟수')
    parser.add_argument('--stress_scenarios', type=int, default=5, help='스트레스 테스트 시나리오 수')
    parser.add_argument('--stress_factor', type=float, default=1.5, help='스트레스 테스트 변동성 증폭 계수')
    
    # 시각화 및 저장 관련 인수
    parser.add_argument('--output_dir', type=str, default='backtest_results', help='결과 저장 디렉토리')
    parser.add_argument('--log_trades', action='store_true', help='거래 로깅 활성화')
    parser.add_argument('--plot', action='store_true', help='결과 플롯 생성 및 저장')
    parser.add_argument('--regime_analysis', action='store_true', help='시장 레짐 분석 수행')
    parser.add_argument('--n_regimes', type=int, default=4, help='시장 레짐 수')
    
    # 실행 모드 관련 인수
    parser.add_argument('--deterministic', action='store_true', help='결정론적 행동 선택 사용')
    parser.add_argument('--seed', type=int, default=42, help='난수 발생기 시드')
    
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_backtest_environment(data, args):
    """백테스트 환경 생성"""
    env = TradingEnv(
        data=data,
        initial_capital=args.initial_capital,
        trading_cost=args.trading_cost,
        slippage=args.slippage,
        risk_free_rate=args.risk_free_rate,
        max_position_size=args.max_position_size,
        stop_loss_pct=args.stop_loss_pct
    )
    return env


def run_single_backtest(env, agent, deterministic=False):
    """단일 백테스트 실행"""
    state = env.reset()
    done = False
    total_reward = 0
    actions = []
    history_buffer = []
    
    # 히스토리 버퍼 초기화 (시퀀스 길이만큼)
    seq_length = agent.seq_length
    for _ in range(seq_length):
        history_buffer.append(np.zeros_like(state))
    
    while not done:
        # 히스토리 버퍼 업데이트
        history = np.array(history_buffer)
        
        # 행동 선택
        action = agent.select_action(state, history=history, deterministic=deterministic)
        
        # 환경 단계 수행
        next_state, reward, done, info = env.step(action)
        
        # 히스토리 버퍼 업데이트 (가장 오래된 상태 제거하고 새 상태 추가)
        history_buffer.pop(0)
        history_buffer.append(state)
        
        # 상태 업데이트
        state = next_state
        total_reward += reward
        
        # 행동 기록
        actions.append(action)
    
    return {
        'total_reward': total_reward,
        'portfolio_values': np.array(env.portfolio_values),
        'actions': np.array(actions),
        'trades': env.trades,
        'position': env.position
    }


def run_monte_carlo_simulations(env, agent, n_simulations=100, deterministic=False):
    """몬테카를로 시뮬레이션 실행"""
    results = []
    
    for i in tqdm(range(n_simulations), desc="몬테카를로 시뮬레이션"):
        # 환경 초기화 (초기 상태가 약간 다를 수 있도록 난수 시드 변경)
        np.random.seed(42 + i)
        
        # 백테스트 실행
        result = run_single_backtest(env, agent, deterministic)
        results.append(result)
    
    return results


def run_stress_test(data, env_args, agent, n_scenarios=5, vol_factor=1.5, deterministic=False):
    """스트레스 테스트 실행"""
    results = []
    
    for i in range(n_scenarios):
        # 변동성 증폭 계수 계산 (각 시나리오마다 다른 계수)
        scenario_factor = vol_factor * (0.8 + 0.4 * i / n_scenarios)
        
        # 가격 데이터 변형 (변동성 증폭)
        stress_data = data.copy()
        
        # 평균 가격 계산
        mean_prices = stress_data['Close'].mean()
        
        # 가격 변동 증폭
        stress_data['Close'] = mean_prices + scenario_factor * (stress_data['Close'] - mean_prices)
        stress_data['High'] = mean_prices + scenario_factor * (stress_data['High'] - mean_prices)
        stress_data['Low'] = mean_prices + scenario_factor * (stress_data['Low'] - mean_prices)
        
        # 스트레스 환경 생성
        stress_env = create_backtest_environment(stress_data, env_args)
        
        # 백테스트 실행
        result = run_single_backtest(stress_env, agent, deterministic)
        result['scenario'] = i
        result['vol_factor'] = scenario_factor
        
        results.append(result)
    
    return results


def analyze_market_regimes(returns, n_regimes=4):
    """시장 레짐 분석"""
    # 레짐 감지
    regime_labels = detect_market_regimes(returns, n_regimes=n_regimes)
    
    # 레짐별 통계
    regime_stats = calculate_regime_statistics(returns, regime_labels)
    
    return regime_labels, regime_stats


def calculate_benchmark_comparison(portfolio_values, benchmark_data):
    """벤치마크 비교 분석"""
    # 전략 수익률 계산
    strategy_returns = calculate_returns(portfolio_values)
    
    # 벤치마크 수익률 계산
    benchmark_values = benchmark_data['Close'].values
    benchmark_returns = calculate_returns(benchmark_values)
    
    # 길이 조정 (필요한 경우)
    min_length = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_length]
    benchmark_returns = benchmark_returns[:min_length]
    
    # 누적 수익률 계산
    strategy_cum_returns = np.cumprod(1 + strategy_returns) - 1
    benchmark_cum_returns = np.cumprod(1 + benchmark_returns) - 1
    
    # 초과 수익률 계산
    excess_returns = strategy_returns - benchmark_returns
    
    # 성과 요약
    comparison = {
        'strategy_final_return': strategy_cum_returns[-1],
        'benchmark_final_return': benchmark_cum_returns[-1],
        'outperformance': strategy_cum_returns[-1] - benchmark_cum_returns[-1],
        'tracking_error': np.std(excess_returns),
        'information_ratio': np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0,
        'correlation': np.corrcoef(strategy_returns, benchmark_returns)[0, 1],
        'beta': np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0,
    }
    
    # 업/다운 마켓 성과
    up_market = benchmark_returns > 0
    down_market = benchmark_returns < 0
    
    if np.sum(up_market) > 0:
        comparison['up_market_performance'] = np.mean(strategy_returns[up_market])
        comparison['up_market_benchmark'] = np.mean(benchmark_returns[up_market])
        if np.sum(benchmark_returns[up_market]) != 0:
            comparison['up_capture'] = np.sum(strategy_returns[up_market]) / np.sum(benchmark_returns[up_market])
    
    if np.sum(down_market) > 0:
        comparison['down_market_performance'] = np.mean(strategy_returns[down_market])
        comparison['down_market_benchmark'] = np.mean(benchmark_returns[down_market])
        if np.sum(benchmark_returns[down_market]) != 0:
            comparison['down_capture'] = np.sum(strategy_returns[down_market]) / np.sum(benchmark_returns[down_market])
    
    return comparison, strategy_returns, benchmark_returns, strategy_cum_returns, benchmark_cum_returns


def plot_backtest_results(results, benchmark_data=None, regime_labels=None, output_dir='backtest_results'):
    """백테스트 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 포트폴리오 가치 플롯
    plt.figure(figsize=(14, 7))
    plt.plot(results['portfolio_values'], label='Portfolio Value')
    
    if benchmark_data is not None:
        # 벤치마크 가치 스케일링 (초기 포트폴리오 가치와 맞추기)
        initial_value = results['portfolio_values'][0]
        benchmark_values = benchmark_data['Close'].values
        scaled_benchmark = benchmark_values * (initial_value / benchmark_values[0])
        
        plt.plot(scaled_benchmark, label='Benchmark', alpha=0.7)
    
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'portfolio_value.png'), dpi=300)
    
    # 수익률 플롯
    returns = np.diff(results['portfolio_values']) / results['portfolio_values'][:-1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(returns, label='Daily Returns')
    plt.title('Daily Returns')
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'daily_returns.png'), dpi=300)
    
    # 수익률 분포
    plt.figure(figsize=(14, 7))
    sns.histplot(returns, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=300)
    
    # 누적 수익률
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=300)
    
    # 롤링 메트릭스
    window = min(252, len(returns) // 4)  # 적어도 4개의 윈도우가 있도록 설정
    
    if window > 10:  # 충분한 데이터가 있는 경우에만 롤링 메트릭스 계산
        rolling_sharpe = np.full(len(returns), np.nan)
        rolling_volatility = np.full(len(returns), np.nan)
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            rolling_sharpe[i - 1] = calculate_sharpe_ratio(window_returns, 0, 252)
            rolling_volatility[i - 1] = np.std(window_returns) * np.sqrt(252)
        
        plt.figure(figsize=(14, 7))
        plt.plot(rolling_sharpe, label=f'Rolling Sharpe Ratio ({window} days)')
        plt.title('Rolling Sharpe Ratio')
        plt.xlabel('Time Step')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'rolling_sharpe.png'), dpi=300)
        
        plt.figure(figsize=(14, 7))
        plt.plot(rolling_volatility, label=f'Rolling Volatility ({window} days)')
        plt.title('Rolling Volatility (Annualized)')
        plt.xlabel('Time Step')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'rolling_volatility.png'), dpi=300)
    
    # 드로다운 플롯
    drawdown, drawdown_details = calculate_drawdowns(results['portfolio_values'])
    
    plt.figure(figsize=(14, 7))
    plt.plot(drawdown, label='Drawdown')
    plt.title('Portfolio Drawdown')
    plt.xlabel('Time Step')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300)
    
    # 포지션 플롯
    actions = results['actions']
    positions = np.zeros(len(actions))
    
    # 포지션 계산 (0: 보유 없음, 1: 롱, -1: 숏)
    current_position = 0
    for i, action in enumerate(actions):
        if action == 1:  # 매수
            current_position = 1
        elif action == 2:  # 매도
            current_position = -1
        positions[i] = current_position
    
    plt.figure(figsize=(14, 7))
    plt.plot(positions, label='Position')
    plt.title('Position Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.yticks([-1, 0, 1], ['Short', 'No Position', 'Long'])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'positions.png'), dpi=300)
    
    # 시장 레짐 분석
    if regime_labels is not None:
        # 레짐별 색상
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
        
        plt.figure(figsize=(14, 10))
        
        # 포트폴리오 가치 플롯
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(results['portfolio_values'], label='Portfolio Value')
        
        if benchmark_data is not None:
            ax1.plot(scaled_benchmark, label='Benchmark', alpha=0.7)
        
        # 레짐 배경 색상
        unique_regimes = np.unique(regime_labels)
        for regime in unique_regimes:
            if np.isnan(regime):
                continue
                
            regime_int = int(regime)
            mask = regime_labels == regime
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                for start_idx in indices:
                    ax1.axvspan(start_idx, start_idx + 1, alpha=0.2, color=colors[regime_int % len(colors)])
        
        ax1.set_title('Portfolio Value with Market Regimes')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # 레짐 플롯
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(regime_labels, label='Market Regime')
        ax2.set_title('Market Regimes')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Regime')
        ax2.set_yticks(range(len(unique_regimes)))
        ax2.set_yticklabels([f'Regime {int(r)}' for r in unique_regimes if not np.isnan(r)])
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'market_regimes.png'), dpi=300)
    
    plt.close('all')


def plot_benchmark_comparison(strategy_returns, benchmark_returns, strategy_cum_returns, benchmark_cum_returns, output_dir='backtest_results'):
    """벤치마크 비교 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 누적 수익률 비교
    plt.figure(figsize=(14, 7))
    plt.plot(strategy_cum_returns, label='Strategy')
    plt.plot(benchmark_cum_returns, label='Benchmark')
    plt.title('Cumulative Returns: Strategy vs Benchmark')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cumulative_returns_comparison.png'), dpi=300)
    
    # 롤링 베타
    window = min(252, len(strategy_returns) // 4)
    
    if window > 10:
        rolling_beta = np.full(len(strategy_returns), np.nan)
        rolling_correlation = np.full(len(strategy_returns), np.nan)
        
        for i in range(window, len(strategy_returns) + 1):
            s_window = strategy_returns[i - window:i]
            b_window = benchmark_returns[i - window:i]
            
            # 상관관계
            rolling_correlation[i - 1] = np.corrcoef(s_window, b_window)[0, 1]
            
            # 베타
            cov = np.cov(s_window, b_window)[0, 1]
            var = np.var(b_window)
            if var > 0:
                rolling_beta[i - 1] = cov / var
        
        plt.figure(figsize=(14, 7))
        plt.plot(rolling_beta, label=f'Rolling Beta ({window} days)')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Beta = 1')
        plt.title('Rolling Beta to Benchmark')
        plt.xlabel('Time Step')
        plt.ylabel('Beta')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'rolling_beta.png'), dpi=300)
        
        plt.figure(figsize=(14, 7))
        plt.plot(rolling_correlation, label=f'Rolling Correlation ({window} days)')
        plt.title('Rolling Correlation to Benchmark')
        plt.xlabel('Time Step')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'rolling_correlation.png'), dpi=300)
    
    # 초과 수익률
    excess_returns = strategy_returns - benchmark_returns
    cumul_excess = np.cumsum(excess_returns)
    
    plt.figure(figsize=(14, 7))
    plt.plot(cumul_excess, label='Cumulative Excess Returns')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Cumulative Excess Returns Over Benchmark')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Excess Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cumulative_excess_returns.png'), dpi=300)
    
    plt.close('all')


def plot_monte_carlo_results(mc_results, output_dir='backtest_results'):
    """몬테카를로 시뮬레이션 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 시각화
    plt.figure(figsize=(14, 7))
    
    # 개별 시뮬레이션 (투명도를 낮게 설정)
    for i in range(min(30, len(all_portfolio_values))):  # 가시성을 위해 최대 30개만 표시
        plt.plot(all_portfolio_values[i], alpha=0.1, color='gray')
    
    # 평균 및 백분위수
    plt.plot(mean_values, label='Mean', color='blue', linewidth=2)
    plt.plot(median_values, label='Median', color='green', linewidth=2)
    plt.fill_between(range(min_length), percentile_5, percentile_95, alpha=0.2, color='blue', label='5-95 Percentile')
    plt.fill_between(range(min_length), percentile_25, percentile_75, alpha=0.3, color='blue', label='25-75 Percentile')
    
    plt.title('Monte Carlo Simulation: Portfolio Value')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'monte_carlo_portfolio.png'), dpi=300)
    
    # 최종 포트폴리오 가치 분포
    final_values = all_portfolio_values[:, -1]
    
    plt.figure(figsize=(14, 7))
    sns.histplot(final_values, kde=True)
    plt.axvline(x=mean_values[-1], color='r', linestyle='--', label=f'Mean: ${mean_values[-1]:.2f}')
    plt.axvline(x=median_values[-1], color='g', linestyle='--', label=f'Median: ${median_values[-1]:.2f}')
    plt.title('Distribution of Final Portfolio Values')
    plt.xlabel('Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'monte_carlo_final_values.png'), dpi=300)
    
    # 수익률 통계
    returns = np.diff(all_portfolio_values, axis=1) / all_portfolio_values[:, :-1]
    mean_returns = np.mean(returns, axis=1)
    
    plt.figure(figsize=(14, 7))
    sns.histplot(mean_returns, kde=True)
    plt.axvline(x=np.mean(mean_returns), color='r', linestyle='--', label=f'Mean: {np.mean(mean_returns):.4f}')
    plt.title('Distribution of Average Daily Returns')
    plt.xlabel('Average Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'monte_carlo_returns.png'), dpi=300)
    
    # 최대 손실 분포
    max_drawdowns = []
    for values in all_portfolio_values:
        drawdown = calculate_max_drawdown(values)
        max_drawdowns.append(drawdown)
    
    plt.figure(figsize=(14, 7))
    sns.histplot(max_drawdowns, kde=True)
    plt.axvline(x=np.mean(max_drawdowns), color='r', linestyle='--', label=f'Mean: {np.mean(max_drawdowns):.4f}')
    plt.title('Distribution of Maximum Drawdowns')
    plt.xlabel('Maximum Drawdown')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'monte_carlo_drawdowns.png'), dpi=300)
    
    plt.close('all')


def plot_stress_test_results(stress_results, output_dir='backtest_results'):
    """스트레스 테스트 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 시나리오별 포트폴리오 가치
    plt.figure(figsize=(14, 7))
    
    for result in stress_results:
        label = f"Scenario {result['scenario']} (Vol Factor: {result['vol_factor']:.2f})"
        plt.plot(result['portfolio_values'], label=label)
    
    plt.title('Stress Test: Portfolio Value Under Different Scenarios')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'stress_test_portfolio.png'), dpi=300)
    
    # 스트레스 테스트 요약
    scenario_data = []
    
    for result in stress_results:
        portfolio_values = result['portfolio_values']
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        scenario_data.append({
            'scenario': result['scenario'],
            'vol_factor': result['vol_factor'],
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(portfolio_values),
            'volatility': np.std(returns) * np.sqrt(252)
        })
    
    # 변동성 계수에 따른 성과 지표
    vol_factors = [data['vol_factor'] for data in scenario_data]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(vol_factors, [data['total_return'] for data in scenario_data], 'o-')
    plt.title('Total Return vs Volatility Factor')
    plt.xlabel('Volatility Factor')
    plt.ylabel('Total Return')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(vol_factors, [data['sharpe_ratio'] for data in scenario_data], 'o-')
    plt.title('Sharpe Ratio vs Volatility Factor')
    plt.xlabel('Volatility Factor')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(vol_factors, [data['max_drawdown'] for data in scenario_data], 'o-')
    plt.title('Maximum Drawdown vs Volatility Factor')
    plt.xlabel('Volatility Factor')
    plt.ylabel('Maximum Drawdown')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(vol_factors, [data['volatility'] for data in scenario_data], 'o-')
    plt.title('Portfolio Volatility vs Volatility Factor')
    plt.xlabel('Volatility Factor')
    plt.ylabel('Portfolio Volatility')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stress_test_metrics.png'), dpi=300)
    
    plt.close('all')


def save_backtest_results(results, metrics, comparison=None, regime_stats=None, mc_analysis=None, stress_analysis=None, output_dir='backtest_results'):
    """백테스트 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 요약
    summary = {
        'portfolio_value': {
            'initial': float(results['portfolio_values'][0]),
            'final': float(results['portfolio_values'][-1]),
            'high': float(np.max(results['portfolio_values'])),
            'low': float(np.min(results['portfolio_values']))
        },
        'return': {
            'total': float((results['portfolio_values'][-1] / results['portfolio_values'][0]) - 1),
            'annualized': float(metrics.get('annualized_return', 0.0))
        },
        'risk': {
            'volatility': float(metrics.get('annualized_volatility', 0.0)),
            'max_drawdown': float(metrics.get('max_drawdown', 0.0)),
            'sharpe_ratio': float(metrics.get('sharpe_ratio', 0.0)),
            'sortino_ratio': float(metrics.get('sortino_ratio', 0.0)),
            'calmar_ratio': float(metrics.get('calmar_ratio', 0.0))
        },
        'trading': {
            'win_rate': float(metrics.get('win_rate', 0.0)),
            'profit_loss_ratio': float(metrics.get('profit_loss_ratio', 0.0)),
            'expectancy': float(metrics.get('expectancy', 0.0)),
            'total_trades': len(results['trades'])
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 벤치마크 비교 추가
    if comparison is not None:
        summary['benchmark'] = {}
        for key, value in comparison.items():
            summary['benchmark'][key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    # 레짐 분석 추가
    if regime_stats is not None:
        summary['regimes'] = {}
        for regime, stats in regime_stats.items():
            summary['regimes'][regime] = {}
            for key, value in stats.items():
                summary['regimes'][regime][key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    # 몬테카를로 분석 추가
    if mc_analysis is not None:
        summary['monte_carlo'] = {}
        for key, value in mc_analysis.items():
            summary['monte_carlo'][key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    # 스트레스 테스트 분석 추가
    if stress_analysis is not None:
        summary['stress_test'] = {}
        for i, scenario in enumerate(stress_analysis):
            summary['stress_test'][f'scenario_{i}'] = {}
            for key, value in scenario.items():
                if key not in ['portfolio_values', 'actions', 'trades']:
                    summary['stress_test'][f'scenario_{i}'][key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    # JSON 파일로 저장
    with open(os.path.join(output_dir, 'backtest_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 포트폴리오 가치 저장
    np.save(os.path.join(output_dir, 'portfolio_values.npy'), results['portfolio_values'])
    
    # 거래 기록 저장
    with open(os.path.join(output_dir, 'trades.json'), 'w') as f:
        json.dump(results['trades'], f, indent=4)


def load_benchmark_data(symbol, start_date, end_date):
    """벤치마크 데이터 로드"""
    try:
        splits = process_data(symbol, start_date, end_date)
        benchmark_data = pd.concat([splits['train'], splits['val'], splits['test']])
        return benchmark_data
    except Exception as e:
        logger.error(f"벤치마크 데이터 로드 오류: {e}")
        return None


def analyze_monte_carlo_results(mc_results):
    """몬테카를로 시뮬레이션 결과 분석"""
    if not mc_results:
        return {}
    
    # 최종 포트폴리오 가치 통계
    final_values = []
    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for result in mc_results:
        values = result['portfolio_values']
        returns = np.diff(values) / values[:-1]
        
        final_values.append(values[-1])
        total_returns.append((values[-1] / values[0]) - 1)
        sharpe_ratios.append(calculate_sharpe_ratio(returns))
        max_drawdowns.append(calculate_max_drawdown(values))
    
    # 분석 결과
    analysis = {
        'final_value_mean': np.mean(final_values),
        'final_value_median': np.median(final_values),
        'final_value_std': np.std(final_values),
        'final_value_5th': np.percentile(final_values, 5),
        'final_value_95th': np.percentile(final_values, 95),
        'total_return_mean': np.mean(total_returns),
        'total_return_median': np.median(total_returns),
        'total_return_std': np.std(total_returns),
        'total_return_5th': np.percentile(total_returns, 5),
        'total_return_95th': np.percentile(total_returns, 95),
        'sharpe_mean': np.mean(sharpe_ratios),
        'sharpe_median': np.median(sharpe_ratios),
        'max_drawdown_mean': np.mean(max_drawdowns),
        'max_drawdown_median': np.median(max_drawdowns),
        'max_drawdown_worst': np.max(max_drawdowns),
        'success_rate': np.mean([r > 0 for r in total_returns])
    }
    
    return analysis


def analyze_trades(trades):
    """거래 분석"""
    if not trades:
        return {}
    
    trade_analysis = {}
    
    # 매수/매도 거래 구분
    buy_trades = [t for t in trades if t[0] in ['buy', 'close_short']]
    sell_trades = [t for t in trades if t[0] in ['short', 'close_long']]
    
    # 거래 타입별 수
    trade_analysis['buy_count'] = len(buy_trades)
    trade_analysis['sell_count'] = len(sell_trades)
    
    # 거래 타입별 비율
    trade_types = {}
    for t in trades:
        trade_type = t[0]
        trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
    
    for trade_type, count in trade_types.items():
        trade_analysis[f'{trade_type}_ratio'] = count / len(trades)
    
    return trade_analysis


def main():
    """메인 함수"""
    import torch
    
    # 인수 파싱
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 랜덤 시드 설정
    set_random_seed(args.seed)
    
    # 모델 로드
    logger.info(f"모델 로드 중: {args.model_path}")
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 모델 하이퍼파라미터 추출
        model_args = checkpoint.get('args', {})
        state_dim = model_args.get('state_dim', 14)  # 기본값 설정
        action_dim = model_args.get('action_dim', 3)  # 기본값 설정
        hidden_dim = model_args.get('hidden_dim', 256)  # 기본값 설정
        seq_length = model_args.get('seq_length', 50)  # 기본값 설정
        
        # DeepSeek GRPO 에이전트 생성
        agent = DeepSeekGRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # 모델 가중치 로드
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        
        logger.info("모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 오류: {e}")
        return
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {args.symbols}")
    
    all_data = []
    for symbol in args.symbols:
        try:
            splits = process_data(symbol, start_date=args.start_date, end_date=args.end_date)
            data = pd.concat([splits['train'], splits['val'], splits['test']])
            all_data.append(data)
        except Exception as e:
            logger.error(f"{symbol} 데이터 로드 오류: {e}")
    
    if not all_data:
        logger.error("데이터 로드 실패")
        return
    
    # 첫 번째 심볼 사용 (추후 다중 자산 지원 확장 가능)
    data = all_data[0]
    
    # 벤치마크 데이터 로드
    benchmark_data = None
    if args.benchmark:
        logger.info(f"벤치마크 데이터 로드 중: {args.benchmark}")
        benchmark_data = load_benchmark_data(args.benchmark, args.start_date, args.end_date)
    
    # 백테스트 환경 생성
    logger.info("백테스트 환경 생성 중")
    env = create_backtest_environment(data, args)
    
    # 단일 백테스트 실행
    logger.info("백테스트 실행 중")
    results = run_single_backtest(env, agent, deterministic=args.deterministic)
    
    # 성과 지표 계산
    logger.info("성과 지표 계산 중")
    portfolio_values = results['portfolio_values']
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    metrics = calculate_performance_summary(
        returns=returns,
        prices=portfolio_values,
        risk_free_rate=args.risk_free_rate
    )
    
    # 거래 통계 계산
    trade_stats = calculate_trade_statistics(results['trades'])
    metrics.update(trade_stats)
    
    # 결과 출력
    logger.info("\n===== 백테스트 결과 =====")
    logger.info(f"초기 자본: ${portfolio_values[0]:.2f}")
    logger.info(f"최종 자본: ${portfolio_values[-1]:.2f}")
    logger.info(f"총 수익률: {metrics['total_return']:.2%}")
    logger.info(f"연간 수익률: {metrics['annualized_return']:.2%}")
    logger.info(f"연간 변동성: {metrics['annualized_volatility']:.2%}")
    logger.info(f"Sharpe 비율: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino 비율: {metrics['sortino_ratio']:.2f}")
    logger.info(f"최대 손실: {metrics['max_drawdown']:.2%}")
    logger.info(f"승률: {metrics['win_rate']:.2%}")
    logger.info(f"총 거래 수: {metrics['total_trades']}")
    
    # 벤치마크 비교
    comparison = None
    strategy_returns = None
    benchmark_returns = None
    strategy_cum_returns = None
    benchmark_cum_returns = None
    
    if benchmark_data is not None:
        logger.info("\n===== 벤치마크 비교 =====")
        comparison, strategy_returns, benchmark_returns, strategy_cum_returns, benchmark_cum_returns = calculate_benchmark_comparison(
            portfolio_values, benchmark_data
        )
        
        logger.info(f"전략 최종 수익률: {comparison['strategy_final_return']:.2%}")
        logger.info(f"벤치마크 최종 수익률: {comparison['benchmark_final_return']:.2%}")
        logger.info(f"초과 성과: {comparison['outperformance']:.2%}")
        logger.info(f"정보 비율: {comparison['information_ratio']:.2f}")
        logger.info(f"베타: {comparison['beta']:.2f}")
        
        if 'up_capture' in comparison:
            logger.info(f"상승장 캡처 비율: {comparison['up_capture']:.2f}")
        
        if 'down_capture' in comparison:
            logger.info(f"하락장 캡처 비율: {comparison['down_capture']:.2f}")
    
    # 시장 레짐 분석
    regime_labels = None
    regime_stats = None
    
    if args.regime_analysis:
        logger.info("\n===== 시장 레짐 분석 =====")
        regime_labels, regime_stats = analyze_market_regimes(returns, n_regimes=args.n_regimes)
        
        for regime, stats in regime_stats.items():
            logger.info(f"\nRegime {regime}:")
            logger.info(f"데이터 포인트 수: {stats['count']}")
            logger.info(f"평균 수익률: {stats['mean_return']:.4%}")
            logger.info(f"표준 편차: {stats['std_return']:.4%}")
            logger.info(f"승률: {stats['win_rate']:.2%}")
            logger.info(f"왜도: {stats['skewness']:.4f}")
            logger.info(f"첨도: {stats['kurtosis']:.4f}")
    
    # 몬테카를로 시뮬레이션
    mc_results = None
    mc_analysis = None
    
    if args.mc_simulations > 0:
        logger.info(f"\n===== 몬테카를로 시뮬레이션 ({args.mc_simulations}회) =====")
        mc_results = run_monte_carlo_simulations(env, agent, n_simulations=args.mc_simulations, deterministic=args.deterministic)
        
        mc_analysis = analyze_monte_carlo_results(mc_results)
        
        logger.info(f"평균 최종 자본: ${mc_analysis['final_value_mean']:.2f}")
        logger.info(f"중앙값 최종 자본: ${mc_analysis['final_value_median']:.2f}")
        logger.info(f"5% 백분위 수익률: {mc_analysis['total_return_5th']:.2%}")
        logger.info(f"95% 백분위 수익률: {mc_analysis['total_return_95th']:.2%}")
        logger.info(f"평균 Sharpe 비율: {mc_analysis['sharpe_mean']:.2f}")
        logger.info(f"평균 최대 손실: {mc_analysis['max_drawdown_mean']:.2%}")
        logger.info(f"최악의 최대 손실: {mc_analysis['max_drawdown_worst']:.2%}")
        logger.info(f"성공률 (양의 수익률): {mc_analysis['success_rate']:.2%}")
    
    # 스트레스 테스트
    stress_results = None
    
    if args.stress_scenarios > 0:
        logger.info(f"\n===== 스트레스 테스트 ({args.stress_scenarios} 시나리오) =====")
        stress_results = run_stress_test(
            data, args, agent, 
            n_scenarios=args.stress_scenarios, 
            vol_factor=args.stress_factor,
            deterministic=args.deterministic
        )
        
        for i, result in enumerate(stress_results):
            portfolio_values = result['portfolio_values']
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            max_dd = calculate_max_drawdown(portfolio_values)
            
            logger.info(f"\n시나리오 {i} (변동성 계수: {result['vol_factor']:.2f}):")
            logger.info(f"최종 자본: ${portfolio_values[-1]:.2f}")
            logger.info(f"총 수익률: {total_return:.2%}")
            logger.info(f"최대 손실: {max_dd:.2%}")
    
    # 결과 저장
    logger.info("\n결과 저장 중...")
    save_backtest_results(
        results, metrics, comparison, regime_stats, mc_analysis, stress_results, output_dir
    )
    
    # 결과 시각화
    if args.plot:
        logger.info("결과 시각화 중...")
        
        # 백테스트 결과 시각화
        plot_backtest_results(results, benchmark_data, regime_labels, output_dir)
        
        # 벤치마크 비교 시각화
        if benchmark_data is not None and strategy_returns is not None and benchmark_returns is not None:
            plot_benchmark_comparison(
                strategy_returns, benchmark_returns, 
                strategy_cum_returns, benchmark_cum_returns, 
                output_dir
            )
        
        # 몬테카를로 시뮬레이션 시각화
        if mc_results is not None:
            plot_monte_carlo_results(mc_results, output_dir)
        
        # 스트레스 테스트 시각화
        if stress_results is not None:
            plot_stress_test_results(stress_results, output_dir)
    
    logger.info(f"\n백테스트 완료! 결과는 '{output_dir}' 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    main()
