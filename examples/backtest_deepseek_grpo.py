#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 기반 GRPO 에이전트에 대한 백테스팅 스크립트

이 스크립트는 훈련된 DeepSeek-R1 GRPO 에이전트의 성능을 다양한 시장 조건에서 
평가하고 결과를 분석 및 시각화합니다.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 자체 모듈 임포트
from src.data.data_processor import process_data, download_stock_data
from src.models.trading_env import TradingEnv
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent  # 아직 구현되지 않은 모듈
from src.utils.backtest_utils import (
    split_data_by_date, 
    generate_market_regimes, 
    run_backtest, 
    monte_carlo_simulation,
    simulate_market_shock
)
from src.utils.evaluation import (
    calculate_comprehensive_metrics, 
    compare_strategies, 
    statistical_significance_test
)
from src.utils.visualization import (
    plot_portfolio_performance,
    plot_return_distribution,
    plot_rolling_metrics,
    plot_trade_analysis,
    plot_market_regimes,
    plot_monte_carlo_simulation,
    plot_comparison_chart,
    create_performance_dashboard
)


def parse_args():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='DeepSeek-R1 GRPO 에이전트 백테스팅')
    
    # 데이터 관련 인수
    parser.add_argument('--symbol', type=str, default='SPY', 
                        help='백테스트할 주식 심볼')
    parser.add_argument('--start_date', type=str, default='2022-01-01', 
                        help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, 
                        help='종료 날짜 (YYYY-MM-DD, 기본값: 현재)')
    
    # 모델 관련 인수
    parser.add_argument('--model_path', type=str, required=True, 
                        help='훈련된 모델 경로')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                        help='모델의 은닉층 차원')
    parser.add_argument('--seq_length', type=int, default=20, 
                        help='시계열 시퀀스 길이')
    
    # 백테스팅 관련 인수
    parser.add_argument('--initial_capital', type=float, default=100000, 
                        help='초기 자본금')
    parser.add_argument('--trading_cost', type=float, default=0.0005, 
                        help='거래 비용 (기본값: 0.05%)')
    parser.add_argument('--slippage', type=float, default=0.0001, 
                        help='슬리피지 (기본값: 0.01%)')
    parser.add_argument('--risk_free_rate', type=float, default=0.02, 
                        help='무위험 수익률 (기본값: 2%)')
    parser.add_argument('--benchmark', action='store_true', 
                        help='벤치마크(Buy-and-Hold) 전략과 비교')
    
    # 고급 분석 인수
    parser.add_argument('--monte_carlo', type=int, default=0, 
                        help='몬테카를로 시뮬레이션 횟수 (0=수행하지 않음)')
    parser.add_argument('--market_shock', action='store_true', 
                        help='시장 충격 시뮬레이션 수행')
    parser.add_argument('--shock_magnitude', type=float, default=-0.1, 
                        help='시장 충격 크기 (기본값: -10%)')
    
    # 결과 관련 인수
    parser.add_argument('--output_dir', type=str, default='backtest_results', 
                        help='결과를 저장할 디렉토리')
    parser.add_argument('--save_results', action='store_true', 
                        help='결과를 파일로 저장')
    parser.add_argument('--verbose', action='store_true', 
                        help='상세 출력 활성화')
    
    return parser.parse_args()


def setup_environment(args):
    """
    백테스팅 환경을 설정합니다.
    
    Args:
        args: 명령행 인수
        
    Returns:
        data: 처리된 데이터
        test_data: 테스트 데이터
        env: 트레이딩 환경
        agent: 훈련된 에이전트
        device: 연산 장치
    """
    # 출력 디렉토리 생성
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 장치 설정 (GPU 사용 가능시 사용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"사용 장치: {device}")
    
    # 데이터 처리
    if args.verbose:
        print(f"{args.symbol} 데이터 처리 중...")
    
    data = process_data(args.symbol, args.start_date, args.end_date)
    
    # 데이터 분할
    if args.verbose:
        print(f"데이터 분할 중... 총 {len(data)} 데이터 포인트")
    
    train_data, val_data, test_data = split_data_by_date(data)
    
    if args.verbose:
        print(f"훈련 데이터: {len(train_data)} 포인트 ({train_data.index[0]} - {train_data.index[-1]})")
        print(f"검증 데이터: {len(val_data)} 포인트 ({val_data.index[0]} - {val_data.index[-1]})")
        print(f"테스트 데이터: {len(test_data)} 포인트 ({test_data.index[0]} - {test_data.index[-1]})")
    
    # 트레이딩 환경 생성
    env = TradingEnv(
        data=test_data,
        initial_capital=args.initial_capital,
        trading_cost=args.trading_cost,
        slippage=args.slippage,
        risk_free_rate=args.risk_free_rate
    )
    
    # 모델 차원 계산
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 에이전트 로드
    agent = DeepSeekGRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_length=args.seq_length,
        hidden_dim=args.hidden_dim,
        device=device
    )
    
    # 훈련된 모델 로드
    agent.load(args.model_path)
    
    if args.verbose:
        print(f"모델 로드 완료: {args.model_path}")
    
    return data, test_data, env, agent, device


def run_benchmark_strategy(env):
    """
    벤치마크 전략(Buy-and-Hold)을 실행합니다.
    
    Args:
        env: 트레이딩 환경
        
    Returns:
        백테스트 결과
    """
    class BuyAndHoldAgent:
        """단순 Buy-and-Hold 전략을 구현하는 에이전트"""
        def __init__(self, action_dim):
            self.action_dim = action_dim
        
        def select_action(self, state):
            """매수 행동 (1) 반환"""
            return 1  # 항상 매수
    
    # 벤치마크 에이전트 생성
    benchmark_agent = BuyAndHoldAgent(env.action_space.n)
    
    # 벤치마크 백테스트 실행
    benchmark_results = run_backtest(env, benchmark_agent, episodes=1, verbose=False)
    
    return benchmark_results


def run_backtest_with_market_regimes(env, agent, test_data):
    """
    시장 레짐 감지와 함께 백테스트를 실행합니다.
    
    Args:
        env: 트레이딩 환경
        agent: 훈련된 에이전트
        test_data: 테스트 데이터
        
    Returns:
        백테스트 결과, 시장 레짐 레이블, 레짐별 성과
    """
    # 시장 레짐 생성
    regime_data = generate_market_regimes(test_data)
    regimes = regime_data['MarketRegime'].values
    
    # 백테스트 실행
    results = run_backtest(env, agent, episodes=1, verbose=True)
    
    # 레짐별 성과 분석
    unique_regimes = np.unique(regimes)
    regime_performance = {}
    
    for regime in unique_regimes:
        regime_indices = np.where(regimes == regime)[0]
        
        if len(regime_indices) > 0:
            # 레짐 기간의 포트폴리오 가치 추출
            regime_values = []
            for idx in regime_indices:
                if idx < len(results['portfolio_values']) - 1:
                    regime_values.append(results['portfolio_values'][idx])
            
            if len(regime_values) > 1:
                # 레짐별 성과 계산
                regime_return = regime_values[-1] / regime_values[0] - 1
                regime_performance[regime] = {
                    'return': regime_return,
                    'duration': len(regime_values),
                    'avg_daily_return': np.mean(np.diff(regime_values) / regime_values[:-1])
                }
    
    return results, regimes, regime_performance


def save_backtest_results(results, metrics, args, output_prefix="backtest"):
    """
    백테스트 결과를 저장합니다.
    
    Args:
        results: 백테스트 결과 딕셔너리
        metrics: 성과 지표 딕셔너리
        args: 명령행 인수
        output_prefix: 출력 파일 접두사
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{output_prefix}_{args.symbol}_{timestamp}")
    
    # 결과 저장
    results_for_save = {k: v for k, v in results.items() if k != 'dates'}
    
    # 날짜 정보가 있는 경우 문자열로 변환
    if 'dates' in results:
        results_for_save['dates'] = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) 
                                     for d in results['dates']]
    
    # JSON으로 저장
    with open(f"{output_path}_results.json", 'w') as f:
        json.dump(results_for_save, f, indent=2)
    
    # 성과 지표 저장
    with open(f"{output_path}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"결과가 저장되었습니다: {output_path}")
    
    return output_path


def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_args()
    
    # 환경 설정
    data, test_data, env, agent, device = setup_environment(args)
    
    # 시장 레짐 분석 실행
    results, regimes, regime_performance = run_backtest_with_market_regimes(env, agent, test_data)
    
    # 성과 지표 계산
    metrics = calculate_comprehensive_metrics(results, risk_free_rate=args.risk_free_rate)
    
    print("\n===== DeepSeek-R1 GRPO 에이전트 백테스트 결과 =====")
    print(f"총 수익률: {metrics['total_return'] * 100:.2f}%")
    print(f"연간 수익률: {metrics['annualized_return'] * 100:.2f}%")
    print(f"변동성 (연간화): {metrics['volatility'] * 100:.2f}%")
    print(f"Sharpe 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino 비율: {metrics['sortino_ratio']:.2f}")
    print(f"최대 손실: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"승률: {metrics['win_rate'] * 100:.2f}%" if 'win_rate' in metrics else "승률: 계산 불가")
    print(f"거래 횟수: {metrics['trade_count']}" if 'trade_count' in metrics else "거래 횟수: 계산 불가")
    
    # 레짐별 성과 출력
    print("\n===== 시장 레짐별 성과 =====")
    for regime, perf in regime_performance.items():
        print(f"{regime}: 수익률 {perf['return'] * 100:.2f}%, 기간 {perf['duration']} 일, "
              f"일평균 수익률 {perf['avg_daily_return'] * 100:.4f}%")
    
    # 벤치마크 비교 (옵션)
    if args.benchmark:
        print("\n===== 벤치마크 전략(Buy-and-Hold) 비교 =====")
        benchmark_results = run_benchmark_strategy(env)
        benchmark_metrics = calculate_comprehensive_metrics(benchmark_results)
        
        print(f"벤치마크 총 수익률: {benchmark_metrics['total_return'] * 100:.2f}%")
        print(f"벤치마크 연간 수익률: {benchmark_metrics['annualized_return'] * 100:.2f}%")
        print(f"벤치마크 Sharpe 비율: {benchmark_metrics['sharpe_ratio']:.2f}")
        print(f"벤치마크 최대 손실: {benchmark_metrics['max_drawdown'] * 100:.2f}%")
        
        # 성과 비교 시각화
        strategy_results = {
            'DeepSeek-R1 GRPO': results,
            'Buy-and-Hold': benchmark_results
        }
        comparison_df = compare_strategies(strategy_results)
        print("\n===== 전략 비교 =====")
        print(comparison_df.round(4).to_string())
        
        # 통계적 유의성 테스트
        strategy_returns = np.diff(results['portfolio_values']) / results['portfolio_values'][:-1]
        benchmark_returns = np.diff(benchmark_results['portfolio_values']) / benchmark_results['portfolio_values'][:-1]
        
        t_test_result = statistical_significance_test(strategy_returns, benchmark_returns, test_type='t-test')
        
        print("\n===== 통계적 유의성 테스트 =====")
        print(f"t-검정 p-값: {t_test_result['p_value']:.4f}")
        print(f"통계적으로 유의미한 차이: {'예' if t_test_result['significant'] else '아니오'}")
        
        # 성과 비교 차트
        fig = plot_comparison_chart(strategy_results, title=f"{args.symbol} - 전략 비교")
        plt.show()
    
    # 몬테카를로 시뮬레이션 (옵션)
    if args.monte_carlo > 0:
        print(f"\n===== 몬테카를로 시뮬레이션 ({args.monte_carlo} 회) =====")
        mc_results = monte_carlo_simulation(env, agent, n_simulations=args.monte_carlo)
        
        print(f"평균 수익률: {mc_results['returns']['mean'] * 100:.2f}%")
        print(f"수익률 95% 신뢰구간: [{mc_results['returns']['ci_lower'] * 100:.2f}%, "
              f"{mc_results['returns']['ci_upper'] * 100:.2f}%]")
        print(f"평균 최대 손실: {mc_results['max_drawdowns']['mean'] * 100:.2f}%")
        print(f"평균 Sharpe 비율: {mc_results['sharpe_ratios']['mean']:.2f}")
        
        # 몬테카를로 시뮬레이션 시각화
        fig = plot_monte_carlo_simulation(mc_results)
        plt.show()
    
    # 시장 충격 시뮬레이션 (옵션)
    if args.market_shock:
        print(f"\n===== 시장 충격 시뮬레이션 (충격 크기: {args.shock_magnitude * 100:.1f}%) =====")
        
        # 시장 충격 데이터 생성
        shocked_data = simulate_market_shock(test_data, shock_magnitude=args.shock_magnitude)
        
        # 시장 충격 환경 생성
        shocked_env = TradingEnv(
            data=shocked_data,
            initial_capital=args.initial_capital,
            trading_cost=args.trading_cost,
            slippage=args.slippage,
            risk_free_rate=args.risk_free_rate
        )
        
        # 시장 충격 백테스트 실행
        shocked_results = run_backtest(shocked_env, agent, episodes=1, verbose=False)
        shocked_metrics = calculate_comprehensive_metrics(shocked_results)
        
        print(f"시장 충격 후 총 수익률: {shocked_metrics['total_return'] * 100:.2f}%")
        print(f"시장 충격 후 최대 손실: {shocked_metrics['max_drawdown'] * 100:.2f}%")
        print(f"시장 충격 전 성과 대비: {(shocked_metrics['total_return'] - metrics['total_return']) * 100:.2f}%")
        
        # 시장 충격 시각화
        strategy_results = {
            'Normal Market': results,
            'Market Shock': shocked_results
        }
        fig = plot_comparison_chart(strategy_results, title=f"{args.symbol} - 시장 충격 영향")
        plt.show()
    
    # 레짐별 성과 시각화
    fig = plot_market_regimes(results, regimes, title=f"{args.symbol} - 시장 레짐별 성과")
    plt.show()
    
    # 포트폴리오 성과 대시보드
    create_performance_dashboard(
        results,
        benchmark_results=benchmark_results if args.benchmark else None,
        title=f"{args.symbol} - DeepSeek-R1 GRPO 에이전트 성과"
    )
    
    # 결과 저장 (옵션)
    if args.save_results:
        save_path = save_backtest_results(results, metrics, args)
        
        if args.benchmark:
            save_backtest_results(benchmark_results, benchmark_metrics, args, output_prefix="benchmark")
        
        if args.market_shock:
            save_backtest_results(shocked_results, shocked_metrics, args, output_prefix="shock")


if __name__ == "__main__":
    main()
