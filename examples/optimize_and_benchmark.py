import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import process_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent
from src.utils.hyperparameter_optimization import HyperparameterOptimizer, ValidationFramework
from src.utils.benchmarking import StrategyBenchmark, GRPOStrategy


def main(args):
    """
    하이퍼파라미터 최적화, 모델 학습, 및 벤치마킹을 수행하는 메인 함수
    
    Args:
        args: 명령행 인수
    """
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 실행 시간 스탬프 (로그 디렉토리용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"로그 디렉토리: {log_dir}")
    print(f"모델 디렉토리: {model_dir}")
    
    # 주식 데이터 다운로드 및 처리
    print(f"\n데이터 다운로드 및 처리 중... 심볼: {args.symbol}")
    data_splits = process_data(args.symbol, start_date=args.start_date, end_date=args.end_date)
    data = pd.concat([data_splits['train'], data_splits['val'], data_splits['test']])
    print(f"처리된 데이터 크기: {len(data)} 행")
    
    # 데이터 저장 (선택적)
    if args.save_data:
        data_path = os.path.join(log_dir, f"{args.symbol}_processed_data.pkl")
        data.to_pickle(data_path)
        print(f"데이터 저장됨: {data_path}")
    
    # 훈련/검증/테스트 분할
    print("\n데이터 분할 중...")
    validation_framework = ValidationFramework(
        data=data,
        validation_method=args.validation_method,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        n_splits=args.n_splits
    )
    
    # 첫 번째 분할 가져오기
    splits = validation_framework.get_splits()
    train_data, val_data = splits[0]
    test_data = validation_framework.get_test_data()
    
    print(f"훈련 데이터: {len(train_data)} 행 ({len(train_data)/len(data):.1%})")
    print(f"검증 데이터: {len(val_data)} 행 ({len(val_data)/len(data):.1%})")
    print(f"테스트 데이터: {len(test_data)} 행 ({len(test_data)/len(data):.1%})")
    
    if args.optimize:
        # 하이퍼파라미터 최적화
        print("\n하이퍼파라미터 최적화 시작...")
        optimizer = HyperparameterOptimizer(
            train_data=train_data,
            val_data=val_data,
            log_dir=os.path.join(log_dir, "hyperopt"),
            device=args.device
        )
        
        optimization_result = optimizer.optimize(
            n_iter=args.n_iter,
            init_points=args.init_points
        )
        
        # 최적 파라미터 저장
        best_params = optimization_result["best_params"]
        best_params_path = os.path.join(log_dir, "best_params.json")
        
        import json
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n최적 파라미터 저장됨: {best_params_path}")
        print("최적 파라미터:")
        for param, value in best_params.items():
            formatted_value = int(value) if param == "hidden_dim" else value
            print(f"  {param}: {formatted_value}")
        
        # 최적 모델 저장 경로
        best_model_path = os.path.join(model_dir, "best_model.pt")
        
        # 최적 모델 사용
        best_agent = optimization_result["best_agent"]
    else:
        # 하이퍼파라미터 최적화 없이 기본 에이전트 생성
        print("\n기본 하이퍼파라미터로 에이전트 생성...")
        
        # 트레이딩 환경 생성 (상태 및 행동 공간 정보 확인용)
        env = TradingEnv(
            data=train_data,
            initial_capital=args.initial_capital,
            trading_cost=args.trading_cost,
            slippage=args.slippage,
            risk_free_rate=args.risk_free_rate,
            max_position_size=args.max_position_size,
            stop_loss_pct=args.stop_loss_pct
        )
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # 기본 에이전트 생성
        best_agent = GRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.learning_rate,
            gamma=args.gamma,
            reward_scale=args.reward_scale,
            penalty_scale=args.penalty_scale,
            device=args.device
        )
        
        # 최적 모델 저장 경로
        best_model_path = os.path.join(model_dir, "basic_model.pt")
    
    # 모델 학습
    if args.train:
        print("\n모델 학습 시작...")
        
        # 트레이딩 환경 생성
        train_env = TradingEnv(
            data=train_data,
            initial_capital=args.initial_capital,
            trading_cost=args.trading_cost,
            slippage=args.slippage,
            risk_free_rate=args.risk_free_rate,
            max_position_size=args.max_position_size,
            stop_loss_pct=args.stop_loss_pct
        )
        
        # 학습 메트릭 추적
        episode_rewards = []
        portfolio_values = []
        
        # 진행 상황 추적용
        from tqdm import tqdm
        
        # 에피소드 반복
        for episode in tqdm(range(args.train_episodes), desc="학습 중"):
            state = train_env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done:
                # 행동 선택
                action = best_agent.select_action(state)
                
                # 환경 스텝
                next_state, reward, done, info = train_env.step(action)
                
                # 경험 저장
                best_agent.store_transition(state, action, reward, next_state, done)
                
                # 정책 업데이트
                if step % args.update_interval == 0:
                    best_agent.update()
                
                # 상태 및 보상 업데이트
                state = next_state
                episode_reward += reward
                step += 1
            
            # 에피소드 메트릭 저장
            episode_rewards.append(episode_reward)
            portfolio_values.append(train_env.portfolio_values[-1])
            
            # 주기적으로 진행 상황 출력
            if (episode + 1) % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval:])
                avg_return = (np.mean(portfolio_values[-args.log_interval:]) / args.initial_capital) - 1
                print(f"에피소드 {episode + 1}/{args.train_episodes}: "
                      f"평균 보상 = {avg_reward:.4f}, 평균 수익률 = {avg_return:.2%}")
        
        # 학습 곡선 그리기
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('에피소드 보상')
        plt.xlabel('에피소드')
        plt.ylabel('총 보상')
        
        plt.subplot(1, 2, 2)
        portfolio_returns = [(val / args.initial_capital) - 1 for val in portfolio_values]
        plt.plot(portfolio_returns)
        plt.title('에피소드 수익률')
        plt.xlabel('에피소드')
        plt.ylabel('수익률')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'learning_curves.png'))
        
        # 모델 저장
        best_agent.save(best_model_path)
        print(f"\n학습된 모델 저장됨: {best_model_path}")
    
    # 벤치마킹
    if args.benchmark:
        print("\n벤치마킹 시작...")
        
        # 벤치마크 도구 생성
        benchmark = StrategyBenchmark(
            data=test_data,
            env_params={
                "initial_capital": args.initial_capital,
                "trading_cost": args.trading_cost,
                "slippage": args.slippage,
                "risk_free_rate": args.risk_free_rate,
                "max_position_size": args.max_position_size,
                "stop_loss_pct": args.stop_loss_pct
            },
            log_dir=os.path.join(log_dir, "benchmark"),
            random_seed=args.seed
        )
        
        # 표준 벤치마크 전략 추가
        benchmark.create_standard_strategies()
        
        # GRPO 에이전트 전략 추가
        benchmark.add_grpo_agent(best_agent, name="GRPO")
        
        # 벤치마크 실행
        results = benchmark.run_all_benchmarks(num_episodes=args.benchmark_episodes)
        
        # 통계적 유의성 검정
        significance = benchmark.statistical_significance_test(
            results=results,
            reference_strategy="Buy and Hold"  # Buy-and-Hold 전략과 비교
        )
        
        # 결과 시각화
        benchmark.plot_results(
            results=results,
            figsize=(16, 16),
            plot_path=os.path.join(log_dir, "benchmark_results.png")
        )
        
        print("\n벤치마크 완료!")
        
        # 최종 결과 요약
        print("\n성능 요약:")
        for strategy_name, result in results.items():
            metrics = result['avg_metrics']
            print(f"{strategy_name}:")
            print(f"  수익률: {metrics['total_return']:.2%}")
            print(f"  Sharpe 비율: {metrics['sharpe_ratio']:.4f}")
            print(f"  최대 손실률: {metrics['max_drawdown']:.2%}")
            print(f"  거래 횟수: {int(metrics['trades_count'])}")
        
        # GRPO vs Buy-and-Hold 통계적 유의성
        if 'GRPO' in significance['total_return']:
            sig_info = significance['total_return']['GRPO']
            sig_text = "통계적으로 유의함" if sig_info['significant'] else "통계적으로 유의하지 않음"
            print(f"\nGRPO vs Buy-and-Hold (수익률): {sig_text} (p={sig_info['p_value']:.4f})")
    
    print("\n작업 완료!")
    print(f"모든 결과는 {log_dir} 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL 트레이딩 모델 최적화 및 벤치마킹")
    
    # 데이터 관련 인수
    data_group = parser.add_argument_group('데이터 인수')
    data_group.add_argument("--symbol", type=str, default="SPY", help="주식 심볼")
    data_group.add_argument("--start_date", type=str, default="2020-01-01", help="시작 날짜 (YYYY-MM-DD)")
    data_group.add_argument("--end_date", type=str, default=None, help="종료 날짜 (YYYY-MM-DD)")
    data_group.add_argument("--save_data", action="store_true", help="처리된 데이터 저장 여부")
    
    # 환경 파라미터
    env_group = parser.add_argument_group('환경 파라미터')
    env_group.add_argument("--initial_capital", type=float, default=100000, help="초기 자본")
    env_group.add_argument("--trading_cost", type=float, default=0.0005, help="거래 비용")
    env_group.add_argument("--slippage", type=float, default=0.0001, help="슬리피지")
    env_group.add_argument("--risk_free_rate", type=float, default=0.02, help="무위험 수익률 (연율화)")
    env_group.add_argument("--max_position_size", type=float, default=1.0, help="최대 포지션 크기")
    env_group.add_argument("--stop_loss_pct", type=float, default=0.02, help="스톱로스 비율")
    
    # 검증 및 분할 관련 인수
    val_group = parser.add_argument_group('검증 인수')
    val_group.add_argument("--validation_method", type=str, default="expanding", 
                          choices=["expanding", "sliding", "k_fold"], help="검증 방법")
    val_group.add_argument("--test_ratio", type=float, default=0.2, help="테스트 데이터 비율")
    val_group.add_argument("--val_ratio", type=float, default=0.2, help="검증 데이터 비율")
    val_group.add_argument("--n_splits", type=int, default=5, help="교차 검증 분할 수")
    
    # 최적화 관련 인수
    opt_group = parser.add_argument_group('최적화 인수')
    opt_group.add_argument("--optimize", action="store_true", help="하이퍼파라미터 최적화 수행")
    opt_group.add_argument("--n_iter", type=int, default=30, help="최적화 반복 횟수")
    opt_group.add_argument("--init_points", type=int, default=5, help="초기 무작위 탐색 포인트 수")
    
    # 학습 관련 인수
    train_group = parser.add_argument_group('학습 인수')
    train_group.add_argument("--train", action="store_true", help="모델 학습 수행")
    train_group.add_argument("--train_episodes", type=int, default=100, help="학습 에피소드 수")
    train_group.add_argument("--update_interval", type=int, default=10, help="정책 업데이트 간격")
    train_group.add_argument("--log_interval", type=int, default=10, help="로깅 간격 (에피소드)")
    
    # 에이전트 하이퍼파라미터 (최적화 사용하지 않을 경우)
    agent_group = parser.add_argument_group('에이전트 인수')
    agent_group.add_argument("--hidden_dim", type=int, default=128, help="은닉층 차원")
    agent_group.add_argument("--learning_rate", type=float, default=3e-4, help="학습률")
    agent_group.add_argument("--gamma", type=float, default=0.99, help="감마 (할인계수)")
    agent_group.add_argument("--reward_scale", type=float, default=1.0, help="보상 스케일")
    agent_group.add_argument("--penalty_scale", type=float, default=0.5, help="페널티 스케일")
    
    # 벤치마킹 관련 인수
    bench_group = parser.add_argument_group('벤치마킹 인수')
    bench_group.add_argument("--benchmark", action="store_true", help="벤치마킹 수행")
    bench_group.add_argument("--benchmark_episodes", type=int, default=5, help="벤치마크 에피소드 수")
    
    # 일반 인수
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="사용할 디바이스 ('cuda' 또는 'cpu')")
    parser.add_argument("--log_dir", type=str, default="logs", help="로그 디렉토리 기본 경로")
    
    args = parser.parse_args()
    
    main(args)
