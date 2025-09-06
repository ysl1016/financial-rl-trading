import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 기존 모듈 임포트
from src.data.data_processor import process_data
from src.models.grpo_agent import GRPOAgent

# 새로 구현한 모듈 임포트
from src.models.multi_asset_env import MultiAssetTradingEnv
from src.utils.reward_functions import RewardFactory
from src.utils.online_learning import OnlineLearningAgent, MarketRegimeDetector


def main():
    """
    다중 자산 트레이딩, 사용자 정의 보상 함수, 온라인 학습을 통합한 예제
    
    이 예제는 다음을 보여줍니다:
    1. 다중 자산 환경 설정 및 사용
    2. 사용자 정의 복합 보상 함수 구성
    3. 시장 레짐 감지 및 온라인 학습
    """
    print("DeepSeek-R1 금융거래 모델 고급 기능 예제")
    print("-" * 60)
    
    # 데이터 및 모델 저장 경로 설정
    data_path = "./data"
    model_path = "./models"
    results_path = "./results"
    
    # 필요한 디렉토리 생성
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # 설정
    config = {
        'symbols': ['SPY', 'QQQ', 'GLD', 'TLT'],  # S&P 500, 나스닥 100, 금, 장기 국채
        'start_date': '2018-01-01',
        'end_date': '2022-01-01',
        'train_ratio': 0.8,
        'initial_capital': 100000,
        'max_position_size': 1.5,
        'batch_size': 64,
        'buffer_capacity': 10000,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'n_episodes': 10,
        'eval_interval': 5,
    }
    
    print(f"설정: {config}")
    print("-" * 60)
    
    print("1. 데이터 다운로드 및 처리")
    
    # 자산 데이터 다운로드 및 처리
    asset_data = {}
    for symbol in config['symbols']:
        print(f"  {symbol} 처리 중...")
        splits = process_data(symbol, start_date=config['start_date'], end_date=config['end_date'])
        data = pd.concat([splits['train'], splits['val'], splits['test']]).reset_index(drop=True)
        asset_data[symbol] = data
    
    # 학습/테스트 분할
    split_idx = int(len(next(iter(asset_data.values()))) * config['train_ratio'])
    
    train_data = {}
    test_data = {}
    for symbol, data in asset_data.items():
        train_data[symbol] = data.iloc[:split_idx].copy()
        test_data[symbol] = data.iloc[split_idx:].copy()
    
    print(f"  데이터 분할: 학습 {split_idx}일, 테스트 {len(next(iter(test_data.values())))}일")
    print("-" * 60)
    
    print("2. 다중 자산 트레이딩 환경 설정")
    
    # 다중 자산 트레이딩 환경 생성
    env = MultiAssetTradingEnv(
        asset_data=train_data,
        initial_capital=config['initial_capital'],
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=config['max_position_size'],
        stop_loss_pct=0.02,
        max_asset_weight=0.6,
        correlation_lookback=20,
        rebalancing_cost=0.0002,
        min_trading_amount=1000.0
    )
    
    # 테스트 환경
    test_env = MultiAssetTradingEnv(
        asset_data=test_data,
        initial_capital=config['initial_capital'],
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=config['max_position_size'],
        stop_loss_pct=0.02,
        max_asset_weight=0.6,
        correlation_lookback=20,
        rebalancing_cost=0.0002,
        min_trading_amount=1000.0
    )
    
    print(f"  환경 생성 완료: {len(config['symbols'])}개 자산, 행동 공간 차원 {env.action_space.shape[0]}")
    print("-" * 60)
    
    print("3. 사용자 정의 보상 함수 설정")
    
    # 사용자 정의 보상 함수 설정
    reward_configs = [
        {'type': 'return', 'scale': 1.0, 'weight': 1.0},
        {'type': 'sharpe', 'risk_free_rate': 0.02, 'window_size': 30, 'weight': 0.5},
        {'type': 'sortino', 'risk_free_rate': 0.02, 'window_size': 30, 'weight': 0.3},
        {'type': 'drawdown', 'max_drawdown': 0.05, 'weight': 0.8},
        {'type': 'trading_cost', 'cost_scale': 1.0, 'weight': 0.3},
        {'type': 'diversification', 'weight': 0.4}
    ]
    
    # 복합 보상 함수 생성
    reward_function = RewardFactory.create_composite_reward(reward_configs)
    
    print(f"  보상 함수 설정: {len(reward_configs)}개 요소")
    print("-" * 60)
    
    print("4. 시장 레짐 감지기 설정")
    
    # 시장 레짐 감지기 설정
    regime_detector = MarketRegimeDetector(
        n_regimes=4,
        lookback_period=60,
        min_samples_per_regime=20,
        stability_threshold=0.6,
        transition_smoothing=5
    )
    
    # 레짐 감지기 초기화 (첫 번째 자산의 학습 데이터로)
    first_symbol = config['symbols'][0]
    regime_detector.update(train_data[first_symbol])
    
    print(f"  레짐 감지기 설정: {regime_detector.n_regimes}개 레짐")
    print("-" * 60)
    
    print("5. GRPO 에이전트 설정")
    
    # GRPO 에이전트 생성
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = GRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        reward_scale=1.0,
        penalty_scale=0.5
    )
    
    # 온라인 학습 메서드 추가
    from src.utils.online_learning import update_from_experiences
    agent.update_from_experiences = lambda experiences, weights=None: update_from_experiences(agent, experiences, weights)
    
    print(f"  에이전트 설정: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    print("-" * 60)
    
    print("6. 온라인 학습 에이전트 설정")
    
    # 온라인 학습 에이전트 생성
    online_agent = OnlineLearningAgent(
        model=agent,
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
        update_frequency=5,
        min_samples_before_update=200,
        save_frequency=500,
        model_path=model_path,
        regime_detector=regime_detector,
        exploration_params={
            'initial_epsilon': 0.5,
            'final_epsilon': 0.01,
            'epsilon_decay': 0.998,
            'min_certainty': 0.3
        }
    )
    
    print(f"  온라인 학습 에이전트 설정: 버퍼 용량 {config['buffer_capacity']}, 배치 크기 {config['batch_size']}")
    print("-" * 60)
    
    print("7. 학습 시작")
    
    # 학습 메트릭 초기화
    train_rewards = []
    train_portfolio_values = []
    eval_rewards = []
    eval_portfolio_values = []
    regime_changes = []
    
    start_time = datetime.now()
    
    # 에피소드 기반 학습
    for episode in range(config['n_episodes']):
        # 에피소드 시작 시간
        episode_start_time = datetime.now()
        
        episode_reward = 0
        portfolio_values = []
        
        # 환경 초기화
        state = env.reset()
        done = False
        
        while not done:
            # 현재 시장 데이터로 레짐 업데이트
            current_idx = env.index
            current_market_data = next(iter(train_data.values())).iloc[max(0, current_idx-10):current_idx+1]
            
            # 행동 선택
            action = online_agent.select_action(state, market_data=current_market_data)
            
            # 환경 스텝
            next_state, reward, done, info = env.step(action)
            
            # 포트폴리오 가치 기록
            portfolio_values.append(info['portfolio_value'])
            
            # 사용자 정의 보상 계산
            custom_reward = reward_function.calculate(
                current_state={'portfolio_value': info['portfolio_value'] - reward},
                next_state={'portfolio_value': info['portfolio_value']},
                action=action,
                info=info
            )
            
            # 경험 저장 (사용자 정의 보상 사용)
            online_agent.store_experience(state, action, custom_reward, next_state, done)
            
            # 총 보상 누적
            episode_reward += custom_reward
            
            # 상태 업데이트
            state = next_state
        
        # 현재 레짐 상태
        regime_info = online_agent.regime_detector.get_regime_info()
        current_regime = regime_info['regime']
        regime_stability = regime_info['stability']
        
        # 학습 메트릭 기록
        train_rewards.append(episode_reward)
        train_portfolio_values.append(portfolio_values)
        
        # 레짐 변화 기록
        regime_changes.extend(online_agent.metrics['regime_changes'])
        
        # 주기적 평가
        if (episode + 1) % config['eval_interval'] == 0 or episode == config['n_episodes'] - 1:
            eval_reward, eval_portfolio_value = evaluate_agent(test_env, online_agent, reward_function)
            eval_rewards.append(eval_reward)
            eval_portfolio_values.append(eval_portfolio_value)
        else:
            eval_reward = None
        
        # 에피소드 소요 시간
        episode_duration = (datetime.now() - episode_start_time).total_seconds()
        
        # 진행 상황 출력
        print(f"  에피소드 {episode + 1}/{config['n_episodes']} 완료 (소요 시간: {episode_duration:.1f}초)")
        print(f"    학습 보상: {episode_reward:.2f}")
        print(f"    최종 포트폴리오 가치: ${portfolio_values[-1]:.2f}")
        print(f"    현재 레짐: {current_regime}, 안정성: {regime_stability:.2f}")
        
        if eval_reward is not None:
            print(f"    평가 보상: {eval_reward:.2f}")
        
        # 모델 저장
        if episode == config['n_episodes'] - 1:
            model_paths = online_agent.save_model(
                custom_path=os.path.join(model_path, "final_model.pt")
            )
            print(f"    모델 저장: {model_paths[0]}")
    
    # 총 학습 소요 시간
    total_duration = (datetime.now() - start_time).total_seconds() / 60.0
    print(f"\n학습 완료! 총 소요 시간: {total_duration:.1f}분")
    print("-" * 60)
    
    print("8. 결과 시각화")
    
    # 결과 시각화
    plt.figure(figsize=(16, 12))
    
    # 1. 학습 보상 그래프
    plt.subplot(2, 2, 1)
    plt.plot(train_rewards, marker='o', linestyle='-', label='학습 보상')
    plt.plot([i * config['eval_interval'] for i in range(len(eval_rewards))], 
             eval_rewards, marker='s', linestyle='--', label='평가 보상')
    plt.title('에피소드별 보상')
    plt.xlabel('에피소드')
    plt.ylabel('누적 보상')
    plt.grid(True)
    plt.legend()
    
    # 2. 포트폴리오 가치 그래프 (마지막 에피소드)
    plt.subplot(2, 2, 2)
    plt.plot(train_portfolio_values[-1], label='학습 (마지막 에피소드)')
    if eval_portfolio_values:
        plt.plot(eval_portfolio_values[-1], label='평가')
    plt.title('포트폴리오 가치 변화')
    plt.xlabel('스텝')
    plt.ylabel('포트폴리오 가치 ($)')
    plt.grid(True)
    plt.legend()
    
    # 3. 레짐 변화 그래프
    plt.subplot(2, 2, 3)
    if regime_changes:
        steps = [change['step'] for change in regime_changes]
        regimes = [change['new_regime'] for change in regime_changes]
        stabilities = [change['stability'] for change in regime_changes]
        
        plt.scatter(steps, regimes, c=stabilities, cmap='viridis', 
                   s=100, alpha=0.7, label='레짐 변화')
        plt.colorbar(label='레짐 안정성')
        plt.title('시장 레짐 변화')
        plt.xlabel('스텝')
        plt.ylabel('레짐')
        plt.yticks(range(regime_detector.n_regimes))
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '레짐 변화 감지되지 않음', 
                ha='center', va='center', fontsize=12)
        plt.title('시장 레짐 변화')
    
    # 4. 자산별 가중치 그래프 (마지막 에피소드 샘플링)
    plt.subplot(2, 2, 4)
    
    # 테스트 환경에서 가중치 추적
    test_env.reset()
    weights_history = []
    state = test_env.reset()
    done = False
    
    # 마지막 몇 개 스텝만 시각화
    max_steps = 50
    step_counter = 0
    
    while not done and step_counter < max_steps:
        # 시장 데이터 가져오기
        current_idx = test_env.index
        current_market_data = next(iter(test_data.values())).iloc[max(0, current_idx-10):current_idx+1]
        
        # 결정론적 행동 선택
        action = online_agent.select_action(state, market_data=current_market_data, deterministic=True)
        
        # 환경 스텝
        next_state, _, done, info = test_env.step(action)
        weights_history.append(info['weights'])
        state = next_state
        step_counter += 1
    
    # 가중치 시각화
    weights_history = np.array(weights_history)
    for i, symbol in enumerate(config['symbols']):
        plt.plot(weights_history[:, i], label=symbol)
    
    plt.title('자산별 포트폴리오 가중치')
    plt.xlabel('스텝')
    plt.ylabel('가중치')
    plt.grid(True)
    plt.legend()
    
    # 그래프 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'multi_asset_results.png'))
    plt.show()
    
    print(f"  결과 그래프 저장: {os.path.join(results_path, 'multi_asset_results.png')}")
    print("-" * 60)
    
    print("예제 완료!")


def evaluate_agent(env, agent, reward_function, max_episodes=1):
    """
    에이전트 평가
    
    Args:
        env: 평가 환경
        agent: 평가할 에이전트
        reward_function: 사용자 정의 보상 함수
        max_episodes: 평가할 에피소드 수
        
    Returns:
        (평균 보상, 포트폴리오 가치 히스토리)
    """
    total_rewards = []
    portfolio_values_list = []
    
    for _ in range(max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = []
        
        while not done:
            # 현재 시장 데이터
            current_idx = env.index
            first_asset = next(iter(env.asset_data.values()))
            current_market_data = first_asset.iloc[max(0, current_idx-10):current_idx+1]
            
            # 결정론적 행동 선택
            action = agent.select_action(state, market_data=current_market_data, deterministic=True)
            
            # 환경 스텝
            next_state, reward, done, info = env.step(action)
            
            # 포트폴리오 가치 기록
            portfolio_values.append(info['portfolio_value'])
            
            # 사용자 정의 보상 계산
            custom_reward = reward_function.calculate(
                current_state={'portfolio_value': info['portfolio_value'] - reward},
                next_state={'portfolio_value': info['portfolio_value']},
                action=action,
                info=info
            )
            
            episode_reward += custom_reward
            state = next_state
        
        total_rewards.append(episode_reward)
        portfolio_values_list.append(portfolio_values)
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    
    # 가장 좋은 성과의 포트폴리오 가치 히스토리 반환
    best_idx = np.argmax(total_rewards)
    best_portfolio_values = portfolio_values_list[best_idx]
    
    return avg_reward, best_portfolio_values


if __name__ == "__main__":
    main()
