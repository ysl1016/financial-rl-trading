import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 기존 모듈 임포트
from src.data.data_processor import process_data, download_stock_data
from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent, GRPONetwork

# 새로 구현한 고급 기능 모듈 임포트
from src.models.multi_asset_env import MultiAssetTradingEnv
from src.utils.reward_functions import RewardFactory, CompositeReward
from src.utils.online_learning import OnlineLearningAgent, MarketRegimeDetector


def main():
    """
    DeepSeek-R1 기반 GRPO 에이전트의 고급 기능 통합 예제
    
    다음 기능을 테스트합니다:
    1. 다중 자산 트레이딩 환경
    2. 사용자 정의 보상 함수
    3. 온라인 학습 기능
    """
    print("DeepSeek-R1 기반 금융거래 강화학습 모델 고급 기능 테스트")
    print("-" * 70)
    
    # 경로 및 설정
    data_path = "./data"
    model_path = "./models"
    results_path = "./results"
    
    # 필요한 디렉토리 생성
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # 설정
    config = {
        'symbols': ['SPY', 'QQQ', 'GLD'],  # 자산 심볼
        'start_date': '2018-01-01',        # 시작 날짜
        'end_date': '2021-12-31',          # 종료 날짜
        'train_ratio': 0.7,                # 학습 데이터 비율
        'initial_capital': 100000,         # 초기 자본
        'max_position_size': 1.5,          # 최대 포지션 크기
        'hidden_dim': 128,                 # 은닉층 차원
        'batch_size': 64,                  # 배치 크기
        'buffer_capacity': 10000,          # 경험 버퍼 크기
        'learning_rate': 3e-4,             # 학습률
        'gamma': 0.99,                     # 할인 계수
        'reward_scale': 1.0,               # 보상 스케일
        'penalty_scale': 0.5,              # 패널티 스케일
        'num_epochs': 50,                  # 학습 에포크 수
        'steps_per_epoch': 1000,           # 에포크당 스텝 수
        'eval_episodes': 5,                # 평가 에피소드 수
        'update_interval': 100,            # 업데이트 간격
        'n_regimes': 4,                    # 시장 레짐 수
    }
    
    print(f"설정: {config}")
    print("-" * 70)
    print("1. 데이터 다운로드 및 처리")
    
    # 1. 데이터 다운로드 및 처리
    asset_data = {}
    for symbol in config['symbols']:
        print(f"  - {symbol} 처리 중...")
        splits = process_data(symbol, start_date=config['start_date'], end_date=config['end_date'])
        data = pd.concat([splits['train'], splits['val'], splits['test']])
        asset_data[symbol] = data
    
    # 학습/테스트 분할
    split_idx = int(len(next(iter(asset_data.values()))) * config['train_ratio'])
    
    train_data = {}
    test_data = {}
    for symbol, data in asset_data.items():
        train_data[symbol] = data.iloc[:split_idx].copy()
        test_data[symbol] = data.iloc[split_idx:].copy()
    
    print(f"  데이터 분할: 학습 {split_idx}일, 테스트 {len(next(iter(test_data.values())))}일")
    print("-" * 70)
    
    print("2. 다중 자산 트레이딩 환경 설정")
    
    # 2. 다중 자산 트레이딩 환경 생성
    env = MultiAssetTradingEnv(
        asset_data=train_data,
        initial_capital=config['initial_capital'],
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=config['max_position_size'],
        stop_loss_pct=0.02,
        max_asset_weight=0.7,  # 단일 자산 최대 비중
        correlation_lookback=20,  # 상관관계 계산을 위한 룩백 기간
        rebalancing_cost=0.0002,  # 리밸런싱 추가 비용
    )
    
    # 테스트 환경 생성
    test_env = MultiAssetTradingEnv(
        asset_data=test_data,
        initial_capital=config['initial_capital'],
        trading_cost=0.0005,
        slippage=0.0001,
        risk_free_rate=0.02,
        max_position_size=config['max_position_size'],
        stop_loss_pct=0.02,
        max_asset_weight=0.7,
        correlation_lookback=20,
        rebalancing_cost=0.0002,
    )
    
    print(f"  환경 생성 완료: 행동 공간 차원 {env.action_space.shape[0]}, 상태 공간 차원 {env.observation_space.shape[0]}")
    print("-" * 70)
    
    print("3. 사용자 정의 보상 함수 설정")
    
    # 3. 사용자 정의 보상 함수 구성
    reward_configs = [
        {'type': 'return', 'scale': 1.0, 'weight': 1.0},
        {'type': 'sharpe', 'risk_free_rate': 0.02, 'window_size': 30, 'weight': 0.5},
        {'type': 'drawdown', 'max_drawdown': 0.05, 'weight': 0.8},
        {'type': 'trading_cost', 'cost_scale': 1.0, 'weight': 0.3},
        {'type': 'diversification', 'weight': 0.4}
    ]
    
    # 복합 보상 함수 생성
    reward_function = RewardFactory.create_composite_reward(reward_configs)
    
    print(f"  보상 함수 구성: {len(reward_configs)}개 요소 결합")
    print("-" * 70)
    
    print("4. GRPO 에이전트 생성")
    
    # 4. GRPO 에이전트 생성
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # GRPO 네트워크 생성
    network = GRPONetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim']
    )
    
    # GRPO 에이전트 생성
    agent = GRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['learning_rate'],
        gamma=config['gamma'],
        reward_scale=config['reward_scale'],
        penalty_scale=config['penalty_scale']
    )
    
    # 온라인 학습을 위한 update_from_experiences 메서드 추가
    # (실제 구현에서는 GRPOAgent 클래스에 직접 통합해야 함)
    agent.update_from_experiences = lambda experiences, weights=None: update_from_experiences(agent, experiences, weights)
    
    print(f"  에이전트 생성 완료: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    print("-" * 70)
    
    print("5. 시장 레짐 감지기 생성")
    
    # 5. 시장 레짐 감지기 생성
    regime_detector = MarketRegimeDetector(
        n_regimes=config['n_regimes'],
        lookback_period=60,
        regime_features=None,  # 기본 특성 사용
        min_samples_per_regime=20,
        stability_threshold=0.6,
        transition_smoothing=5
    )
    
    # 레짐 감지기 초기화 (첫 번째 자산 데이터 사용)
    first_asset = next(iter(train_data.values()))
    regime_detector.update(first_asset)
    
    print(f"  레짐 감지기 생성 완료: {config['n_regimes']}개 레짐 감지")
    print("-" * 70)
    
    print("6. 온라인 학습 에이전트 생성")
    
    # 6. 온라인 학습 에이전트 생성
    online_agent = OnlineLearningAgent(
        model=agent,
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
        update_frequency=config['update_interval'],
        min_samples_before_update=200,
        save_frequency=1000,
        model_path=model_path,
        regime_detector=regime_detector
    )
    
    print(f"  온라인 학습 에이전트 생성 완료: 버퍼 용량 {config['buffer_capacity']}, 배치 크기 {config['batch_size']}")
    print("-" * 70)
    
    print("7. 학습 시작")
    
    # 7. 학습
    train_rewards = []
    eval_rewards = []
    
    # 현재 시간 기록
    start_time = datetime.now()
    
    for epoch in range(config['num_epochs']):
        episode_rewards = []
        episode_reward = 0
        state = env.reset()
        
        # 에포크 시작 시간
        epoch_start_time = datetime.now()
        
        # 경험 수집
        for step in range(config['steps_per_epoch']):
            # 행동 선택 (현재 시장 데이터로 레짐 업데이트)
            current_idx = env.index
            current_market_data = next(iter(train_data.values())).iloc[max(0, current_idx-10):current_idx+1]
            action = online_agent.select_action(state, market_data=current_market_data)
            
            # 환경 스텝
            next_state, reward, done, info = env.step(action)
            
            # 사용자 정의 보상 계산
            custom_reward = reward_function.calculate(
                current_state={'portfolio_value': info['portfolio_value'] - reward},
                next_state={'portfolio_value': info['portfolio_value']},
                action=action,
                info=info
            )
            
            # 경험 저장
            online_agent.store_experience(state, action, custom_reward, next_state, done)
            
            episode_reward += custom_reward
            
            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0
            else:
                state = next_state
        
        # 에포크 종료 시간 및 소요 시간
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        # 에이전트 평가
        eval_reward = evaluate_agent(test_env, online_agent, config['eval_episodes'])
        
        # 지표 저장
        train_rewards.append(np.mean(episode_rewards) if episode_rewards else 0)
        eval_rewards.append(eval_reward)
        
        # 진행 상황 출력
        print(f"\n에포크 {epoch + 1}/{config['num_epochs']} (소요 시간: {epoch_duration:.1f}초)")
        print(f"  학습 보상: {train_rewards[-1]:.2f}, 평가 보상: {eval_reward:.2f}")
        print(f"  현재 레짐: {online_agent.current_regime}, 안정성: {online_agent.regime_stability:.2f}")
        print(f"  탐색 입실론: {online_agent.epsilon:.3f}, 버퍼 크기: {len(online_agent.buffer.buffer)}")
        
        # 주기적 모델 저장
        if (epoch + 1) % 10 == 0:
            model_paths = online_agent.save_model(
                custom_path=os.path.join(model_path, f"model_epoch_{epoch+1}.pt")
            )
            print(f"  모델 저장 완료: {model_paths[0]}")
    
    # 총 학습 소요 시간
    total_duration = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n학습 완료! 총 소요 시간: {total_duration:.1f}분")
    print("-" * 70)
    
    print("8. 결과 시각화")
    
    # 8. 결과 시각화
    plt.figure(figsize=(15, 10))
    
    # 보상 그래프
    plt.subplot(2, 2, 1)
    plt.plot(train_rewards, label='학습')
    plt.plot(eval_rewards, label='평가')
    plt.title('에포크별 보상')
    plt.xlabel('에포크')
    plt.ylabel('평균 보상')
    plt.legend()
    plt.grid(True)
    
    # 레짐 변화 그래프
    plt.subplot(2, 2, 2)
    regime_changes = online_agent.metrics['regime_changes']
    if regime_changes:
        steps = [change['step'] for change in regime_changes]
        regimes = [change['new_regime'] for change in regime_changes]
        plt.scatter(steps, regimes, c=regimes, cmap='viridis', s=100)
        plt.title('시장 레짐 변화')
        plt.xlabel('스텝')
        plt.ylabel('레짐')
        plt.yticks(range(config['n_regimes']))
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '레짐 변화 없음', ha='center', va='center')
        plt.title('시장 레짐 변화')
    
    # 최종 포트폴리오 가치 그래프
    plt.subplot(2, 2, 3)
    state = test_env.reset()
    done = False
    portfolio_values = [test_env.portfolio_value]
    
    while not done:
        action = online_agent.select_action(state, deterministic=True)
        next_state, _, done, info = test_env.step(action)
        portfolio_values.append(info['portfolio_value'])
        state = next_state
    
    plt.plot(portfolio_values)
    plt.title('테스트 환경에서의 포트폴리오 가치')
    plt.xlabel('스텝')
    plt.ylabel('포트폴리오 가치 ($)')
    plt.grid(True)
    
    # 포트폴리오 가중치 그래프
    plt.subplot(2, 2, 4)
    state = test_env.reset()
    weights_history = []
    
    for _ in range(min(100, len(next(iter(test_data.values()))))):
        action = online_agent.select_action(state, deterministic=True)
        next_state, _, done, info = test_env.step(action)
        weights_history.append(info['weights'])
        state = next_state
        if done:
            break
    
    weights_history = np.array(weights_history)
    for i, symbol in enumerate(config['symbols']):
        plt.plot(weights_history[:, i], label=symbol)
    
    plt.title('자산별 포트폴리오 가중치')
    plt.xlabel('스텝')
    plt.ylabel('가중치')
    plt.legend()
    plt.grid(True)
    
    # 그래프 저장
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'training_results.png'))
    plt.show()
    
    print(f"  결과 그래프 저장 완료: {os.path.join(results_path, 'training_results.png')}")
    print("-" * 70)
    
    print("9. 최종 모델 저장")
    
    # 9. 최종 모델 저장
    final_model_paths = online_agent.save_model(
        custom_path=os.path.join(model_path, "final_model.pt")
    )
    
    print(f"  최종 모델 저장 완료:")
    print(f"  - 모델: {final_model_paths[0]}")
    print(f"  - 레짐 감지기: {final_model_paths[1]}")
    print(f"  - 에이전트 상태: {final_model_paths[2]}")
    print("-" * 70)
    
    print("모든 작업 완료!")


def evaluate_agent(env, agent, num_episodes):
    """에이전트 성능 평가"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)


def update_from_experiences(agent, experiences, weights=None):
    """
    배치 경험으로부터 GRPO 에이전트 업데이트
    
    이 함수는 온라인 학습 모듈에서 임포트하여 실제 GRPOAgent에 통합해야 합니다.
    """
    if not experiences:
        return {'loss': 0.0}
    
    # 경험 배치 언패킹
    states, actions, rewards, next_states, dones = zip(*experiences)
    
    # 텐서 변환
    states = torch.FloatTensor(np.array(states)).to(agent.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
    
    # NumPy 배열로 변환 (actions이 리스트가 아닌 경우를 처리)
    if isinstance(actions[0], (int, float, np.integer, np.floating)):
        actions = np.array(actions).reshape(-1)
    else:
        actions = np.array(actions)
    
    actions = torch.FloatTensor(actions).to(agent.device)
    rewards = torch.FloatTensor(np.array(rewards)).to(agent.device)
    dones = torch.FloatTensor(np.array(dones)).to(agent.device)
    
    # 가중치 처리
    if weights is not None:
        weights = torch.FloatTensor(weights).to(agent.device)
    else:
        weights = torch.ones_like(rewards)
    
    # 행동을 원핫 인코딩으로 변환 (연속 행동 공간을 위한 적응)
    actions_onehot = actions
    if len(actions.shape) == 1:
        # 이산 행동인 경우, 원핫 인코딩으로 변환
        actions_onehot = torch.zeros(len(actions), agent.action_dim).to(agent.device)
        actions_long = actions.long()
        actions_onehot.scatter_(1, actions_long.unsqueeze(1), 1)
    
    # 정책 확률
    action_probs = agent.network(states)
    
    # 현재 상태-행동 가치
    current_q = agent.network.estimate_q_value(states, actions_onehot).squeeze()
    
    # 다음 상태 최대 Q-값 (연속 행동 공간 대응)
    with torch.no_grad():
        next_action_probs = agent.network(next_states)
        next_actions = torch.argmax(next_action_probs, dim=1)
        
        next_actions_onehot = torch.zeros(len(next_actions), agent.action_dim).to(agent.device)
        next_actions_onehot.scatter_(1, next_actions.unsqueeze(1), 1)
        
        next_q = agent.network.estimate_q_value(next_states, next_actions_onehot).squeeze()
        
        # 타겟 Q-값
        target_q = rewards + agent.gamma * next_q * (1 - dones)
    
    # TD 오류 계산
    td_errors = target_q - current_q
    
    # 이점(advantage) 계산
    advantages = td_errors
    
    # 정책 손실 계산 (보상-페널티 분리)
    positive_mask = advantages > 0
    negative_mask = ~positive_mask
    
    # 손실 계산 (연속 행동 공간에 맞게 조정)
    policy_loss = torch.tensor(0.0).to(agent.device)
    
    # 가치 손실
    q_loss = (weights * (current_q - target_q.detach()).pow(2)).mean()
    
    # 총 손실
    total_loss = policy_loss + q_loss
    
    # 그라디언트 업데이트
    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()
    
    return {
        'loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'q_loss': q_loss.item(),
        'mean_reward': rewards.mean().item(),
        'mean_advantage': advantages.mean().item(),
        'td_errors': td_errors.detach().cpu().numpy()
    }


if __name__ == "__main__":
    main()
