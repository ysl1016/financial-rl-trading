#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 기반 GRPO 에이전트 학습 스크립트

고급 기술적 지표, 거시경제 데이터, 뉴스 감성 정보를 활용하는
향상된 강화학습 트레이딩 모델의 학습 예제
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.enhanced_processor import EnhancedDataProcessor
from src.models.enhanced_trading_env import EnhancedTradingEnv
from src.models.deepseek_grpo_agent import DeepSeekGRPOAgent

# 로거 설정
logger = logging.getLogger('DeepSeekGRPO-Train')

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="DeepSeek-R1 기반 GRPO 에이전트 학습")
    
    # 데이터 관련 인수
    parser.add_argument('--symbol', type=str, default='SPY', help='트레이딩 심볼')
    parser.add_argument('--start-date', type=str, default='2018-01-01', help='학습 시작 날짜')
    parser.add_argument('--end-date', type=str, default='2022-12-31', help='학습 종료 날짜')
    parser.add_argument('--test-start-date', type=str, default='2023-01-01', help='테스트 시작 날짜')
    parser.add_argument('--test-end-date', type=str, default=None, help='테스트 종료 날짜')
    parser.add_argument('--use-macro', action='store_true', help='거시경제 데이터 사용')
    parser.add_argument('--use-news', action='store_true', help='뉴스 감성 데이터 사용')
    parser.add_argument('--data-cache-dir', type=str, default='data/cache', help='데이터 캐시 디렉토리')
    
    # 환경 관련 인수
    parser.add_argument('--initial-capital', type=float, default=100000, help='초기 자본금')
    parser.add_argument('--trading-cost', type=float, default=0.0005, help='거래 비용')
    parser.add_argument('--slippage', type=float, default=0.0001, help='슬리피지')
    parser.add_argument('--risk-free-rate', type=float, default=0.02, help='무위험 이자율 (연간)')
    parser.add_argument('--max-position-size', type=float, default=1.0, help='최대 포지션 크기')
    parser.add_argument('--stop-loss-pct', type=float, default=0.02, help='손절매 비율')
    parser.add_argument('--window-size', type=int, default=20, help='관찰 윈도우 크기')
    parser.add_argument('--reward-type', type=str, default='sharpe',
                        choices=['simple', 'sharpe', 'sortino', 'calmar'],
                        help='보상 함수 유형')
    parser.add_argument('--action-type', type=str, default='discrete',
                        choices=['discrete', 'continuous', 'hybrid'],
                        help='행동 공간 유형')
    
    # 모델 관련 인수
    parser.add_argument('--hidden-dim', type=int, default=256, help='히든 레이어 차원')
    parser.add_argument('--lr', type=float, default=3e-4, help='학습률')
    parser.add_argument('--gamma', type=float, default=0.99, help='할인 계수')
    parser.add_argument('--kl-coef', type=float, default=0.01, help='KL 발산 계수')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='엔트로피 계수')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO 클리핑 파라미터')
    
    # 학습 관련 인수
    parser.add_argument('--num-epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--steps-per-epoch', type=int, default=1000, help='에포크 당 스텝 수')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기')
    parser.add_argument('--update-epochs', type=int, default=3, help='업데이트 에포크 수')
    parser.add_argument('--update-interval', type=int, default=1000, help='업데이트 간격')
    parser.add_argument('--eval-episodes', type=int, default=5, help='평가 에피소드 수')
    parser.add_argument('--save-interval', type=int, default=10, help='저장 간격 (에포크)')
    
    # 기타 인수
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='학습 장치 (GPU/CPU)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='체크포인트 디렉토리')
    parser.add_argument('--log-dir', type=str, default='logs', help='로그 디렉토리')
    parser.add_argument('--use-existing-data', action='store_true',
                        help='기존 처리된 데이터 사용 (다운로드 스킵)')
    parser.add_argument('--no-render', action='store_true', help='렌더링 비활성화')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='로그 레벨')
    
    return parser.parse_args()

def setup_logging(args):
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 로그 파일 경로
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'train_{args.symbol}_{timestamp}.log')
    
    # 로거 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"로깅 설정 완료. 로그 파일: {log_file}")
    return log_file

def setup_directories(args):
    """디렉토리 설정"""
    # 캐시 디렉토리
    os.makedirs(args.data_cache_dir, exist_ok=True)
    
    # 체크포인트 디렉토리
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.symbol)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 결과 디렉토리
    results_dir = os.path.join('results', args.symbol)
    os.makedirs(results_dir, exist_ok=True)
    
    return checkpoint_dir, results_dir

def set_random_seeds(seed):
    """랜덤 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(args, existing_train_data=None, existing_test_data=None):
    """데이터 준비"""
    if existing_train_data is not None and existing_test_data is not None and args.use_existing_data:
        logger.info("기존 데이터 사용")
        train_data = existing_train_data
        test_data = existing_test_data
    else:
        logger.info(f"{args.symbol} 데이터 처리 중...")
        
        # 데이터 프로세서 초기화
        processor = EnhancedDataProcessor(
            use_advanced_indicators=True,
            use_macro_data=args.use_macro,
            use_news_data=args.use_news,
            normalization_method='adaptive',
            data_cache_dir=args.data_cache_dir
        )
        
        # 학습 데이터 처리
        train_data = processor.process_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # 테스트 데이터 처리
        test_data = processor.process_data(
            symbol=args.symbol,
            start_date=args.test_start_date,
            end_date=args.test_end_date
        )
        
        logger.info(f"데이터 처리 완료: 학습 {len(train_data)} 행, 테스트 {len(test_data)} 행")
    
    return train_data, test_data

def create_env(data, args):
    """트레이딩 환경 생성"""
    env = EnhancedTradingEnv(
        data=data,
        initial_capital=args.initial_capital,
        trading_cost=args.trading_cost,
        slippage=args.slippage,
        risk_free_rate=args.risk_free_rate,
        max_position_size=args.max_position_size,
        stop_loss_pct=args.stop_loss_pct,
        window_size=args.window_size,
        reward_type=args.reward_type,
        action_type=args.action_type,
        use_risk_adjustment=True,
        use_market_indicators=True
    )
    
    return env

def create_agent(env, args, checkpoint_dir):
    """GRPO 에이전트 생성"""
    # 상태 및 행동 차원 확인
    state_dim = env.observation_space.shape[0]
    
    if args.action_type == 'discrete':
        action_dim = env.action_space.n
    elif args.action_type == 'continuous':
        action_dim = env.action_space.shape[0]
    else:  # 'hybrid'
        action_dim = env.action_space.spaces[0].n
    
    # 특성 차원 계산 (시계열 특성에 사용)
    feature_dim = len(env.feature_columns)
    
    # 에이전트 생성
    agent = DeepSeekGRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_length=args.window_size,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        clip_epsilon=args.clip_epsilon,
        device=args.device,
        checkpoint_dir=checkpoint_dir
    )
    
    return agent

def train_agent(env, agent, args):
    """에이전트 학습"""
    logger.info("에이전트 학습 시작")
    
    # 학습 지표 추적
    train_rewards = []
    eval_rewards = []
    train_metrics = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"에포크 {epoch+1}/{args.num_epochs} 시작")
        episode_rewards = []
        episode_lengths = []
        
        # 환경 초기화
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 히스토리 버퍼 초기화 (시계열 특성용)
        history_buffer = []
        feature_dim = len(env.feature_columns)
        
        # 초기 히스토리 버퍼 채우기
        for _ in range(args.window_size):
            history_buffer.append(np.zeros(feature_dim))
        
        # 환경에서 현재 특성 추출
        current_features = np.array([state[i] for i in range(len(env.feature_columns))])
        history_buffer.append(current_features)
        history_buffer.pop(0)
        
        # 프로그레스 바
        progress_bar = tqdm(range(args.steps_per_epoch), desc=f"에포크 {epoch+1}")
        
        for step in progress_bar:
            # 현재 히스토리로 행동 선택
            action, _ = agent.select_action(
                state,
                np.array(history_buffer),
                deterministic=False,
                exploration_scale=1.0 - 0.5 * epoch / args.num_epochs  # 점차 탐색 감소
            )
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 히스토리 버퍼 업데이트
            next_features = np.array([next_state[i] for i in range(len(env.feature_columns))])
            history_buffer.append(next_features)
            history_buffer.pop(0)
            
            # 경험 저장
            agent.store_transition(
                state, action, reward, next_state, done,
                np.array(history_buffer[:-1])  # 현재 상태 제외한 히스토리
            )
            
            # 상태 업데이트
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 프로그레스 바 업데이트
            progress_bar.set_postfix({
                'reward': f"{reward:.4f}",
                'acc_reward': f"{episode_reward:.4f}"
            })
            
            # 에피소드 종료 처리
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 성능 로깅
                logger.info(f"에피소드 종료: 보상 {episode_reward:.4f}, 길이 {episode_length}, "
                          f"최종 가치 ${info['portfolio_value']:.2f}")
                
                # 환경 초기화
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # 히스토리 버퍼 초기화
                history_buffer = []
                for _ in range(args.window_size):
                    history_buffer.append(np.zeros(feature_dim))
            
            # 주기적 업데이트
            if (step + 1) % args.update_interval == 0:
                update_metrics = agent.update(
                    epochs=args.update_epochs,
                    batch_size=args.batch_size
                )
                
                if update_metrics:
                    train_metrics.append(update_metrics)
                    logger.debug(f"업데이트 완료: 정책 손실 {update_metrics['policy_loss']:.4f}, "
                                f"가치 손실 {update_metrics.get('value_loss', 0):.4f}")
        
        # 에포크 평균 보상
        epoch_reward = np.mean(episode_rewards) if episode_rewards else 0
        train_rewards.append(epoch_reward)
        
        # 에이전트 평가
        eval_reward = evaluate_agent(env, agent, args.eval_episodes)
        eval_rewards.append(eval_reward)
        
        # 진행 상황 출력
        logger.info(f"에포크 {epoch+1} 완료: 평균 훈련 보상 {epoch_reward:.4f}, "
                   f"평가 보상 {eval_reward:.4f}")
        
        # 주기적으로 모델 저장
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                args.checkpoint_dir,
                args.symbol,
                f"deepseek_grpo_epoch_{epoch+1}.pt"
            )
            agent.save(save_path)
            logger.info(f"체크포인트 저장: {save_path}")
    
    # 최종 모델 저장
    final_path = os.path.join(
        args.checkpoint_dir,
        args.symbol,
        "deepseek_grpo_final.pt"
    )
    agent.save(final_path)
    logger.info(f"최종 모델 저장: {final_path}")
    
    return train_rewards, eval_rewards, train_metrics

def evaluate_agent(env, agent, num_episodes):
    """에이전트 평가"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        # 히스토리 버퍼 초기화
        history_buffer = []
        feature_dim = len(env.feature_columns)
        for _ in range(agent.seq_length):
            history_buffer.append(np.zeros(feature_dim))
        
        while not done:
            # 현재 특성 추출
            current_features = np.array([state[i] for i in range(len(env.feature_columns))])
            history_buffer.append(current_features)
            history_buffer.pop(0)
            
            # 결정론적 행동 선택 (평가 시)
            action, _ = agent.select_action(
                state, np.array(history_buffer), deterministic=True
            )
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
            episode_length += 1
        
        # 에피소드 결과 추가
        total_rewards.append(total_reward)
        
        logger.debug(f"평가 에피소드 {episode+1}/{num_episodes}: "
                    f"보상 {total_reward:.4f}, 길이 {episode_length}, "
                    f"최종 가치 ${info['portfolio_value']:.2f}")
    
    # 평균 보상 반환
    mean_reward = np.mean(total_rewards)
    logger.info(f"평가 완료: 평균 보상 {mean_reward:.4f}")
    
    return mean_reward

def backtest_agent(env, agent, args, results_dir):
    """에이전트 백테스트"""
    logger.info("백테스트 시작")
    
    # 환경 초기화
    state = env.reset()
    done = False
    
    # 히스토리 버퍼 초기화
    history_buffer = []
    feature_dim = len(env.feature_columns)
    for _ in range(agent.seq_length):
        history_buffer.append(np.zeros(feature_dim))
    
    # 백테스트 지표
    portfolio_values = [env.initial_capital]
    positions = [0]  # 0: 무포지션, 1: 롱, -1: 숏
    actions = []
    rewards = []
    dates = []
    
    # 날짜 인덱스 확인
    if isinstance(env.data.index, pd.DatetimeIndex):
        dates.append(env.data.index[env.index])
    else:
        dates.append(env.index)
    
    step = 0
    while not done:
        step += 1
        
        # 현재 특성 추출
        current_features = np.array([state[i] for i in range(len(env.feature_columns))])
        history_buffer.append(current_features)
        history_buffer.pop(0)
        
        # 결정론적 행동 선택 (백테스트 시)
        action, action_info = agent.select_action(
            state, np.array(history_buffer), deterministic=True
        )
        
        # 행동 정보 저장
        actions.append({
            'action': action,
            'action_probs': action_info['action_probs'].tolist(),
            'regime_probs': action_info['regime_probs'].tolist(),
            'exploration_temp': action_info['exploration_temp']
        })
        
        # 환경에서 스텝 실행
        next_state, reward, done, info = env.step(action)
        
        # 지표 업데이트
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        rewards.append(reward)
        
        # 날짜 추가
        if isinstance(env.data.index, pd.DatetimeIndex):
            if env.index < len(env.data):
                dates.append(env.data.index[env.index])
        else:
            dates.append(env.index)
        
        # 정보 출력 (100 스텝마다)
        if step % 100 == 0:
            logger.info(f"백테스트 스텝 {step}: 포트폴리오 가치 ${info['portfolio_value']:.2f}, "
                       f"위치 {info['position']}, 수익률 {info['return']:.4f}")
        
        # 상태 업데이트
        state = next_state
    
    # 백테스트 결과
    metrics = env.get_performance_metrics()
    
    # 결과 출력
    logger.info("백테스트 완료:")
    logger.info(f"초기 자본: ${metrics['initial_capital']:.2f}")
    logger.info(f"최종 가치: ${metrics['final_value']:.2f}")
    logger.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
    logger.info(f"연간 수익률: {metrics['annualized_return']*100:.2f}%")
    logger.info(f"Sharpe 비율: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"Sortino 비율: {metrics['sortino_ratio']:.4f}")
    logger.info(f"최대 손실: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"총 거래 횟수: {metrics['total_trades']}")
    
    # 결과 시각화 및 저장
    plot_backtest_results(
        dates, portfolio_values, positions, actions, metrics,
        args.symbol, results_dir
    )
    
    # 결과 저장
    save_backtest_results(
        dates, portfolio_values, positions, actions, rewards, metrics,
        args.symbol, results_dir
    )
    
    return metrics

def plot_backtest_results(dates, portfolio_values, positions, actions, metrics, symbol, results_dir):
    """백테스트 결과 시각화"""
    # 날짜 변환
    if not isinstance(dates[0], datetime):
        date_objects = [datetime.strptime(str(d), '%Y-%m-%d') if isinstance(d, str) else None for d in dates]
    else:
        date_objects = dates
    
    # 결측 날짜 처리
    valid_indices = [i for i, d in enumerate(date_objects) if d is not None]
    if valid_indices:
        valid_dates = [date_objects[i] for i in valid_indices]
        valid_values = [portfolio_values[i] for i in valid_indices]
        valid_positions = [positions[i] for i in valid_indices]
    else:
        valid_dates = list(range(len(portfolio_values)))
        valid_values = portfolio_values
        valid_positions = positions
    
    # 포트폴리오 가치 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(valid_dates, valid_values, 'b-', linewidth=1.5)
    plt.title(f'{symbol} 백테스트 결과', fontsize=14)
    plt.ylabel('포트폴리오 가치 ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 매수/매도 지점 표시
    buy_dates = [valid_dates[i] for i in range(1, len(valid_positions)) 
                if valid_positions[i] > 0 and valid_positions[i-1] <= 0]
    buy_values = [valid_values[i] for i in range(1, len(valid_positions)) 
                if valid_positions[i] > 0 and valid_positions[i-1] <= 0]
    
    sell_dates = [valid_dates[i] for i in range(1, len(valid_positions)) 
                if valid_positions[i] <= 0 and valid_positions[i-1] > 0]
    sell_values = [valid_values[i] for i in range(1, len(valid_positions)) 
                if valid_positions[i] <= 0 and valid_positions[i-1] > 0]
    
    plt.scatter(buy_dates, buy_values, color='green', marker='^', s=50, label='매수')
    plt.scatter(sell_dates, sell_values, color='red', marker='v', s=50, label='매도')
    
    # 포지션 시각화
    plt.subplot(2, 1, 2)
    plt.plot(valid_dates, valid_positions, 'r-', linewidth=1.5)
    plt.title('포지션 (1=롱, 0=무포지션, -1=숏)', fontsize=14)
    plt.ylabel('포지션', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # 성과 지표 표시
    info_text = (
        f"초기 자본: ${metrics['initial_capital']:,.2f}\n"
        f"최종 가치: ${metrics['final_value']:,.2f}\n"
        f"총 수익률: {metrics['total_return']*100:.2f}%\n"
        f"연간 수익률: {metrics['annualized_return']*100:.2f}%\n"
        f"Sharpe 비율: {metrics['sharpe_ratio']:.4f}\n"
        f"Sortino 비율: {metrics['sortino_ratio']:.4f}\n"
        f"최대 손실: {metrics['max_drawdown']*100:.2f}%\n"
        f"총 거래 횟수: {metrics['total_trades']}"
    )
    
    plt.figtext(0.15, 0.01, info_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(results_dir, f'{symbol}_backtest_{timestamp}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    logger.info(f"백테스트 차트 저장: {fig_path}")
    
    plt.close()

def save_backtest_results(dates, portfolio_values, positions, actions, rewards, metrics, symbol, results_dir):
    """백테스트 결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV 저장 경로
    csv_path = os.path.join(results_dir, f'{symbol}_backtest_{timestamp}.csv')
    
    # 날짜 형식 변환
    if not isinstance(dates[0], str):
        date_strs = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]
    else:
        date_strs = dates
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Date': date_strs,
        'PortfolioValue': portfolio_values,
        'Position': positions,
        'Reward': rewards + [np.nan],  # 보상은 하나 적음
    })
    
    # CSV로 저장
    df.to_csv(csv_path, index=False)
    logger.info(f"백테스트 데이터 저장: {csv_path}")
    
    # 액션 정보 및 지표 JSON 저장
    json_path = os.path.join(results_dir, f'{symbol}_backtest_details_{timestamp}.json')
    
    # 메트릭 JSON 직렬화 가능하게 변환
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            json_metrics[k] = v
        else:
            json_metrics[k] = float(v)
    
    # JSON 저장
    with open(json_path, 'w') as f:
        json.dump({
            'metrics': json_metrics,
            'actions': actions
        }, f, indent=2)
    
    logger.info(f"백테스트 상세 정보 저장: {json_path}")

def plot_training_metrics(train_rewards, eval_rewards, train_metrics, args, results_dir):
    """학습 지표 시각화"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 보상 그래프
    plt.figure(figsize=(15, 10))
    
    # 1. 보상 그래프
    plt.subplot(2, 2, 1)
    plt.plot(train_rewards, label='Train')
    plt.plot(eval_rewards, label='Eval')
    plt.title('학습 및 평가 보상', fontsize=14)
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('평균 보상', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 손실 그래프
    if train_metrics:
        epochs = list(range(len(train_metrics)))
        policy_losses = [m['policy_loss'] for m in train_metrics]
        value_losses = [m.get('value_loss', 0) for m in train_metrics]
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, policy_losses, label='Policy Loss')
        plt.plot(epochs, value_losses, label='Value Loss')
        plt.title('학습 손실', fontsize=14)
        plt.xlabel('업데이트 횟수', fontsize=12)
        plt.ylabel('손실', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 엔트로피 및 KL 발산
        entropies = [m.get('entropy', 0) for m in train_metrics]
        kl_divs = [m.get('kl_div', 0) for m in train_metrics]
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, entropies, label='Entropy')
        plt.plot(epochs, kl_divs, label='KL Divergence')
        plt.title('엔트로피 및 KL 발산', fontsize=14)
        plt.xlabel('업데이트 횟수', fontsize=12)
        plt.ylabel('값', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 메타 컨트롤러 파라미터
        reward_scales = [m.get('reward_scale', 1.0) for m in train_metrics]
        penalty_scales = [m.get('penalty_scale', 0.5) for m in train_metrics]
        exploration_temps = [m.get('exploration_temp', 1.0) for m in train_metrics]
        
        plt.subplot(2, 2, 4)
        plt.plot(epochs, reward_scales, label='Reward Scale')
        plt.plot(epochs, penalty_scales, label='Penalty Scale')
        plt.plot(epochs, exploration_temps, label='Exploration Temp')
        plt.title('메타 컨트롤러 파라미터', fontsize=14)
        plt.xlabel('업데이트 횟수', fontsize=12)
        plt.ylabel('값', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일 저장
    metrics_path = os.path.join(results_dir, f'{args.symbol}_metrics_{timestamp}.png')
    plt.savefig(metrics_path, dpi=200, bbox_inches='tight')
    logger.info(f"학습 지표 차트 저장: {metrics_path}")
    
    plt.close()
    
    # 학습 지표 JSON 저장
    json_path = os.path.join(results_dir, f'{args.symbol}_training_metrics_{timestamp}.json')
    
    with open(json_path, 'w') as f:
        json.dump({
            'train_rewards': train_rewards,
            'eval_rewards': eval_rewards,
            'train_metrics': train_metrics,
            'args': vars(args)
        }, f, indent=2)
    
    logger.info(f"학습 지표 데이터 저장: {json_path}")

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 랜덤 시드 설정
    set_random_seeds(args.seed)
    
    # 로깅 설정
    log_file = setup_logging(args)
    
    # 디렉토리 설정
    checkpoint_dir, results_dir = setup_directories(args)
    
    try:
        # 데이터 준비
        train_data, test_data = prepare_data(args)
        
        # 학습 환경 생성
        train_env = create_env(train_data, args)
        
        # 에이전트 생성
        agent = create_agent(train_env, args, checkpoint_dir)
        
        # 에이전트 학습
        train_rewards, eval_rewards, train_metrics = train_agent(train_env, agent, args)
        
        # 학습 지표 시각화
        plot_training_metrics(train_rewards, eval_rewards, train_metrics, args, results_dir)
        
        # 테스트 환경 생성
        test_env = create_env(test_data, args)
        
        # 백테스트 수행
        backtest_metrics = backtest_agent(test_env, agent, args, results_dir)
        
        # 최종 성공 메시지
        logger.info("학습 및 백테스트 완료!")
        logger.info(f"결과 디렉토리: {results_dir}")
        logger.info(f"로그 파일: {log_file}")
        
        return 0
    
    except Exception as e:
        logger.exception(f"오류 발생: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
