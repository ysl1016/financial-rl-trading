import numpy as np
import pandas as pd
import torch
import json
import os
import time
from datetime import datetime
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from src.models.trading_env import TradingEnv
from src.models.grpo_agent import GRPOAgent


class HyperparameterOptimizer:
    """
    GRPO 에이전트의 하이퍼파라미터 최적화를 위한 클래스입니다.
    Bayesian Optimization을 사용하여 최적의 하이퍼파라미터 조합을 찾습니다.
    """
    
    def __init__(self, train_data, val_data, log_dir="logs/hyperopt", 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        하이퍼파라미터 최적화기를 초기화합니다.
        
        Args:
            train_data (pd.DataFrame): 훈련 데이터
            val_data (pd.DataFrame): 검증 데이터
            log_dir (str): 로그 디렉토리 경로
            device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.train_data = train_data
        self.val_data = val_data
        self.log_dir = log_dir
        self.device = device
        
        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 하이퍼파라미터 탐색 공간 정의
        self.pbounds = {
            "learning_rate": (1e-5, 1e-3),       # 학습률
            "hidden_dim": (32, 256),             # 은닉층 차원
            "gamma": (0.9, 0.999),               # 감마(할인계수)
            "reward_scale": (0.5, 2.0),          # 보상 스케일
            "penalty_scale": (0.1, 1.0),         # 페널티 스케일
        }
        
        # 최적화 결과 저장 변수
        self.best_params = None
        self.best_val_sharpe = -np.inf
        self.best_agent = None
        
        # 최적화 진행 상황 로그
        self.results_log = []
    
    def _create_env(self, data, **env_params):
        """
        트레이딩 환경을 생성합니다.
        
        Args:
            data (pd.DataFrame): 환경에 사용할 데이터
            **env_params: 환경 파라미터
            
        Returns:
            TradingEnv: 생성된 트레이딩 환경
        """
        default_params = {
            "initial_capital": 100000,
            "trading_cost": 0.0005,
            "slippage": 0.0001,
            "risk_free_rate": 0.02,
            "max_position_size": 1.0,
            "stop_loss_pct": 0.02
        }
        
        # 기본 파라미터를 env_params로 업데이트
        env_params = {**default_params, **env_params}
        
        return TradingEnv(
            data=data,
            initial_capital=env_params["initial_capital"],
            trading_cost=env_params["trading_cost"],
            slippage=env_params["slippage"],
            risk_free_rate=env_params["risk_free_rate"],
            max_position_size=env_params["max_position_size"],
            stop_loss_pct=env_params["stop_loss_pct"]
        )
    
    def _create_agent(self, env, **agent_params):
        """
        GRPO 에이전트를 생성합니다.
        
        Args:
            env (TradingEnv): 트레이딩 환경
            **agent_params: 에이전트 파라미터
            
        Returns:
            GRPOAgent: 생성된 GRPO 에이전트
        """
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        return GRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(agent_params["hidden_dim"]),
            lr=agent_params["learning_rate"],
            gamma=agent_params["gamma"],
            reward_scale=agent_params["reward_scale"],
            penalty_scale=agent_params["penalty_scale"],
            device=self.device
        )
    
    def _train_agent(self, env, agent, num_episodes=10, update_interval=10):
        """
        에이전트를 훈련시킵니다.
        
        Args:
            env (TradingEnv): 트레이딩 환경
            agent (GRPOAgent): 훈련할 에이전트
            num_episodes (int): 훈련 에피소드 수
            update_interval (int): 정책 업데이트 간격
            
        Returns:
            float: 평균 에피소드 보상
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                # 행동 선택
                action = agent.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, _ = env.step(action)
                
                # 트랜지션 저장
                agent.store_transition(state, action, reward, next_state, done)
                
                # 일정 간격으로 정책 업데이트
                if step_count % update_interval == 0:
                    agent.update()
                
                state = next_state
                episode_reward += reward
                step_count += 1
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
    def _evaluate_agent(self, env, agent, num_episodes=5):
        """
        에이전트의 성능을 평가합니다.
        
        Args:
            env (TradingEnv): 트레이딩 환경
            agent (GRPOAgent): 평가할 에이전트
            num_episodes (int): 평가 에피소드 수
            
        Returns:
            dict: 평가 지표 (평균 보상, Sharpe 비율, 수익률, 최대 손실률)
        """
        episode_rewards = []
        sharpe_ratios = []
        returns = []
        max_drawdowns = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 결정론적으로 행동 선택
                action = agent.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # 포트폴리오 가치 시퀀스
            portfolio_values = np.array(env.portfolio_values)
            
            # 일별 수익률 계산
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # 총 수익률
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            returns.append(total_return)
            
            # Sharpe 비율 계산 (연율화)
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.sqrt(252) * (np.mean(daily_returns) - (0.02/252)) / np.std(daily_returns)
                sharpe_ratios.append(sharpe)
            
            # 최대 손실률 계산
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            max_drawdowns.append(max_drawdown)
        
        # 평균 지표 계산
        mean_reward = np.mean(episode_rewards)
        mean_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        mean_return = np.mean(returns)
        mean_max_drawdown = np.mean(max_drawdowns)
        
        return {
            "mean_reward": mean_reward,
            "mean_sharpe": mean_sharpe,
            "mean_return": mean_return,
            "mean_max_drawdown": mean_max_drawdown
        }
    
    def objective_function(self, learning_rate, hidden_dim, gamma, reward_scale, penalty_scale):
        """
        Bayesian Optimization을 위한 목적 함수.
        주어진 하이퍼파라미터로 에이전트를 훈련하고 검증 성능을 반환합니다.
        
        Args:
            learning_rate (float): 학습률
            hidden_dim (float): 은닉층 차원 (정수로 변환됨)
            gamma (float): 할인 계수
            reward_scale (float): 보상 스케일
            penalty_scale (float): 페널티 스케일
            
        Returns:
            float: 검증 데이터에서의 Sharpe 비율 (최대화 대상)
        """
        # 환경 생성
        train_env = self._create_env(self.train_data)
        val_env = self._create_env(self.val_data)
        
        # 에이전트 생성
        agent_params = {
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
            "reward_scale": reward_scale,
            "penalty_scale": penalty_scale
        }
        agent = self._create_agent(train_env, **agent_params)
        
        # 에이전트 훈련
        _ = self._train_agent(train_env, agent, num_episodes=5, update_interval=10)
        
        # 검증 데이터에서 에이전트 평가
        metrics = self._evaluate_agent(val_env, agent, num_episodes=3)
        
        # 결과 로깅
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": agent_params,
            "metrics": metrics
        }
        self.results_log.append(log_entry)
        
        # 현재까지 최고 성능인 경우 저장
        if metrics["mean_sharpe"] > self.best_val_sharpe:
            self.best_val_sharpe = metrics["mean_sharpe"]
            self.best_params = agent_params
            self.best_agent = agent
            
            # 최고 성능 모델 저장
            model_path = os.path.join(self.log_dir, "best_model.pt")
            agent.save(model_path)
        
        print(f"lr={learning_rate:.6f}, hdim={int(hidden_dim)}, gamma={gamma:.6f}, "
              f"rs={reward_scale:.3f}, ps={penalty_scale:.3f} -> "
              f"Sharpe={metrics['mean_sharpe']:.4f}, Return={metrics['mean_return']:.4f}")
        
        return metrics["mean_sharpe"]
    
    def optimize(self, n_iter=30, init_points=5, load_previous=False):
        """
        Bayesian Optimization을 실행하여 최적의 하이퍼파라미터를 찾습니다.
        
        Args:
            n_iter (int): 최적화 반복 횟수
            init_points (int): 초기 무작위 탐색 포인트 수
            load_previous (bool): 이전 최적화 로그 로드 여부
            
        Returns:
            dict: 최적 하이퍼파라미터와 성능 지표
        """
        # 최적화 시작 시간
        start_time = time.time()
        
        # 결과 로그 초기화
        self.results_log = []
        self.best_val_sharpe = -np.inf
        
        # 로그 파일 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"bayes_opt_log_{timestamp}.json")
        results_file = os.path.join(self.log_dir, f"results_log_{timestamp}.json")
        
        # Bayesian Optimization 객체 생성
        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.pbounds,
            random_state=42
        )
        
        # 로거 설정
        logger = JSONLogger(path=log_file)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        
        # 이전 로그 로드 (선택적)
        if load_previous and os.path.exists(log_file):
            load_logs(optimizer, logs=[log_file])
            print(f"Previous optimization logs loaded: {log_file}")
        
        # 최적화 실행
        print(f"Starting hyperparameter optimization with {n_iter} iterations...")
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        
        # 최적화 결과 저장
        with open(results_file, 'w') as f:
            json.dump(self.results_log, f, indent=2)
        
        # 최적화 소요 시간
        elapsed_time = time.time() - start_time
        
        # 최고 성능 파라미터 출력
        print("\nHyperparameter optimization completed!")
        print(f"Total time: {elapsed_time / 60:.2f} minutes")
        print("\nBest parameters:")
        for param, value in self.best_params.items():
            formatted_value = int(value) if param == "hidden_dim" else value
            print(f"  {param}: {formatted_value}")
        
        print(f"\nBest validation Sharpe ratio: {self.best_val_sharpe:.4f}")
        
        return {
            "best_params": self.best_params,
            "best_val_sharpe": self.best_val_sharpe,
            "best_agent": self.best_agent,
            "optimizer": optimizer,
            "log_file": log_file,
            "results_file": results_file
        }


class ValidationFramework:
    """
    하이퍼파라미터 튜닝을 위한 시간 기반 교차 검증 프레임워크
    """
    
    def __init__(self, data, validation_method="expanding", test_ratio=0.2, 
                 val_ratio=0.2, n_splits=5):
        """
        검증 프레임워크를 초기화합니다.
        
        Args:
            data (pd.DataFrame): 전체 데이터셋
            validation_method (str): 검증 방법 ('expanding', 'sliding', 'k_fold')
            test_ratio (float): 테스트 데이터 비율
            val_ratio (float): 검증 데이터 비율
            n_splits (int): 교차 검증 분할 수 (k_fold 방법에서 사용)
        """
        self.data = data
        self.validation_method = validation_method
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.n_splits = n_splits
    
    def _split_data_by_date(self, ratio):
        """
        날짜순으로 데이터를 분할합니다.
        
        Args:
            ratio (float): 분할 비율
            
        Returns:
            tuple: 첫 부분, 두 번째 부분 데이터
        """
        split_idx = int(len(self.data) * (1 - ratio))
        first_part = self.data.iloc[:split_idx]
        second_part = self.data.iloc[split_idx:]
        
        return first_part, second_part
    
    def _split_train_val_test(self):
        """
        훈련, 검증, 테스트 데이터를 분할합니다.
        
        Returns:
            tuple: (훈련 데이터, 검증 데이터, 테스트 데이터)
        """
        # 먼저 테스트 데이터 분리
        train_val_data, test_data = self._split_data_by_date(self.test_ratio)
        
        # 그 다음 훈련과 검증 데이터 분리
        val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
        train_data, val_data = self._split_data_by_date(val_ratio_adjusted)
        
        return train_data, val_data, test_data
    
    def get_expanding_window_splits(self):
        """
        확장 윈도우 방식의 교차 검증 분할을 반환합니다.
        
        Returns:
            list: (훈련 데이터, 검증 데이터) 튜플 리스트
        """
        train_data, val_data, _ = self._split_train_val_test()
        total_train_data = pd.concat([train_data, val_data])
        
        splits = []
        min_train_size = len(train_data) // self.n_splits
        
        for i in range(self.n_splits):
            end_idx = min_train_size * (i + 1)
            if i == self.n_splits - 1:
                end_idx = len(train_data)
            
            train_split = total_train_data.iloc[:end_idx]
            val_split = val_data
            
            splits.append((train_split, val_split))
        
        return splits
    
    def get_sliding_window_splits(self):
        """
        슬라이딩 윈도우 방식의 교차 검증 분할을 반환합니다.
        
        Returns:
            list: (훈련 데이터, 검증 데이터) 튜플 리스트
        """
        _, _, _ = self._split_train_val_test()  # 데이터 분할 준비
        
        window_size = len(self.data) // (self.n_splits + 1)
        val_size = int(window_size * self.val_ratio / self.test_ratio)
        
        splits = []
        
        for i in range(self.n_splits):
            train_start = i * window_size
            train_end = train_start + window_size
            val_start = train_end
            val_end = val_start + val_size
            
            if val_end > len(self.data):
                break
            
            train_split = self.data.iloc[train_start:train_end]
            val_split = self.data.iloc[val_start:val_end]
            
            splits.append((train_split, val_split))
        
        return splits
    
    def get_k_fold_splits(self):
        """
        K-Fold 방식의 교차 검증 분할을 반환합니다.
        
        Returns:
            list: (훈련 데이터, 검증 데이터) 튜플 리스트
        """
        train_val_data, _, _ = self._split_train_val_test()
        
        fold_size = len(train_val_data) // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size
            
            val_split = train_val_data.iloc[val_start:val_end]
            train_split = pd.concat([
                train_val_data.iloc[:val_start],
                train_val_data.iloc[val_end:]
            ])
            
            splits.append((train_split, val_split))
        
        return splits
    
    def get_splits(self):
        """
        선택한 검증 방법에 따른 교차 검증 분할을 반환합니다.
        
        Returns:
            list: (훈련 데이터, 검증 데이터) 튜플 리스트
        """
        if self.validation_method == "expanding":
            return self.get_expanding_window_splits()
        elif self.validation_method == "sliding":
            return self.get_sliding_window_splits()
        elif self.validation_method == "k_fold":
            return self.get_k_fold_splits()
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")
    
    def get_test_data(self):
        """
        테스트 데이터를 반환합니다.
        
        Returns:
            pd.DataFrame: 테스트 데이터
        """
        _, _, test_data = self._split_train_val_test()
        return test_data


class SensitivityAnalysis:
    """
    하이퍼파라미터 민감도 분석 도구
    """
    
    def __init__(self, train_data, val_data, base_params, log_dir="logs/sensitivity",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        민감도 분석 도구를 초기화합니다.
        
        Args:
            train_data (pd.DataFrame): 훈련 데이터
            val_data (pd.DataFrame): 검증 데이터
            base_params (dict): 기준 하이퍼파라미터
            log_dir (str): 로그 디렉토리 경로
            device (str): 훈련 디바이스 ('cuda' 또는 'cpu')
        """
        self.train_data = train_data
        self.val_data = val_data
        self.base_params = base_params
        self.log_dir = log_dir
        self.device = device
        
        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 최적화기 생성 (도구 재사용)
        self.optimizer = HyperparameterOptimizer(
            train_data=train_data,
            val_data=val_data,
            log_dir=log_dir,
            device=device
        )
    
    def analyze_parameter(self, param_name, values):
        """
        단일 하이퍼파라미터의 민감도를 분석합니다.
        
        Args:
            param_name (str): 분석할 파라미터 이름
            values (list): 분석할 파라미터 값 리스트
            
        Returns:
            dict: 파라미터 값별 성능 지표
        """
        results = []
        
        for value in values:
            # 기본 파라미터 복사 및 현재 값으로 업데이트
            params = self.base_params.copy()
            params[param_name] = value
            
            # 환경 생성
            train_env = self.optimizer._create_env(self.train_data)
            val_env = self.optimizer._create_env(self.val_data)
            
            # 에이전트 생성
            agent = self.optimizer._create_agent(train_env, **params)
            
            # 훈련 및 평가
            _ = self.optimizer._train_agent(train_env, agent, num_episodes=5)
            metrics = self.optimizer._evaluate_agent(val_env, agent, num_episodes=3)
            
            # 결과 저장
            results.append({
                "param_value": value,
                "metrics": metrics
            })
            
            print(f"{param_name}={value} -> Sharpe={metrics['mean_sharpe']:.4f}, "
                  f"Return={metrics['mean_return']:.4f}")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.log_dir, f"sensitivity_{param_name}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_full_analysis(self, param_ranges=None):
        """
        모든 주요 하이퍼파라미터에 대한 민감도 분석을 실행합니다.
        
        Args:
            param_ranges (dict): 하이퍼파라미터별 분석 범위
            
        Returns:
            dict: 파라미터별 민감도 분석 결과
        """
        # 기본 분석 범위 설정
        default_ranges = {
            "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            "hidden_dim": [32, 64, 128, 256],
            "gamma": [0.9, 0.95, 0.97, 0.99],
            "reward_scale": [0.5, 1.0, 1.5, 2.0],
            "penalty_scale": [0.1, 0.3, 0.5, 0.7, 1.0]
        }
        
        # 사용자 지정 범위 적용
        if param_ranges is None:
            param_ranges = default_ranges
        else:
            for key, default_value in default_ranges.items():
                if key not in param_ranges:
                    param_ranges[key] = default_value
        
        # 각 파라미터에 대한 민감도 분석
        full_results = {}
        
        for param_name, values in param_ranges.items():
            print(f"\nAnalyzing sensitivity to {param_name}...")
            results = self.analyze_parameter(param_name, values)
            full_results[param_name] = results
        
        # 종합 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"sensitivity_summary_{timestamp}.json")
        
        with open(summary_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        return full_results


def run_hyperparameter_optimization(data_path, n_iter=30, validation_method="expanding", 
                                   log_dir="logs/hyperopt"):
    """
    데이터 로드부터 하이퍼파라미터 최적화 실행까지의 전체 과정을 수행하는 헬퍼 함수입니다.
    
    Args:
        data_path (str): 데이터 파일 경로
        n_iter (int): 최적화 반복 횟수
        validation_method (str): 검증 방법
        log_dir (str): 로그 디렉토리 경로
        
    Returns:
        dict: 최적화 결과
    """
    # 데이터 로드
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    elif data_path.endswith('.pkl'):
        data = pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 검증 프레임워크 생성
    validation = ValidationFramework(
        data=data,
        validation_method=validation_method,
        test_ratio=0.2,
        val_ratio=0.2,
        n_splits=5
    )
    
    # 검증 분할 가져오기 (첫 번째 분할만 사용)
    splits = validation.get_splits()
    train_data, val_data = splits[0]
    
    # 하이퍼파라미터 최적화기 생성
    optimizer = HyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        log_dir=log_dir
    )
    
    # 최적화 실행
    result = optimizer.optimize(n_iter=n_iter, init_points=5)
    
    # 테스트 데이터로 최종 평가
    test_data = validation.get_test_data()
    test_env = optimizer._create_env(test_data)
    
    test_metrics = optimizer._evaluate_agent(
        test_env, result["best_agent"], num_episodes=10
    )
    
    print("\nTest data performance:")
    print(f"  Sharpe ratio: {test_metrics['mean_sharpe']:.4f}")
    print(f"  Return: {test_metrics['mean_return']:.4f}")
    print(f"  Max drawdown: {test_metrics['mean_max_drawdown']:.4f}")
    
    return {
        "best_params": result["best_params"],
        "best_val_sharpe": result["best_val_sharpe"],
        "test_metrics": test_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="하이퍼파라미터 최적화 도구")
    parser.add_argument("--data", type=str, required=True, help="데이터 파일 경로")
    parser.add_argument("--n_iter", type=int, default=30, help="최적화 반복 횟수")
    parser.add_argument("--validation", type=str, default="expanding", 
                       choices=["expanding", "sliding", "k_fold"], help="검증 방법")
    parser.add_argument("--log_dir", type=str, default="logs/hyperopt", help="로그 디렉토리 경로")
    
    args = parser.parse_args()
    
    result = run_hyperparameter_optimization(
        data_path=args.data,
        n_iter=args.n_iter,
        validation_method=args.validation,
        log_dir=args.log_dir
    )
