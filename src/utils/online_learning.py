import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional, Callable
from collections import deque
import joblib
import os
import time
from sklearn.cluster import KMeans


class MarketRegimeDetector:
    """
    시장 레짐(상태) 감지 모듈
    
    특징:
    - 다양한 시장 지표 분석을 통한 레짐 감지
    - 군집화 알고리즘을 통한 시장 상태 분류
    - 레짐 전환 감지 및 알림
    """
    
    def __init__(
        self,
        n_regimes: int = 4,  # 감지할 시장 레짐 수
        lookback_period: int = 60,  # 레짐 감지에 사용할 룩백 기간
        regime_features: List[str] = None,  # 레짐 감지에 사용할 특성
        min_samples_per_regime: int = 20,  # 레짐 감지에 필요한 최소 표본 수
        stability_threshold: float = 0.6,  # 레짐 안정성 임계값 (0-1)
        transition_smoothing: int = 5,  # 레짐 전환 평활화 기간
    ):
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.regime_features = regime_features or [
            'returns_mean', 'returns_std', 'volume_change',
            'rsi', 'macd_signal', 'bb_width', 'atr_normalized'
        ]
        self.min_samples_per_regime = min_samples_per_regime
        self.stability_threshold = stability_threshold
        self.transition_smoothing = transition_smoothing
        
        # 군집화 모델
        self.regime_model = KMeans(n_clusters=n_regimes, random_state=42)
        
        # 레짐 히스토리
        self.regime_history = []
        self.regime_probabilities = []
        
        # 레짐 특성 데이터
        self.feature_history = []
        
        # 현재 레짐 및 안정성
        self.current_regime = None
        self.regime_stability = 0.0
        
        # 레짐별 특성
        self.regime_profiles = {i: {} for i in range(n_regimes)}
        self.fitted = False
    
    def extract_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """시장 데이터에서 레짐 감지에 사용할 특성 추출"""
        features = pd.DataFrame(index=market_data.index)
        
        # 기본 수익률 및 변동성 특성
        close_prices = market_data['Close']
        returns = close_prices.pct_change().fillna(0)
        
        # 윈도우 기반 특성
        window_size = min(20, len(returns) // 4)
        features['returns_mean'] = returns.rolling(window=window_size).mean().fillna(0)
        features['returns_std'] = returns.rolling(window=window_size).std().fillna(0)
        
        # 거래량 변화
        if 'Volume' in market_data.columns:
            volume = market_data['Volume']
            features['volume_change'] = volume.pct_change().rolling(window=window_size).mean().fillna(0)
        else:
            features['volume_change'] = 0
        
        # 기술적 지표가 이미 계산되어 있는 경우 사용
        if 'RSI_norm' in market_data.columns:
            features['rsi'] = market_data['RSI_norm']
        else:
            features['rsi'] = 0
            
        if 'MACDSignal_norm' in market_data.columns:
            features['macd_signal'] = market_data['MACDSignal_norm']
        else:
            features['macd_signal'] = 0
            
        if 'BBWidth_norm' in market_data.columns:
            features['bb_width'] = market_data['BBWidth_norm']
        else:
            features['bb_width'] = 0
            
        if 'ATR_norm' in market_data.columns:
            features['atr_normalized'] = market_data['ATR_norm']
        else:
            features['atr_normalized'] = 0
        
        # 사용할 특성만 선택
        selected_features = features[self.regime_features].copy()
        
        # NaN 값 처리
        selected_features.fillna(0, inplace=True)
        
        return selected_features
    
    def update(self, market_data: pd.DataFrame) -> int:
        """
        새로운 시장 데이터로 레짐 감지기 업데이트
        
        Args:
            market_data: 시장 데이터 DataFrame (최소한 'Close' 열 필요)
            
        Returns:
            int: 감지된 현재 시장 레짐
        """
        # 레짐 특성 추출
        features = self.extract_regime_features(market_data)
        self.feature_history.append(features.iloc[-1].values)
        
        # 특성 히스토리 유지
        if len(self.feature_history) > self.lookback_period:
            self.feature_history = self.feature_history[-self.lookback_period:]
        
        # 충분한 데이터가 있는 경우에만 모델 학습/예측
        if len(self.feature_history) >= self.min_samples_per_regime:
            feature_array = np.array(self.feature_history)
            
            # 모델이 학습되지 않았으면 학습
            if not self.fitted:
                self.regime_model.fit(feature_array)
                self.fitted = True
                
                # 레짐별 프로파일 초기화
                labels = self.regime_model.labels_
                for regime in range(self.n_regimes):
                    regime_samples = feature_array[labels == regime]
                    if len(regime_samples) > 0:
                        self.regime_profiles[regime] = {
                            'center': self.regime_model.cluster_centers_[regime],
                            'volatility': np.mean(regime_samples[:, 1]),  # returns_std 특성
                            'samples': len(regime_samples)
                        }
            
            # 현재 레짐 예측
            current_features = feature_array[-1].reshape(1, -1)
            regime_label = self.regime_model.predict(current_features)[0]
            
            # 거리 기반 확률 계산
            distances = np.zeros(self.n_regimes)
            for i in range(self.n_regimes):
                distances[i] = np.linalg.norm(current_features - self.regime_model.cluster_centers_[i])
            
            # 거리의 역수를 확률로 변환
            inv_distances = 1.0 / (distances + 1e-9)
            probabilities = inv_distances / np.sum(inv_distances)
            
            # 레짐 히스토리 및 확률 업데이트
            self.regime_history.append(regime_label)
            self.regime_probabilities.append(probabilities)
            
            # 히스토리 유지
            if len(self.regime_history) > self.transition_smoothing:
                self.regime_history = self.regime_history[-self.transition_smoothing:]
                self.regime_probabilities = self.regime_probabilities[-self.transition_smoothing:]
            
            # 평활화된 레짐 결정 (최근 N일의 최빈값)
            if len(self.regime_history) >= self.transition_smoothing:
                # 각 레짐의 평균 확률
                avg_probabilities = np.mean(self.regime_probabilities, axis=0)
                smoothed_regime = np.argmax(avg_probabilities)
                
                # 레짐 안정성 계산 (현재 레짐의 확률)
                self.regime_stability = avg_probabilities[smoothed_regime]
                self.current_regime = smoothed_regime
            else:
                self.current_regime = regime_label
                self.regime_stability = probabilities[regime_label]
        
        return self.current_regime if self.current_regime is not None else 0
    
    def get_regime_info(self) -> Dict:
        """현재 레짐에 대한 정보 반환"""
        if self.current_regime is None:
            return {
                'regime': 0,
                'stability': 0.0,
                'profile': {},
                'probabilities': np.ones(self.n_regimes) / self.n_regimes
            }
        
        return {
            'regime': self.current_regime,
            'stability': self.regime_stability,
            'profile': self.regime_profiles.get(self.current_regime, {}),
            'probabilities': self.regime_probabilities[-1] if self.regime_probabilities else np.zeros(self.n_regimes)
        }
    
    def is_transition_detected(self, threshold: float = 0.3) -> bool:
        """레짐 전환이 감지되었는지 확인"""
        if len(self.regime_history) < self.transition_smoothing:
            return False
        
        # 최근 N일 중 현재 레짐과 다른 레짐의 비율
        diff_regime_ratio = sum(1 for r in self.regime_history[:-1] 
                                if r != self.current_regime) / (len(self.regime_history) - 1)
        
        return diff_regime_ratio > threshold
    
    def save(self, filepath: str):
        """모델과 상태 저장"""
        save_data = {
            'model': self.regime_model,
            'fitted': self.fitted,
            'regime_profiles': self.regime_profiles,
            'feature_history': self.feature_history,
            'regime_history': self.regime_history,
            'regime_probabilities': self.regime_probabilities,
            'current_regime': self.current_regime,
            'regime_stability': self.regime_stability,
            'config': {
                'n_regimes': self.n_regimes,
                'lookback_period': self.lookback_period,
                'regime_features': self.regime_features,
                'min_samples_per_regime': self.min_samples_per_regime,
                'stability_threshold': self.stability_threshold,
                'transition_smoothing': self.transition_smoothing
            }
        }
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'MarketRegimeDetector':
        """저장된 모델과 상태 로드"""
        load_data = joblib.load(filepath)
        
        config = load_data['config']
        detector = cls(
            n_regimes=config['n_regimes'],
            lookback_period=config['lookback_period'],
            regime_features=config['regime_features'],
            min_samples_per_regime=config['min_samples_per_regime'],
            stability_threshold=config['stability_threshold'],
            transition_smoothing=config['transition_smoothing']
        )
        
        detector.regime_model = load_data['model']
        detector.fitted = load_data['fitted']
        detector.regime_profiles = load_data['regime_profiles']
        detector.feature_history = load_data['feature_history']
        detector.regime_history = load_data['regime_history']
        detector.regime_probabilities = load_data['regime_probabilities']
        detector.current_regime = load_data['current_regime']
        detector.regime_stability = load_data['regime_stability']
        
        return detector


class ExperienceBuffer:
    """
    온라인 학습을 위한 경험 버퍼
    
    특징:
    - 우선순위 기반 샘플링
    - 다양한 레짐의 경험 저장
    - 버퍼 크기 제한 및 오래된 경험 제거
    """
    
    def __init__(
        self, 
        capacity: int = 10000,
        alpha: float = 0.6,  # 우선순위 지수
        beta: float = 0.4,   # 중요도 샘플링 초기 베타
        beta_increment: float = 0.001,  # 베타 증가율
        regime_balancing: bool = True,  # 레짐 균형 유지 여부
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.regime_balancing = regime_balancing
        
        # 버퍼 초기화
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.regimes = deque(maxlen=capacity)
        
        # 통계 정보
        self.insertion_count = 0
        self.regime_counts = {}
    
    def add(self, experience: Tuple, priority: float = None, regime: int = 0):
        """
        경험 추가
        
        Args:
            experience: (state, action, reward, next_state, done) 튜플
            priority: 경험의 우선순위 (None이면 최대 우선순위로 설정)
            regime: 경험이 발생한 시장 레짐
        """
        # 우선순위가 없으면 현재 최대 우선순위 사용
        if priority is None:
            priority = max(self.priorities, default=1.0)
        
        # 우선순위 지수 적용
        priority = (priority + 1e-5) ** self.alpha
        
        # 경험 추가
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.regimes.append(regime)
        
        # 레짐 카운트 업데이트
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1
        
        # 삽입 횟수 증가
        self.insertion_count += 1
    
    def sample(self, batch_size: int, current_regime: int = None) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        경험 배치 샘플링
        
        Args:
            batch_size: 샘플링할 배치 크기
            current_regime: 현재 시장 레짐 (레짐 균형 유지 시 사용)
            
        Returns:
            experiences: 샘플링된 경험 리스트
            indices: 샘플링된 경험의 인덱스
            weights: 중요도 샘플링 가중치
        """
        # 버퍼가 비어있거나 배치 크기보다 작은 경우
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 레짐 균형 샘플링
        if self.regime_balancing and current_regime is not None:
            # 현재 레짐과 다른 레짐의 샘플 비율 (70% 현재 레짐, 30% 다른 레짐)
            current_regime_ratio = 0.7
            current_regime_size = int(batch_size * current_regime_ratio)
            other_regime_size = batch_size - current_regime_size
            
            # 현재 레짐 및 다른 레짐의 인덱스 리스트
            current_regime_indices = [i for i, r in enumerate(self.regimes) if r == current_regime]
            other_regime_indices = [i for i, r in enumerate(self.regimes) if r != current_regime]
            
            # 각 레짐에서의 우선순위
            if current_regime_indices:
                current_regime_priorities = np.array([self.priorities[i] for i in current_regime_indices])
                current_regime_probs = current_regime_priorities / current_regime_priorities.sum()
                current_indices = np.random.choice(
                    current_regime_indices,
                    size=min(current_regime_size, len(current_regime_indices)),
                    replace=False,
                    p=current_regime_probs
                )
            else:
                current_indices = []
            
            if other_regime_indices:
                other_regime_priorities = np.array([self.priorities[i] for i in other_regime_indices])
                other_regime_probs = other_regime_priorities / other_regime_priorities.sum()
                other_indices = np.random.choice(
                    other_regime_indices,
                    size=min(other_regime_size, len(other_regime_indices)),
                    replace=False,
                    p=other_regime_probs
                )
            else:
                other_indices = []
            
            # 인덱스 결합
            indices = np.concatenate([current_indices, other_indices])
            
            # 배치 크기 조정
            if len(indices) < batch_size:
                # 추가 샘플링 (모든 레짐에서)
                all_priorities = np.array(self.priorities)
                all_probs = all_priorities / all_priorities.sum()
                
                additional_indices = np.random.choice(
                    len(self.buffer),
                    size=batch_size - len(indices),
                    replace=False,
                    p=all_probs
                )
                
                indices = np.concatenate([indices, additional_indices])
        else:
            # 우선순위 기반 샘플링
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        
        # 중요도 샘플링 가중치 계산
        total = len(self.buffer)
        weights = (total * np.array([probs[i] for i in indices])) ** -self.beta
        weights /= weights.max()  # 정규화
        
        # 베타 값 업데이트
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 경험 가져오기
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """경험의 우선순위 업데이트"""
        for i, priority in zip(indices, priorities):
            if i < len(self.priorities):
                self.priorities[i] = (priority + 1e-5) ** self.alpha
    
    def get_stats(self) -> Dict:
        """버퍼 통계 정보 반환"""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'insertion_count': self.insertion_count,
            'regime_counts': self.regime_counts.copy(),
            'mean_priority': np.mean(self.priorities) if self.priorities else 0.0
        }


class OnlineLearningAgent:
    """
    온라인 학습이 가능한 강화학습 에이전트
    
    특징:
    - 경험 우선순위 기반 리플레이
    - 시장 레짐 감지 및 적응
    - 안전한 탐색 메커니즘
    - 점진적 모델 업데이트
    """
    
    def __init__(
        self, 
        model,  # 강화학습 모델 (GRPO 에이전트)
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        update_frequency: int = 10,  # 몇 번의 상호작용마다 모델 업데이트할지
        min_samples_before_update: int = 200,  # 업데이트 전 최소 샘플 수
        save_frequency: int = 1000,  # 몇 번의 업데이트마다 모델 저장할지
        model_path: str = './models',  # 모델 저장 경로
        regime_detector: MarketRegimeDetector = None,  # 시장 레짐 감지기
        exploration_params: Dict = None,  # 탐색 파라미터
    ):
        self.model = model
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.min_samples_before_update = min_samples_before_update
        self.save_frequency = save_frequency
        self.model_path = model_path
        
        # 경험 버퍼
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)
        
        # 레짐 감지기
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        # 탐색 파라미터
        self.exploration_params = exploration_params or {
            'initial_epsilon': 0.5,  # 초기 입실론 (탐색 확률)
            'final_epsilon': 0.01,   # 최종 입실론
            'epsilon_decay': 0.995,  # 입실론 감소율
            'min_certainty': 0.3,    # 최소 확실성 요구사항 (레짐 변화 시 증가)
        }
        
        # 현재 탐색 상태
        self.epsilon = self.exploration_params['initial_epsilon']
        self.current_regime = 0
        self.regime_stability = 0.0
        
        # 상호작용 및 업데이트 카운터
        self.interaction_count = 0
        self.update_count = 0
        
        # 성능 지표
        self.metrics = {
            'rewards': [],
            'losses': [],
            'updates': [],
            'regime_changes': []
        }
        
        # 저장 경로 생성
        os.makedirs(self.model_path, exist_ok=True)
    
    def select_action(self, state, market_data=None, deterministic=False):
        """
        현재 상태에 기반하여 행동 선택
        
        Args:
            state: 환경 상태
            market_data: 레짐 감지에 사용할 시장 데이터
            deterministic: 결정론적 행동 선택 여부
            
        Returns:
            선택된 행동
        """
        # 시장 레짐 업데이트 (시장 데이터가 있는 경우)
        if market_data is not None:
            new_regime = self.regime_detector.update(market_data)
            regime_info = self.regime_detector.get_regime_info()
            
            # 레짐 변화 감지
            if new_regime != self.current_regime:
                self.metrics['regime_changes'].append({
                    'step': self.interaction_count,
                    'old_regime': self.current_regime,
                    'new_regime': new_regime,
                    'stability': regime_info['stability']
                })
                
                # 레짐 변화 시 탐색 증가
                self.epsilon = max(
                    self.epsilon,
                    self.exploration_params['initial_epsilon'] * regime_info['stability']
                )
            
            # 현재 레짐 및 안정성 업데이트
            self.current_regime = new_regime
            self.regime_stability = regime_info['stability']
        
        # 결정론적 행동 선택 또는 활용 단계인 경우
        if deterministic or (np.random.random() > self.epsilon):
            # 모델을 사용하여 행동 선택
            action = self.model.select_action(state, deterministic=True)
        else:
            # 탐색: 무작위 행동 또는 안전한 탐색
            if hasattr(self.model, 'action_space'):
                # 환경의 행동 공간에서 무작위 샘플링
                action = self.model.action_space.sample()
            else:
                # 기본 안전 탐색: 약한 신호로 무작위 행동
                action_dim = len(state) if hasattr(state, '__len__') else 1
                action = np.random.uniform(-0.2, 0.2, size=action_dim)
        
        # 입실론 감소
        if not deterministic:
            self.epsilon = max(
                self.exploration_params['final_epsilon'],
                self.epsilon * self.exploration_params['epsilon_decay']
            )
        
        # 상호작용 카운트 증가
        self.interaction_count += 1
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done, td_error=None):
        """경험 저장"""
        experience = (state, action, reward, next_state, done)
        
        # TD 오류가 없는 경우 보상의 절댓값으로 우선순위 설정
        priority = abs(td_error) if td_error is not None else abs(reward)
        
        # 경험 버퍼에 추가
        self.buffer.add(experience, priority=priority, regime=self.current_regime)
        
        # 보상 기록
        self.metrics['rewards'].append(reward)
        
        # 업데이트 조건 체크
        if (self.interaction_count % self.update_frequency == 0 and 
            len(self.buffer.buffer) >= self.min_samples_before_update):
            self.update_model()
    
    def update_model(self):
        """모델 업데이트"""
        # 배치 샘플링
        experiences, indices, weights = self.buffer.sample(
            self.batch_size, current_regime=self.current_regime
        )
        
        if not experiences:
            return
        
        # 모델 업데이트
        update_info = self.model.update_from_experiences(experiences, weights)
        
        # 우선순위 업데이트
        if 'td_errors' in update_info:
            self.buffer.update_priorities(indices, np.abs(update_info['td_errors']))
        
        # 손실 기록
        if 'loss' in update_info:
            self.metrics['losses'].append(update_info['loss'])
        
        # 모델 저장
        self.update_count += 1
        self.metrics['updates'].append({
            'step': self.interaction_count,
            'loss': update_info.get('loss', 0),
            'regime': self.current_regime
        })
        
        if self.update_count % self.save_frequency == 0:
            self.save_model()
    
    def save_model(self, custom_path=None):
        """모델 및 에이전트 상태 저장"""
        # 저장 경로 설정
        path = custom_path or os.path.join(
            self.model_path, 
            f"model_{int(time.time())}_{self.update_count}.pt"
        )
        
        # 모델 저장
        if hasattr(self.model, 'save'):
            self.model.save(path)
        
        # 에이전트 상태 저장
        agent_state = {
            'interaction_count': self.interaction_count,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'current_regime': self.current_regime,
            'regime_stability': self.regime_stability,
            'metrics': self.metrics,
        }
        
        # 레짐 감지기 저장
        regime_detector_path = os.path.join(
            self.model_path,
            f"regime_detector_{int(time.time())}_{self.update_count}.joblib"
        )
        self.regime_detector.save(regime_detector_path)
        
        # 에이전트 상태 저장
        agent_state_path = os.path.join(
            self.model_path,
            f"agent_state_{int(time.time())}_{self.update_count}.joblib"
        )
        joblib.dump(agent_state, agent_state_path)
        
        return path, regime_detector_path, agent_state_path
    
    @classmethod
    def load(cls, model, model_path, regime_detector_path, agent_state_path, **kwargs):
        """저장된 모델 및 에이전트 상태 로드"""
        # 모델 로드
        if hasattr(model, 'load'):
            model.load(model_path)
        
        # 레짐 감지기 로드
        regime_detector = MarketRegimeDetector.load(regime_detector_path)
        
        # 에이전트 상태 로드
        agent_state = joblib.load(agent_state_path)
        
        # 에이전트 생성
        agent = cls(model=model, regime_detector=regime_detector, **kwargs)
        
        # 상태 복원
        agent.interaction_count = agent_state['interaction_count']
        agent.update_count = agent_state['update_count']
        agent.epsilon = agent_state['epsilon']
        agent.current_regime = agent_state['current_regime']
        agent.regime_stability = agent_state['regime_stability']
        agent.metrics = agent_state['metrics']
        
        return agent
    
    def get_metrics(self):
        """성능 지표 반환"""
        # 버퍼 통계 추가
        metrics = self.metrics.copy()
        metrics['buffer_stats'] = self.buffer.get_stats()
        metrics['regime_info'] = self.regime_detector.get_regime_info()
        metrics['exploration'] = {
            'epsilon': self.epsilon,
            'interaction_count': self.interaction_count,
            'update_count': self.update_count
        }
        
        return metrics


# GRPO 에이전트와 통합하기 위한 메서드 - 실제 구현에서는 GRPO 에이전트 클래스에 추가됩니다
def update_from_experiences(agent, experiences, weights=None):
    """
    배치 경험으로부터 GRPO 에이전트 업데이트
    
    이 메서드는 GRPOAgent 클래스에 추가되어야 합니다.
    """
    if not experiences:
        return {'loss': 0.0}
    
    # 경험 배치 언패킹
    states, actions, rewards, next_states, dones = zip(*experiences)
    
    # 텐서 변환
    states = torch.FloatTensor(np.array(states)).to(agent.device)
    actions = torch.LongTensor(np.array(actions)).to(agent.device)
    rewards = torch.FloatTensor(np.array(rewards)).to(agent.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
    dones = torch.FloatTensor(np.array(dones)).to(agent.device)
    
    # 가중치 처리
    if weights is not None:
        weights = torch.FloatTensor(weights).to(agent.device)
    else:
        weights = torch.ones_like(rewards)
    
    # 행동 원핫 인코딩
    actions_onehot = torch.zeros(len(actions), agent.action_dim).to(agent.device)
    actions_onehot.scatter_(1, actions.unsqueeze(1), 1)
    
    # 정책 확률
    action_probs = agent.network(states)
    log_probs = torch.log(action_probs + 1e-10)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 현재 상태-행동 가치
    current_q = agent.network.estimate_q_value(states, actions_onehot).squeeze()
    
    # 다음 상태 최대 Q-값
    with torch.no_grad():
        next_action_probs = agent.network(next_states)
        next_q_values = []
        
        for i in range(agent.action_dim):
            next_action_onehot = torch.zeros_like(actions_onehot)
            next_action_onehot[:, i] = 1
            q = agent.network.estimate_q_value(next_states, next_action_onehot).squeeze()
            next_q_values.append(q)
        
        next_q_values = torch.stack(next_q_values, dim=1)
        next_q = (next_q_values * next_action_probs).sum(dim=1)
        
        # 타겟 Q-값
        target_q = rewards + agent.gamma * next_q * (1 - dones)
    
    # TD 오류 계산
    td_errors = target_q - current_q
    
    # 이점(advantage) 계산
    advantages = td_errors
    
    # 정책 손실 계산 (보상-패널티 분리)
    positive_mask = advantages > 0
    negative_mask = ~positive_mask
    
    policy_loss = torch.zeros(1, device=agent.device)
    
    if positive_mask.any():
        reward_loss = -selected_log_probs[positive_mask] * advantages[positive_mask] * agent.reward_scale
        policy_loss = policy_loss + (reward_loss * weights[positive_mask]).mean()
    
    if negative_mask.any():
        penalty_loss = selected_log_probs[negative_mask] * advantages[negative_mask].abs() * agent.penalty_scale
        policy_loss = policy_loss + (penalty_loss * weights[negative_mask]).mean()
    
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


# 사용 예시
if __name__ == "__main__":
    # 간단한 테스트를 위한 더미 모델 클래스
    class DummyModel:
        def __init__(self, action_dim=3):
            self.action_dim = action_dim
            self.action_space = type('obj', (object,), {
                'sample': lambda: np.random.randint(0, action_dim)
            })
            self.device = 'cpu'
        
        def select_action(self, state, deterministic=False):
            return np.random.randint(0, self.action_dim)
        
        def update_from_experiences(self, experiences, weights=None):
            # 더미 업데이트
            return {
                'loss': np.random.random(),
                'td_errors': np.random.randn(len(experiences))
            }
        
        def save(self, path):
            print(f"모델을 {path}에 저장했습니다.")
        
        def load(self, path):
            print(f"모델을 {path}에서 로드했습니다.")
    
    # 더미 모델 생성
    model = DummyModel(action_dim=3)
    
    # 온라인 학습 에이전트 생성
    agent = OnlineLearningAgent(
        model=model,
        buffer_capacity=1000,
        batch_size=32,
        update_frequency=5,
        min_samples_before_update=50,
        save_frequency=100,
        model_path='./test_models'
    )
    
    # 간단한 테스트: 몇 번의 상호작용 및 경험 저장
    for i in range(100):
        state = np.random.rand(10)  # 더미 상태
        action = agent.select_action(state)
        next_state = np.random.rand(10)  # 더미 다음 상태
        reward = np.random.randn()  # 더미 보상
        done = np.random.random() < 0.1  # 더미 종료 상태
        
        agent.store_experience(state, action, reward, next_state, done)
    
    # 지표 확인
    metrics = agent.get_metrics()
    print(f"총 상호작용: {metrics['exploration']['interaction_count']}")
    print(f"총 업데이트: {metrics['exploration']['update_count']}")
    print(f"버퍼 크기: {metrics['buffer_stats']['size']}")
    
    # 모델 저장
    model_path, regime_detector_path, agent_state_path = agent.save_model()
    
    # 새 에이전트로 로드
    new_agent = OnlineLearningAgent.load(
        model=DummyModel(action_dim=3),
        model_path=model_path,
        regime_detector_path=regime_detector_path,
        agent_state_path=agent_state_path
    )
    
    # 로드된 에이전트 지표 확인
    new_metrics = new_agent.get_metrics()
    print(f"로드된 에이전트 상호작용: {new_metrics['exploration']['interaction_count']}")
