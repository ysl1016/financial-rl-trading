"""
DeepSeek-R1 기반 GRPO(Generalized Reward-Penalty Optimization) 에이전트 구현

이 모듈은 DeepSeek-R1 아키텍처를 활용한 강화학습 에이전트를 구현합니다.
트랜스포머 기반 아키텍처와 고급 강화학습 기법을 결합하여 금융 시계열 데이터의
복잡한 패턴을 학습하고 효과적인 트레이딩 전략을 개발합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any, Union


class PositionalEncoding(nn.Module):
    """
    시계열 데이터의 순서 정보를 인코딩하는 위치 인코딩 레이어
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: 모델의 차원
            dropout: 드롭아웃 비율
            max_len: 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩 행렬 생성
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록 (파라미터로는 학습되지 않음)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        Returns:
            위치 정보가 추가된 텐서
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalPooling(nn.Module):
    """
    시계열 특성을 집계하는 어텐션 기반 풀링 레이어
    """
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: 은닉층 차원
        """
        super(TemporalPooling, self).__init__()
        self.output_dim = hidden_dim
        
        # 어텐션 가중치 계산을 위한 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [batch_size, seq_len, hidden_dim]
            
        Returns:
            집계된 텐서 [batch_size, hidden_dim]
        """
        # 어텐션 스코어 계산
        attn_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 시퀀스 차원에 대해 소프트맥스
        
        # 가중 합계 계산
        context = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        return context


class FeatureAttention(nn.Module):
    """
    기술적 지표와 같은 다양한 특성 간의 중요도를 학습하는 어텐션 모듈
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        """
        Args:
            feature_dim: 특성 차원
            hidden_dim: 은닉층 차원
        """
        super(FeatureAttention, self).__init__()
        
        # 특성 투영 레이어
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 셀프 어텐션 레이어
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 특성 중요도 예측 레이어
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 입력 특성 [batch_size, feature_dim]
            
        Returns:
            중요도가 반영된 특성, 중요도 가중치
        """
        batch_size = features.size(0)
        
        # 특성 투영
        projected = self.feature_projection(features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 셀프 어텐션 적용
        attn_output, _ = self.self_attention(projected, projected, projected)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 특성 중요도 계산
        importance_weights = self.importance_predictor(attn_output)  # [batch_size, feature_dim]
        
        # 중요도 가중치 적용
        weighted_features = features * importance_weights
        
        return weighted_features, importance_weights


class DeepSeekPolicyNetwork(nn.Module):
    """
    DeepSeek-R1 기반 정책 네트워크
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            hidden_dim: 은닉층 차원
            num_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout: 드롭아웃 비율
        """
        super(DeepSeekPolicyNetwork, self).__init__()
        
        # 상태 임베딩
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 행동 예측 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            
        Returns:
            행동 로짓 [batch_size, action_dim]
        """
        # 상태 임베딩
        x = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 트랜스포머 인코더 통과
        x = self.transformer_encoder(x)
        
        # 행동 예측
        action_logits = self.action_head(x.squeeze(1))  # [batch_size, action_dim]
        
        return action_logits


class DistributionalValueNetwork(nn.Module):
    """
    분포형 가치 네트워크 - 상태-행동 가치 분포를 모델링
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 num_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            hidden_dim: 은닉층 차원
            num_atoms: 분포 원자 수
            v_min: 최소 가치
            v_max: 최대 가치
        """
        super(DistributionalValueNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        # 상태-행동 임베딩
        self.state_action_embedding = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 가치 분포 헤드
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atoms)
        )
        
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            action_onehot: 원핫 인코딩된 행동 [batch_size, action_dim]
            
        Returns:
            가치 분포, 기대 가치
        """
        # 상태-행동 결합
        sa_pair = torch.cat([state, action_onehot], dim=-1)
        
        # 임베딩
        embedded = self.state_action_embedding(sa_pair).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 트랜스포머 통과
        transformed = self.transformer(embedded).squeeze(1)  # [batch_size, hidden_dim]
        
        # 로짓 출력
        logits = self.value_head(transformed)  # [batch_size, num_atoms]
        
        # 분포 계산
        probabilities = F.softmax(logits, dim=-1)  # [batch_size, num_atoms]
        
        # 기대 가치 계산
        support = self.support.to(probabilities.device)
        expected_value = torch.sum(probabilities * support, dim=1, keepdim=True)  # [batch_size, 1]
        
        return probabilities, expected_value


class MarketRegimeDetector(nn.Module):
    """
    시장 레짐 감지를 위한 네트워크
    """
    def __init__(self, state_dim: int, hidden_dim: int, num_regimes: int = 4):
        """
        Args:
            state_dim: 상태 차원
            hidden_dim: 은닉층 차원
            num_regimes: 시장 레짐 수
        """
        super(MarketRegimeDetector, self).__init__()
        
        # 레짐 분류 네트워크
        self.regime_classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_regimes)
        )
        
        # 하이퍼파라미터 조정 네트워크
        self.param_adapter = nn.Sequential(
            nn.Linear(state_dim + num_regimes, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [reward_scale, penalty_scale, exploration_temp]
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: 상태 텐서 [batch_size, state_dim]
            
        Returns:
            레짐 확률, 보상 스케일, 페널티 스케일, 탐색 온도
        """
        # 레짐 확률 계산
        regime_logits = self.regime_classifier(state)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # 상태와 레짐 확률 결합
        combined = torch.cat([state, regime_probs], dim=-1)
        
        # 파라미터 조정값 계산
        params = self.param_adapter(combined)
        
        # 개별 파라미터 추출 및 제약 적용
        reward_scale = torch.sigmoid(params[:, 0]) * 2.0 + 0.5  # [0.5, 2.5] 범위
        penalty_scale = torch.sigmoid(params[:, 1]) * 1.5  # [0, 1.5] 범위
        exploration_temp = torch.sigmoid(params[:, 2]) * 5.0 + 0.5  # [0.5, 5.5] 범위
        
        return regime_probs, reward_scale, penalty_scale, exploration_temp


class DeepSeekGRPONetwork(nn.Module):
    """
    DeepSeek-R1 기반 GRPO 통합 네트워크
    """
    def __init__(self, state_dim: int, action_dim: int, seq_length: int, 
                 feature_dim: int = 0, hidden_dim: int = 256):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            seq_length: 시퀀스 길이
            feature_dim: 시퀀스가 아닌 특성 차원 (기본값: state_dim)
            hidden_dim: 은닉층 차원
        """
        super(DeepSeekGRPONetwork, self).__init__()
        
        # 시퀀스가 아닌 특성 차원이 명시되지 않은 경우, state_dim 사용
        if feature_dim == 0:
            feature_dim = state_dim
        
        # 시간적 인코더 (시계열 데이터 처리)
        self.temporal_encoder = nn.Sequential(
            PositionalEncoding(hidden_dim, dropout=0.1, max_len=seq_length),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=4
            )
        )
        
        # 시계열 데이터 임베딩
        self.sequence_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 시간적 풀링
        self.temporal_pooling = TemporalPooling(hidden_dim)
        
        # 특성 어텐션
        self.feature_attention = FeatureAttention(state_dim, hidden_dim)
        
        # 정책 네트워크
        self.policy_network = DeepSeekPolicyNetwork(
            state_dim=hidden_dim * 2,  # 시간 특성 + 어텐션 특성
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # 가치 네트워크
        self.value_network = DistributionalValueNetwork(
            state_dim=hidden_dim * 2,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # 시장 레짐 감지기
        self.regime_detector = MarketRegimeDetector(
            state_dim=hidden_dim * 2,
            hidden_dim=hidden_dim
        )
    
    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        시계열 시퀀스 인코딩
        
        Args:
            sequence: 시계열 데이터 [batch_size, seq_length, feature_dim]
            
        Returns:
            인코딩된 시간 특성 [batch_size, hidden_dim]
        """
        # 시퀀스 임베딩
        embedded_seq = self.sequence_embedding(sequence)  # [batch_size, seq_length, hidden_dim]
        
        # 트랜스포머 인코더 통과
        encoded_seq = self.temporal_encoder(embedded_seq)  # [batch_size, seq_length, hidden_dim]
        
        # 시간적 풀링
        temporal_features = self.temporal_pooling(encoded_seq)  # [batch_size, hidden_dim]
        
        return temporal_features
    
    def forward(self, state: torch.Tensor, sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순방향 전파
        
        Args:
            state: 현재 상태 [batch_size, state_dim]
            sequence: 과거 시계열 데이터 [batch_size, seq_length, feature_dim] (옵션)
            
        Returns:
            출력 딕셔너리 (정책 로짓, 가치 분포 등)
        """
        # 시계열 데이터가 없는 경우 더미 특성 생성
        if sequence is None:
            temporal_features = torch.zeros(
                state.size(0), 
                self.policy_network.state_embedding.in_features // 2,
                device=state.device
            )
        else:
            # 시계열 인코딩
            temporal_features = self.encode_sequence(sequence)
            
        # 특성 어텐션
        weighted_features, importance_weights = self.feature_attention(state)
        
        # 특성 결합
        combined_features = torch.cat([temporal_features, weighted_features], dim=-1)
        
        # 정책 로짓 계산
        policy_logits = self.policy_network(combined_features)
        
        # 레짐 감지 및 파라미터 조정
        regime_probs, reward_scale, penalty_scale, exploration_temp = self.regime_detector(combined_features)
        
        # 각 행동에 대한 가치 분포 계산 (현재는 선택된 행동에 대해서만 계산)
        action_probs = F.softmax(policy_logits, dim=-1)
        selected_actions = torch.argmax(action_probs, dim=-1)
        selected_actions_onehot = F.one_hot(selected_actions, num_classes=action_probs.size(1)).float()
        value_dist, expected_value = self.value_network(combined_features, selected_actions_onehot)
        
        return {
            'policy_logits': policy_logits,
            'action_probs': action_probs,
            'value_dist': value_dist,
            'expected_value': expected_value,
            'regime_probs': regime_probs,
            'reward_scale': reward_scale,
            'penalty_scale': penalty_scale,
            'exploration_temp': exploration_temp,
            'importance_weights': importance_weights
        }


class DeepSeekGRPOAgent:
    """
    DeepSeek-R1 기반 GRPO 에이전트
    
    기존 GRPO 알고리즘에 트랜스포머 아키텍처, 분포형 가치 함수,
    KL 발산 제약, 시장 레짐 감지 등의 고급 기능을 통합한 에이전트
    """
    def __init__(self, state_dim: int, action_dim: int, seq_length: int = 20, 
                 feature_dim: int = 0, hidden_dim: int = 256, lr: float = 3e-4, 
                 gamma: float = 0.99, kl_weight: float = 0.01, entropy_weight: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            seq_length: 시계열 시퀀스 길이
            feature_dim: 시퀀스가 아닌 특성 차원 (기본값: state_dim)
            hidden_dim: 은닉층 차원
            lr: 학습률
            gamma: 할인 계수
            kl_weight: KL 발산 가중치
            entropy_weight: 엔트로피 가중치
            device: 연산 장치
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.feature_dim = feature_dim if feature_dim > 0 else state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.device = device
        
        # 네트워크 초기화
        self.network = DeepSeekGRPONetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # 최적화기
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 경험 버퍼
        self.reset_buffers()
        
        # 이전 정책 저장 (KL 발산 계산용)
        self.old_policy_logits = None
    
    def reset_buffers(self):
        """경험 버퍼 초기화"""
        self.states = []
        self.sequences = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_sequences = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, history: Optional[np.ndarray] = None, 
                     deterministic: bool = False) -> int:
        """
        상태에 따른 행동 선택
        
        Args:
            state: 현재 상태
            history: 과거 시계열 데이터 (옵션)
            deterministic: 결정론적 행동 선택 여부
            
        Returns:
            선택된 행동
        """
        with torch.no_grad():
            # 텐서 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 시퀀스 데이터 처리
            sequence_tensor = None
            if history is not None:
                sequence_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
            
            # 네트워크 순방향 전파
            outputs = self.network(state_tensor, sequence_tensor)
            
            if deterministic:
                # 결정론적 행동 (최대 확률)
                action = torch.argmax(outputs['action_probs'], dim=-1)
            else:
                # 확률적 행동 (탐색 온도 적용)
                temp = outputs['exploration_temp'].item()
                probs = outputs['action_probs'] ** (1.0 / temp)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = Categorical(probs)
                action = dist.sample()
        
        return action.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, 
                        history: Optional[np.ndarray] = None, 
                        next_history: Optional[np.ndarray] = None):
        """
        경험 저장
        
        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
            history: 현재 시점까지의 시계열 데이터 (옵션)
            next_history: 다음 시점까지의 시계열 데이터 (옵션)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if history is not None:
            self.sequences.append(history)
        
        if next_history is not None:
            self.next_sequences.append(next_history)
    
    def update(self) -> Dict[str, float]:
        """
        GRPO 알고리즘으로 네트워크 업데이트
        
        Returns:
            학습 지표를 포함하는 딕셔너리
        """
        if len(self.states) == 0:
            return {}
        
        # 데이터 준비
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # 시퀀스 데이터 처리
        sequences = None
        next_sequences = None
        
        if self.sequences and len(self.sequences) == len(self.states):
            sequences = torch.FloatTensor(np.array(self.sequences)).to(self.device)
        
        if self.next_sequences and len(self.next_sequences) == len(self.states):
            next_sequences = torch.FloatTensor(np.array(self.next_sequences)).to(self.device)
        
        # 행동 원핫 인코딩
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # 현재 상태 네트워크 순방향 전파
        outputs = self.network(states, sequences)
        
        # 정책 확률 및 로그 확률
        action_probs = outputs['action_probs']
        log_probs = torch.log(action_probs + 1e-10)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 보상 스케일 및 페널티 스케일
        reward_scale = outputs['reward_scale'].squeeze()
        penalty_scale = outputs['penalty_scale'].squeeze()
        
        # 가치 추정
        expected_value = outputs['expected_value'].squeeze()
        
        # 다음 상태 가치 추정
        with torch.no_grad():
            next_outputs = self.network(next_states, next_sequences)
            next_action_probs = next_outputs['action_probs']
            next_actions = torch.argmax(next_action_probs, dim=-1)
            next_actions_onehot = F.one_hot(next_actions, num_classes=self.action_dim).float()
            
            next_value_dist, next_expected_value = self.network.value_network(
                torch.cat([
                    next_outputs['regime_probs'],
                    next_outputs['importance_weights']
                ], dim=-1),
                next_actions_onehot
            )
            
            # 타겟 가치 계산
            target_value = rewards.unsqueeze(1) + self.gamma * next_expected_value * (1 - dones.unsqueeze(1))
        
        # 이점(advantage) 계산
        advantages = target_value - expected_value
        
        # 정책 손실 계산 (보상-페널티 분리)
        positive_mask = advantages > 0
        negative_mask = ~positive_mask
        
        policy_loss = torch.zeros(1, device=self.device)
        
        # 이익이 있는 경우 (보상 강화)
        if positive_mask.any():
            reward_loss = -selected_log_probs[positive_mask] * advantages[positive_mask] * reward_scale[positive_mask]
            policy_loss = policy_loss + reward_loss.mean()
        
        # 손실이 있는 경우 (페널티)
        if negative_mask.any():
            penalty_loss = selected_log_probs[negative_mask] * advantages[negative_mask].abs() * penalty_scale[negative_mask]
            policy_loss = policy_loss + penalty_loss.mean()
        
        # 가치 손실 (MSE)
        value_loss = F.mse_loss(expected_value, target_value.detach())
        
        # KL 발산 정규화 (현재 정책과 이전 정책 사이)
        kl_loss = torch.zeros(1, device=self.device)
        if self.old_policy_logits is not None:
            old_action_probs = F.softmax(self.old_policy_logits, dim=-1)
            kl_div = (old_action_probs * torch.log(old_action_probs / (action_probs + 1e-10) + 1e-10)).sum(dim=1)
            kl_loss = kl_div.mean()
        
        # 엔트로피 정규화
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        
        # 총 손실
        total_loss = policy_loss + 0.5 * value_loss + self.kl_weight * kl_loss - self.entropy_weight * entropy
        
        # 그라디언트 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # 그라디언트 클리핑
        self.optimizer.step()
        
        # 이전 정책 저장
        self.old_policy_logits = outputs['policy_logits'].detach()
        
        # 버퍼 초기화
        self.reset_buffers()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_loss': kl_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_reward_scale': reward_scale.mean().item(),
            'mean_penalty_scale': penalty_scale.mean().item()
        }
    
    def save(self, path: str):
        """
        모델 저장
        
        Args:
            path: 저장 경로
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'gamma': self.gamma,
            'kl_weight': self.kl_weight,
            'entropy_weight': self.entropy_weight
        }, path)
        
        print(f"모델이 저장되었습니다: {path}")
    
    def load(self, path: str):
        """
        모델 로드
        
        Args:
            path: 로드할 모델 경로
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 설정 업데이트
        self.state_dim = checkpoint.get('state_dim', self.state_dim)
        self.action_dim = checkpoint.get('action_dim', self.action_dim)
        self.seq_length = checkpoint.get('seq_length', self.seq_length)
        self.feature_dim = checkpoint.get('feature_dim', self.feature_dim)
        self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.kl_weight = checkpoint.get('kl_weight', self.kl_weight)
        self.entropy_weight = checkpoint.get('entropy_weight', self.entropy_weight)
        
        # 네트워크 재구성 (설정이 변경된 경우)
        self.network = DeepSeekGRPONetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_length=self.seq_length,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 가중치 로드
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # 최적화기 로드
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"모델을 로드했습니다: {path}")
