import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    트랜스포머 모델을 위한 위치 인코딩
    시퀀스 데이터에서 위치 정보를 추가하기 위한 컴포넌트
    """
    def __init__(
        self, 
        d_model: int, 
        dropout: float = 0.1, 
        max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
            
        Returns:
            위치 인코딩이 적용된 입력
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    시계열 인코더 모듈
    
    시계열 금융 데이터에서 시간적 패턴을 추출하는 트랜스포머 기반 인코더
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 50
    ):
        super(TemporalEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 특성 임베딩
        self.feature_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 시간 특성 집계 (어텐션 기반)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 시계열 특성 [batch_size, seq_length, feature_dim]
            mask: 패딩 마스크 (선택적)
            
        Returns:
            인코딩된 시계열 특성 [batch_size, hidden_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # 특성 임베딩
        x = self.feature_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # 위치 인코딩 적용
        x = self.positional_encoding(x)
        
        # 트랜스포머 인코더 적용
        if mask is not None:
            # mask: [batch_size, seq_length] -> [seq_length, batch_size]
            mask = mask.transpose(0, 1)
            encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            encoded = self.transformer_encoder(x)
        
        # 어텐션 기반 풀링
        attn_weights = self.attention_pool(encoded)  # [batch_size, seq_length, 1]
        context = torch.bmm(attn_weights.transpose(1, 2), encoded).squeeze(1)  # [batch_size, hidden_dim]
        
        # 최종 출력
        output = self.output_layer(context)
        
        return output


class FeatureAttention(nn.Module):
    """
    특성 어텐션 모듈
    
    다양한 기술적 지표, 거시경제 지표, 뉴스 감성 등의 특성 간
    중요도를 동적으로 계산하는 어텐션 메커니즘
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(FeatureAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 특성 변환 레이어
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 멀티헤드 셀프 어텐션
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 특성 중요도 계산 레이어
        self.importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 특성 집계 레이어
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 특성 [batch_size, feature_dim]
            
        Returns:
            encoded_features: 인코딩된 특성 [batch_size, hidden_dim]
            importance_weights: 특성별 중요도 [batch_size, feature_dim]
        """
        batch_size, feature_dim = x.shape
        
        # 특성을 개별 차원으로 변환
        # [batch_size, feature_dim] -> [batch_size, feature_dim, 1]
        x_expanded = x.unsqueeze(-1)
        
        # 특성 임베딩
        # [batch_size, feature_dim, 1] -> [batch_size, feature_dim, hidden_dim]
        feature_embeddings = self.transform(x_expanded)
        
        # 셀프 어텐션 적용
        attn_output, attn_weights = self.self_attention(
            query=feature_embeddings,
            key=feature_embeddings,
            value=feature_embeddings
        )
        
        # 특성 중요도 계산
        importance_weights = self.importance(attn_output).squeeze(-1)  # [batch_size, feature_dim]
        
        # 중요도 적용된 특성 집계
        weighted_embeddings = attn_output * importance_weights.unsqueeze(-1)
        aggregated = weighted_embeddings.sum(dim=1)  # [batch_size, hidden_dim]
        
        # 최종 특성 표현
        encoded_features = self.aggregate(aggregated)
        
        return encoded_features, importance_weights


class DeepSeekPolicyNetwork(nn.Module):
    """
    DeepSeek 기반 정책 네트워크
    
    트랜스포머 아키텍처를 활용한 정책 네트워크로, 시장 상태에 따른 행동 확률 생성
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(DeepSeekPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 상태 임베딩
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 트랜스포머 기반 디코더 레이어
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # 행동 프로젝션 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 초기 쿼리 토큰
        self.register_buffer('query_token', torch.zeros(1, 1, hidden_dim))
    
    def forward(
        self, 
        state: torch.Tensor, 
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            state: 상태 [batch_size, state_dim]
            memory: 메모리 (시계열 인코더 출력) [batch_size, seq_length, hidden_dim]
            
        Returns:
            action_probs: 행동 확률 [batch_size, action_dim]
        """
        batch_size = state.size(0)
        
        # 상태 임베딩
        state_embedded = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 초기 쿼리 토큰 확장
        query = self.query_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        
        # 메모리가 없으면 상태 임베딩 사용
        if memory is None:
            memory = state_embedded
        
        # 트랜스포머 디코더 적용
        decoded = self.transformer_decoder(
            tgt=query, 
            memory=memory
        )  # [batch_size, 1, hidden_dim]
        
        # 행동 로짓 계산
        action_logits = self.action_head(decoded.squeeze(1))  # [batch_size, action_dim]
        
        # 확률 변환
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs


class DistributionalValueNetwork(nn.Module):
    """
    분포형 가치 네트워크
    
    상태-행동 가치의 분포를 예측하여 불확실성을 명시적으로 모델링
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        super(DistributionalValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 원자 간격 및 지원 계산
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        # 상태-행동 임베딩
        self.state_action_embedding = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 가치 분포 헤드
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atoms)
        )
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: 상태 [batch_size, state_dim]
            action_onehot: 원-핫 인코딩된 행동 [batch_size, action_dim]
            
        Returns:
            value_dist: 가치 분포 [batch_size, num_atoms]
            expected_value: 기대 가치 [batch_size, 1]
        """
        # 상태-행동 쌍 결합
        sa_pair = torch.cat([state, action_onehot], dim=-1)
        
        # 임베딩
        embedded = self.state_action_embedding(sa_pair)
        
        # 가치 분포 계산
        logits = self.value_head(embedded)
        value_dist = F.softmax(logits, dim=-1)
        
        # 기대 가치 계산
        device = state.device
        support = self.support.to(device)
        expected_value = torch.sum(value_dist * support.unsqueeze(0), dim=1, keepdim=True)
        
        return value_dist, expected_value


class MetaController(nn.Module):
    """
    메타 컨트롤러
    
    시장 레짐 감지 및 학습 하이퍼파라미터 조정을 위한 모듈
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_regimes: int = 4
    ):
        super(MetaController, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # 상태 인코딩
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 시장 레짐 감지기
        self.regime_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_regimes)
        )
        
        # 파라미터 조정 네트워크
        self.parameter_adapter = nn.Sequential(
            nn.Linear(hidden_dim + num_regimes, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # [reward_scale, penalty_scale, exploration_temp]
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: 상태 [batch_size, state_dim]
            
        Returns:
            regime_probs: 시장 레짐 확률 [batch_size, num_regimes]
            reward_scale: 보상 스케일 [batch_size]
            penalty_scale: 페널티 스케일 [batch_size]
            exploration_temp: 탐색 온도 [batch_size]
        """
        # 상태 인코딩
        encoded = self.state_encoder(state)  # [batch_size, hidden_dim]
        
        # 시장 레짐 확률 계산
        regime_logits = self.regime_detector(encoded)  # [batch_size, num_regimes]
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # 상태와 레짐 확률 결합
        combined = torch.cat([encoded, regime_probs], dim=-1)
        
        # 파라미터 조정값 계산
        params = self.parameter_adapter(combined)  # [batch_size, 3]
        
        # 개별 파라미터 추출 및 제약 적용
        reward_scale = torch.sigmoid(params[:, 0]) * 2.0  # [0, 2] 범위
        penalty_scale = torch.sigmoid(params[:, 1]) * 1.0  # [0, 1] 범위
        exploration_temp = torch.sigmoid(params[:, 2]) * 5.0  # [0, 5] 범위
        
        return regime_probs, reward_scale, penalty_scale, exploration_temp


class DeepSeekGRPOAgent:
    """
    DeepSeek-R1 기반 GRPO 에이전트
    
    금융 시장 트레이딩을 위한 강화학습 에이전트로, 트랜스포머 기반 아키텍처를
    활용하여 시장 패턴 인식 및 의사결정 수행
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_length: int = 20,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        kl_coef: float = 0.01,
        entropy_coef: float = 0.01,
        clip_epsilon: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            seq_length: 시퀀스 길이
            feature_dim: 특성 차원 (기본값: state_dim)
            hidden_dim: 히든 레이어 차원
            num_atoms: 분포형 가치 네트워크 원자 수
            v_min: 가치 분포 최소값
            v_max: 가치 분포 최대값
            lr: 학습률
            gamma: 할인 계수
            kl_coef: KL 발산 계수
            entropy_coef: 엔트로피 계수
            clip_epsilon: PPO 클리핑 파라미터
            device: 학습 장치 (GPU/CPU)
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.feature_dim = feature_dim if feature_dim is not None else state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.gamma = gamma
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.checkpoint_dir = checkpoint_dir
        
        # 네트워크 컴포넌트 초기화
        self.temporal_encoder = TemporalEncoder(
            feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            max_seq_length=seq_length
        ).to(device)
        
        self.feature_attention = FeatureAttention(
            feature_dim=state_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        self.policy_network = DeepSeekPolicyNetwork(
            state_dim=hidden_dim * 2,  # 시간 특성 + 어텐션 특성
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        self.value_network = DistributionalValueNetwork(
            state_dim=hidden_dim * 2,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(device)
        
        self.meta_controller = MetaController(
            state_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_regimes=4
        ).to(device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self._get_all_parameters(), lr=lr)
        
        # 체크포인트 디렉토리 생성
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 경험 버퍼 초기화
        self.reset_buffers()
        
        # 학습 지표 추적
        self.train_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_div': [],
            'entropy': [],
            'total_loss': [],
            'reward_scale': [],
            'penalty_scale': [],
            'exploration_temp': []
        }
    
    def _get_all_parameters(self):
        """모든 네트워크 파라미터 결합"""
        params = list(self.temporal_encoder.parameters()) + \
                 list(self.feature_attention.parameters()) + \
                 list(self.policy_network.parameters()) + \
                 list(self.value_network.parameters()) + \
                 list(self.meta_controller.parameters())
        return params
    
    def reset_buffers(self):
        """경험 버퍼 초기화"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.histories = []
        self.log_probs = []
        self.old_action_probs = None
    
    def _process_history(self, history):
        """히스토리 데이터 처리"""
        if history is None:
            # 히스토리가 없는 경우 더미 시퀀스 생성
            return torch.zeros(1, self.seq_length, self.feature_dim).to(self.device)
        
        # 히스토리 텐서 변환
        if isinstance(history, torch.Tensor):
            return history.to(self.device)
        else:
            return torch.FloatTensor(history).to(self.device)
    
    def _process_state(self, state):
        """상태 데이터 처리"""
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        else:
            return torch.FloatTensor(state).to(self.device)
    
    def select_action(
        self, 
        state, 
        history=None, 
        deterministic=False, 
        exploration_scale=1.0
    ):
        """
        정책에 따라 행동 선택
        
        Args:
            state: 현재 상태
            history: 과거 상태 시퀀스 (선택적)
            deterministic: 결정론적 선택 여부
            exploration_scale: 탐색 스케일 (0-1)
            
        Returns:
            action: 선택된 행동
            action_info: 행동 관련 추가 정보
        """
        with torch.no_grad():
            # 상태 및 히스토리 텐서 변환
            state_tensor = self._process_state(state).unsqueeze(0)
            history_tensor = self._process_history(history)
            
            # 추론
            action_probs, regime_probs, _, _, exploration_temp = self._forward(state_tensor, history_tensor)
            
            # 행동 선택
            if deterministic:
                # 결정론적 행동 (최대 확률)
                action = torch.argmax(action_probs, dim=-1)
            else:
                # 확률적 행동 (탐색 온도 적용)
                temp = float(exploration_temp.item()) * exploration_scale
                if temp < 0.1:
                    temp = 0.1  # 최소 온도 제한
                
                # 온도 적용 확률 계산
                if temp != 1.0:
                    # 온도 스케일링
                    scaled_probs = action_probs ** (1.0 / temp)
                    # 재정규화
                    scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)
                else:
                    scaled_probs = action_probs
                
                # 분포에서 샘플링
                action_dist = torch.distributions.Categorical(scaled_probs)
                action = action_dist.sample()
                
                # 로그 확률 저장
                log_prob = action_dist.log_prob(action)
                self.log_probs.append(log_prob)
            
            # 현재 정책 확률 저장
            self.old_action_probs = action_probs
            
            # 추가 정보 구성
            action_info = {
                'action_probs': action_probs.cpu().numpy(),
                'regime_probs': regime_probs.cpu().numpy(),
                'exploration_temp': exploration_temp.item()
            }
            
        return action.item(), action_info
    
    def _forward(self, state, history, return_values=False):
        """
        네트워크 순전파
        
        Args:
            state: 현재 상태 [batch_size, state_dim]
            history: 과거 상태 시퀀스 [batch_size, seq_length, feature_dim]
            return_values: 가치 추정치 반환 여부
            
        Returns:
            action_probs: 행동 확률
            regime_probs: 시장 레짐 확률
            reward_scale: 보상 스케일
            penalty_scale: 페널티 스케일
            exploration_temp: 탐색 온도
            (선택적) value_dist: 가치 분포
            (선택적) expected_value: 기대 가치
        """
        # 시계열 인코딩
        temporal_features = self.temporal_encoder(history)
        
        # 특성 어텐션
        features, importance_weights = self.feature_attention(state)
        
        # 특성 결합
        combined_features = torch.cat([temporal_features, features], dim=-1)
        
        # 메타 컨트롤러로 파라미터 조정
        regime_probs, reward_scale, penalty_scale, exploration_temp = \
            self.meta_controller(combined_features)
        
        # 행동 확률 계산
        action_probs = self.policy_network(combined_features, memory=temporal_features.unsqueeze(1))
        
        if return_values:
            # 가치 계산 (모든 행동에 대해)
            values = []
            for a in range(self.action_dim):
                action_onehot = torch.zeros(state.size(0), self.action_dim, device=self.device)
                action_onehot[:, a] = 1.0
                value_dist, expected_value = self.value_network(combined_features, action_onehot)
                values.append((value_dist, expected_value))
            
            return action_probs, regime_probs, reward_scale, penalty_scale, exploration_temp, values
        else:
            return action_probs, regime_probs, reward_scale, penalty_scale, exploration_temp
    
    def store_transition(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done, 
        history=None
    ):
        """
        경험 저장
        
        Args:
            state: 현재 상태
            action: 선택된 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
            history: 과거 상태 시퀀스 (선택적)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if history is not None:
            self.histories.append(history)
    
    def update(self, epochs=3, batch_size=64):
        """
        GRPO 알고리즘으로 정책 업데이트
        
        Args:
            epochs: 훈련 에포크 수
            batch_size: 배치 크기
            
        Returns:
            metrics: 학습 지표 사전
        """
        if len(self.states) == 0:
            return {}
        
        # 데이터 준비
        all_states = torch.FloatTensor(np.array(self.states)).to(self.device)
        all_actions = torch.LongTensor(self.actions).to(self.device)
        all_rewards = torch.FloatTensor(self.rewards).to(self.device)
        all_next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        all_dones = torch.FloatTensor(self.dones).to(self.device)
        
        # 히스토리 준비
        if len(self.histories) > 0:
            all_histories = torch.FloatTensor(np.array(self.histories)).to(self.device)
        else:
            all_histories = torch.zeros(len(all_states), self.seq_length, self.feature_dim).to(self.device)
        
        # 행동 원-핫 인코딩
        all_actions_onehot = F.one_hot(all_actions, num_classes=self.action_dim).float()
        
        # 모든 로그 확률 합치기
        if len(self.log_probs) > 0 and len(self.log_probs) == len(self.states):
            all_log_probs = torch.cat(self.log_probs).to(self.device)
        else:
            # 로그 확률이 없으면 현재 정책에서 계산
            with torch.no_grad():
                tmp_probs = []
                for i in range(0, len(all_states), batch_size):
                    batch_states = all_states[i:i+batch_size]
                    batch_histories = all_histories[i:i+batch_size]
                    batch_actions = all_actions[i:i+batch_size]
                    
                    # 행동 확률 계산
                    action_probs, _, _, _, _ = self._forward(batch_states, batch_histories)
                    
                    # 행동의 로그 확률 계산
                    dist = torch.distributions.Categorical(action_probs)
                    log_probs = dist.log_prob(batch_actions)
                    tmp_probs.append(log_probs)
                
                all_log_probs = torch.cat(tmp_probs)
        
        # 이점(advantage) 계산
        with torch.no_grad():
            advantages = []
            values = []
            
            for i in range(0, len(all_states), batch_size):
                batch_states = all_states[i:i+batch_size]
                batch_histories = all_histories[i:i+batch_size]
                batch_actions = all_actions[i:i+batch_size]
                batch_actions_onehot = all_actions_onehot[i:i+batch_size]
                
                # 현재 가치 추정
                _, _, _, _, _, batch_values = self._forward(
                    batch_states, batch_histories, return_values=True
                )
                
                # 선택한 행동의 가치
                action_indices = batch_actions.long()
                batch_action_values = torch.cat([
                    v[1] for v in [batch_values[a.item()] for a in action_indices]
                ])
                values.append(batch_action_values)
            
            all_values = torch.cat(values)
            
            # 타겟 가치 및 이점 계산
            advantages = []
            returns = []
            
            # GAE(Generalized Advantage Estimation) 계산
            gae = 0
            for i in reversed(range(len(all_rewards))):
                if i == len(all_rewards) - 1:
                    # 마지막 스텝의 경우
                    nextval = 0 if all_dones[i] else all_values[i].item()
                else:
                    nextval = all_values[i+1].item()
                
                delta = all_rewards[i] + self.gamma * nextval - all_values[i].item()
                gae = delta + self.gamma * 0.95 * (1.0 - all_dones[i]) * gae
                returns.insert(0, gae + all_values[i].item())
                advantages.insert(0, gae)
            
            all_advantages = torch.FloatTensor(advantages).to(self.device)
            all_returns = torch.FloatTensor(returns).to(self.device)
            
            # 이점 정규화
            if len(all_advantages) > 1:
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-9)
        
        # 몇 가지 초기 지표 계산
        regime_metrics = []
        for i in range(0, len(all_states), batch_size):
            batch_states = all_states[i:i+batch_size]
            batch_histories = all_histories[i:i+batch_size]
            
            # 메타 컨트롤러 출력
            _, regime_probs, reward_scale, penalty_scale, exploration_temp = \
                self._forward(batch_states, batch_histories)
            
            batch_metrics = {
                'regime_probs': regime_probs.detach().cpu().numpy().mean(axis=0),
                'reward_scale': reward_scale.detach().cpu().numpy().mean(),
                'penalty_scale': penalty_scale.detach().cpu().numpy().mean(),
                'exploration_temp': exploration_temp.detach().cpu().numpy().mean()
            }
            regime_metrics.append(batch_metrics)
        
        # 학습 준비
        total_loss_avg = 0
        policy_loss_avg = 0
        value_loss_avg = 0
        kl_div_avg = 0
        entropy_avg = 0
        
        # 에포크별 학습
        for epoch in range(epochs):
            # 배치 단위 학습
            for i in range(0, len(all_states), batch_size):
                batch_states = all_states[i:i+batch_size]
                batch_histories = all_histories[i:i+batch_size]
                batch_actions = all_actions[i:i+batch_size]
                batch_actions_onehot = all_actions_onehot[i:i+batch_size]
                batch_log_probs = all_log_probs[i:i+batch_size]
                batch_advantages = all_advantages[i:i+batch_size]
                batch_returns = all_returns[i:i+batch_size]
                
                # 순전파
                batch_action_probs, _, batch_reward_scale, batch_penalty_scale, _, batch_values = \
                    self._forward(batch_states, batch_histories, return_values=True)
                
                # 현재 행동의 로그 확률 계산
                dist = torch.distributions.Categorical(batch_action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # 정책 비율 계산 (중요도 가중치)
                ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # PPO 클리핑
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # 이점 부호에 따른 손실 분리
                positive_mask = batch_advantages > 0
                negative_mask = ~positive_mask
                
                # 보상-패널티 분리 손실
                policy_loss_pos = -torch.min(surr1, surr2)[positive_mask].mean() if positive_mask.any() else 0
                policy_loss_neg = -torch.min(surr1, surr2)[negative_mask].mean() if negative_mask.any() else 0
                
                # 스케일링 적용
                policy_loss_pos *= batch_reward_scale.mean()
                policy_loss_neg *= batch_penalty_scale.mean()
                
                # 정책 손실 결합
                policy_loss = policy_loss_pos + policy_loss_neg
                
                # 선택한 행동의 가치 분포 및 기대값
                action_indices = batch_actions.long()
                value_dists = [batch_values[a.item()][0] for a in action_indices]
                expected_values = [batch_values[a.item()][1] for a in action_indices]
                
                batch_value_dists = torch.cat(value_dists, dim=0)
                batch_expected_values = torch.cat(expected_values, dim=0)
                
                # 가치 손실 (Huber 손실)
                value_loss = F.smooth_l1_loss(batch_expected_values, batch_returns.unsqueeze(1))
                
                # KL 발산 계산 (현재 정책과 이전 정책 사이)
                if self.old_action_probs is not None:
                    old_probs = self.old_action_probs[i:i+batch_size] if i < len(self.old_action_probs) else None
                    if old_probs is not None and old_probs.size(0) == batch_action_probs.size(0):
                        kl_div = (old_probs * torch.log(old_probs / (batch_action_probs + 1e-8) + 1e-8)).sum(dim=1).mean()
                    else:
                        kl_div = torch.tensor(0.0).to(self.device)
                else:
                    kl_div = torch.tensor(0.0).to(self.device)
                
                # 엔트로피 계산
                entropy = dist.entropy().mean()
                
                # 총 손실
                total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + self.kl_coef * kl_div
                
                # 옵티마이저 스텝
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._get_all_parameters(), 0.5)  # 그라디언트 클리핑
                self.optimizer.step()
                
                # 지표 기록
                total_loss_avg += total_loss.item()
                policy_loss_avg += policy_loss.item()
                value_loss_avg += value_loss.item()
                kl_div_avg += kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div
                entropy_avg += entropy.item()
        
        # 평균 계산
        n_batches = (len(all_states) + batch_size - 1) // batch_size * epochs
        total_loss_avg /= n_batches
        policy_loss_avg /= n_batches
        value_loss_avg /= n_batches
        kl_div_avg /= n_batches
        entropy_avg /= n_batches
        
        # 평균 레짐 지표
        avg_regime_probs = np.mean([m['regime_probs'] for m in regime_metrics], axis=0)
        avg_reward_scale = np.mean([m['reward_scale'] for m in regime_metrics])
        avg_penalty_scale = np.mean([m['penalty_scale'] for m in regime_metrics])
        avg_exploration_temp = np.mean([m['exploration_temp'] for m in regime_metrics])
        
        # 학습 지표 추적
        for k, v in {
            'policy_loss': policy_loss_avg,
            'value_loss': value_loss_avg,
            'kl_div': kl_div_avg,
            'entropy': entropy_avg,
            'total_loss': total_loss_avg,
            'reward_scale': avg_reward_scale,
            'penalty_scale': avg_penalty_scale,
            'exploration_temp': avg_exploration_temp
        }.items():
            self.train_metrics[k].append(v)
        
        # 버퍼 초기화
        self.reset_buffers()
        
        return {
            'policy_loss': policy_loss_avg,
            'value_loss': value_loss_avg,
            'kl_div': kl_div_avg,
            'entropy': entropy_avg,
            'total_loss': total_loss_avg,
            'regime_probs': avg_regime_probs.tolist(),
            'reward_scale': avg_reward_scale,
            'penalty_scale': avg_penalty_scale,
            'exploration_temp': avg_exploration_temp
        }
    
    def save(self, path=None):
        """
        모델 저장
        
        Args:
            path: 저장 경로 (None이면 체크포인트 디렉토리에 저장)
        """
        if path is None and self.checkpoint_dir:
            path = os.path.join(self.checkpoint_dir, "deepseek_grpo_model.pt")
        
        torch.save({
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'feature_attention': self.feature_attention.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'meta_controller': self.meta_controller.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_metrics': self.train_metrics,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'feature_dim': self.feature_dim,
            'seq_length': self.seq_length,
            'hidden_dim': self.hidden_dim
        }, path)
        
        logger.info(f"모델 저장 완료: {path}")
    
    def load(self, path):
        """
        모델 로드
        
        Args:
            path: 로드 경로
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])
        self.feature_attention.load_state_dict(checkpoint['feature_attention'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.meta_controller.load_state_dict(checkpoint['meta_controller'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        
        logger.info(f"모델 로드 완료: {path}")
    
    def get_feature_importance(self, state, history=None):
        """
        특성 중요도 추출
        
        Args:
            state: 현재 상태
            history: 과거 상태 시퀀스 (선택적)
            
        Returns:
            importance_weights: 특성별 중요도
        """
        with torch.no_grad():
            # 상태 및 히스토리 텐서 변환
            state_tensor = self._process_state(state).unsqueeze(0)
            history_tensor = self._process_history(history)
            
            # 특성 어텐션 적용
            _, importance_weights = self.feature_attention(state_tensor)
            
            return importance_weights.cpu().numpy()
