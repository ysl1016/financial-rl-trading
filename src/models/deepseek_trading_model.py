import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class DeepSeekTradingModel(nn.Module):
    """
    DeepSeek-R1 기반 금융 트레이딩 모델
    
    트랜스포머 아키텍처를 사용하여 시계열 금융 데이터를 처리하고,
    GRPO(Generalized Reward-Penalty Optimization) 알고리즘에 최적화된 구조입니다.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        use_gru_overlay: bool = True,
        use_feature_attention: bool = True,
        use_regime_detection: bool = True,
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            action_dim: 행동 공간 차원
            hidden_dim: 히든 레이어 차원
            num_layers: 트랜스포머 레이어 수
            num_heads: 멀티헤드 어텐션 헤드 수
            dropout: 드롭아웃 비율
            max_seq_length: 최대 시퀀스 길이
            use_gru_overlay: GRU 오버레이 사용 여부
            use_feature_attention: 특성 어텐션 메커니즘 사용 여부
            use_regime_detection: 시장 레짐 감지 메커니즘 사용 여부
        """
        super(DeepSeekTradingModel, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.use_gru_overlay = use_gru_overlay
        self.use_feature_attention = use_feature_attention
        self.use_regime_detection = use_regime_detection
        
        # 임베딩 레이어
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 위치 인코딩 (시간적 정보 보존)
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # 트랜스포머 인코더
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
        
        # GRU 오버레이 (시간적 의존성 강화)
        if use_gru_overlay:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # 특성 어텐션 메커니즘
        if use_feature_attention:
            self.feature_attention = FeatureAttention(
                feature_dim=hidden_dim,
                num_heads=num_heads // 2  # 헤드 수 감소
            )
        
        # 시장 레짐 감지기
        if use_regime_detection:
            self.regime_detector = RegimeDetector(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_regimes=4
            )
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 가치 헤드 (상태 가치 추정)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 불확실성 헤드 (상태 가치의 분산 추정)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 항상 양수 보장
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 입력 시계열 데이터 [batch_size, seq_length, input_dim]
            mask: 어텐션 마스크 (선택 사항)
            
        Returns:
            dict: 모델 출력 (정책 로짓, 가치 추정, 불확실성 추정 등)
        """
        batch_size, seq_length, _ = x.shape
        
        # 특성 임베딩
        x = self.feature_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # 위치 인코딩 적용
        x = self.positional_encoding(x)  # [batch_size, seq_length, hidden_dim]
        
        # 트랜스포머 인코더 통과
        x_transformer = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # GRU 오버레이 (선택적)
        if self.use_gru_overlay:
            x_gru, _ = self.gru(x_transformer)
            x = x_gru + x_transformer  # 잔차 연결
        else:
            x = x_transformer
        
        # 마지막 시퀀스 토큰 사용 (현재 상태)
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # 특성 어텐션 적용 (선택적)
        attention_weights = None
        if self.use_feature_attention:
            x, attention_weights = self.feature_attention(x)
        
        # 시장 레짐 감지 (선택적)
        regime_probs = None
        if self.use_regime_detection:
            regime_probs = self.regime_detector(x)
        
        # 정책 및 가치 헤드
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        uncertainty = self.uncertainty_head(x)
        
        return {
            "policy_logits": policy_logits,
            "value": value,
            "uncertainty": uncertainty,
            "attention_weights": attention_weights,
            "regime_probs": regime_probs,
            "features": x
        }
    
    def get_action_probs(self, policy_logits: torch.Tensor) -> torch.Tensor:
        """정책 로짓을 확률로 변환"""
        return F.softmax(policy_logits, dim=-1)
    
    def get_action(
        self,
        policy_logits: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """행동 선택 (확률적 또는 결정론적)"""
        if deterministic:
            return torch.argmax(policy_logits, dim=-1)
        
        # 온도 스케일링
        scaled_logits = policy_logits / temperature
        action_probs = F.softmax(scaled_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        return action_dist.sample()

# 필요한 보조 모듈들
class PositionalEncoding(nn.Module):
    """트랜스포머를 위한 위치 인코딩"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, embedding_dim]
        """
        if x.dim() == 3:
            # [batch_size, seq_length, embedding_dim]
            x = x + self.pe[:x.size(1), :].transpose(0, 1)
        else:
            # [seq_length, batch_size, embedding_dim]
            x = x + self.pe[:x.size(0), :]
            
        return self.dropout(x)

class FeatureAttention(nn.Module):
    """특성 간 중요도를 계산하는 어텐션 메커니즘"""
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super(FeatureAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.feature_importance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, feature_dim]
            
        Returns:
            weighted_features: 가중치가 적용된 특성
            attention_weights: 특성별 중요도
        """
        # 차원 확장 [batch_size, 1, feature_dim]
        x_expanded = x.unsqueeze(1)
        
        # 셀프 어텐션
        attn_output, _ = self.multihead_attn(x_expanded, x_expanded, x_expanded)
        attn_output = attn_output.squeeze(1)  # [batch_size, feature_dim]
        
        # 특성 중요도 계산
        attention_weights = self.feature_importance(attn_output)
        
        # 가중치 적용
        weighted_features = x * attention_weights
        
        return weighted_features, attention_weights

class RegimeDetector(nn.Module):
    """시장 레짐(체제)을 감지하는 모듈"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_regimes: int = 4):
        super(RegimeDetector, self).__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, feature_dim]
            
        Returns:
            regime_probs: 각 레짐의 확률 [batch_size, num_regimes]
        """
        return self.detector(x)
