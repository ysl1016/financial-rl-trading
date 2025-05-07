import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union


class HybridTemporalBlock(nn.Module):
    """
    트랜스포머와 LSTM/GRU를 결합한 하이브리드 시간적 블록
    
    시계열 데이터에서 다양한 시간적 패턴을 효과적으로 포착하기 위한 설계
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        use_bidirectional: bool = True,
        transformer_first: bool = True,
        layer_norm_eps: float = 1e-5,
        use_gated_connection: bool = True
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            hidden_dim: 히든 레이어 차원
            num_heads: 트랜스포머 어텐션 헤드 수
            dropout: 드롭아웃 비율
            rnn_type: RNN 유형 ('lstm' 또는 'gru')
            use_bidirectional: 양방향 RNN 사용 여부
            transformer_first: 트랜스포머를 먼저 적용할지 여부
            layer_norm_eps: 레이어 정규화 epsilon
            use_gated_connection: 게이트된 스킵 연결 사용 여부
        """
        super(HybridTemporalBlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_first = transformer_first
        self.use_gated_connection = use_gated_connection
        
        # 트랜스포머 레이어
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps
        )
        
        # RNN 레이어
        num_directions = 2 if use_bidirectional else 1
        self.rnn_output_dim = hidden_dim * num_directions
        
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=0  # 단일 레이어라 드롭아웃 필요 없음
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=0  # 단일 레이어라 드롭아웃 필요 없음
            )
        
        # 출력 프로젝션 (RNN 출력 차원을 원래 차원으로 변환)
        self.output_projection = nn.Linear(self.rnn_output_dim, input_dim)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        
        # 게이트된 연결 (학습 가능한 가중치로 두 출력을 결합)
        if use_gated_connection:
            self.gate = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid()
            )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        transformer_mask: Optional[torch.Tensor] = None,
        rnn_hidden: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 입력 시퀀스 [batch_size, seq_length, input_dim]
            transformer_mask: 트랜스포머 어텐션 마스크 (선택적)
            rnn_hidden: RNN 초기 은닉 상태 (선택적)
            
        Returns:
            output: 처리된 시퀀스 [batch_size, seq_length, input_dim]
            hidden: RNN 최종 은닉 상태 (LSTM인 경우 (h, c) 튜플)
        """
        if self.transformer_first:
            # 트랜스포머 -> RNN 순서
            transformer_out = self.transformer_layer(x, src_mask=transformer_mask)
            transformer_out = self.norm1(transformer_out)
            
            rnn_out, hidden = self.rnn(transformer_out, rnn_hidden)
            rnn_out = self.output_projection(rnn_out)
            rnn_out = self.dropout(rnn_out)
            rnn_out = self.norm2(rnn_out)
            
            first_out, second_out = transformer_out, rnn_out
        else:
            # RNN -> 트랜스포머 순서
            rnn_out, hidden = self.rnn(x, rnn_hidden)
            rnn_out = self.output_projection(rnn_out)
            rnn_out = self.dropout(rnn_out)
            rnn_out = self.norm1(rnn_out)
            
            transformer_out = self.transformer_layer(rnn_out, src_mask=transformer_mask)
            transformer_out = self.norm2(transformer_out)
            
            first_out, second_out = rnn_out, transformer_out
        
        # 두 출력 결합
        if self.use_gated_connection:
            # 게이트된 연결
            gate_input = torch.cat([first_out, second_out], dim=-1)
            gate = self.gate(gate_input)
            output = gate * first_out + (1 - gate) * second_out
        else:
            # 단순 잔차 연결 (residual connection)
            output = first_out + second_out
        
        return output, hidden


class DeepSeekHybridModel(nn.Module):
    """
    DeepSeek-R1 기반 하이브리드 금융 트레이딩 모델
    
    트랜스포머와 순환 신경망을 결합하여 다양한 시간적 패턴을 포착
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hybrid_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        use_bidirectional: bool = True,
        max_seq_length: int = 100,
        use_temporal_fusion: bool = True
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            action_dim: 행동 공간 차원
            hidden_dim: 히든 레이어 차원
            num_hybrid_layers: 하이브리드 블록 수
            num_heads: 트랜스포머 어텐션 헤드 수
            dropout: 드롭아웃 비율
            rnn_type: RNN 유형 ('lstm' 또는 'gru')
            use_bidirectional: 양방향 RNN 사용 여부
            max_seq_length: 최대 시퀀스 길이
            use_temporal_fusion: 다중 시간 스케일 퓨전 사용 여부
        """
        super(DeepSeekHybridModel, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.max_seq_length = max_seq_length
        self.use_temporal_fusion = use_temporal_fusion
        
        # 입력 임베딩
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # 하이브리드 블록 스택
        self.hybrid_blocks = nn.ModuleList()
        for i in range(num_hybrid_layers):
            # 레이어마다 번갈아가며 트랜스포머/RNN 우선 순서 변경
            transformer_first = (i % 2 == 0)
            
            block = HybridTemporalBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,  # 양방향이면 실제로는 hidden_dim 크기
                num_heads=num_heads,
                dropout=dropout,
                rnn_type=rnn_type,
                use_bidirectional=use_bidirectional,
                transformer_first=transformer_first
            )
            self.hybrid_blocks.append(block)
        
        # 다중 시간 스케일 퓨전
        if use_temporal_fusion:
            self.temporal_fusion = TemporalFusionDecoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
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
        
        # 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
            dict: 모델 출력 (정책 로짓, 가치 추정 등)
        """
        batch_size, seq_length, _ = x.shape
        
        # 입력 임베딩
        x = self.input_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # 위치 인코딩 적용
        x = self.positional_encoding(x)  # [batch_size, seq_length, hidden_dim]
        
        # 하이브리드 블록 통과
        hidden_states = None
        temporal_features = []
        
        for i, block in enumerate(self.hybrid_blocks):
            # 중간 특성 저장 (다중 시간 스케일 퓨전용)
            if i > 0:
                temporal_features.append(x)
            
            # 하이브리드 블록 통과
            x, hidden_states = block(x, transformer_mask=mask, rnn_hidden=hidden_states)
        
        # 마지막 특성 추가
        temporal_features.append(x)
        
        # 다중 시간 스케일 퓨전
        if self.use_temporal_fusion and len(temporal_features) > 1:
            x = self.temporal_fusion(temporal_features)
        
        # 마지막 시퀀스 토큰 사용 (현재 상태)
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # 정책 및 가치 헤드
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return {
            "policy_logits": policy_logits,
            "value": value,
            "features": x
        }


class PositionalEncoding(nn.Module):
    """
    트랜스포머를 위한 위치 인코딩
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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
            x: [batch_size, seq_length, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalFusionDecoder(nn.Module):
    """
    다중 시간 스케일 퓨전 디코더
    
    다양한 레이어에서 포착된 시간적 패턴을 통합
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            hidden_dim: 히든 레이어 차원
            num_heads: 크로스 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(TemporalFusionDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 크로스 어텐션 층
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 게이트된 퓨전 네트워크
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            temporal_features: 다양한 레이어의 특성 리스트 [batch_size, seq_length, hidden_dim]
            
        Returns:
            fused_features: 통합된 특성 [batch_size, seq_length, hidden_dim]
        """
        # 마지막 레이어 특성을 쿼리로 사용
        query = self.norm1(temporal_features[-1])
        batch_size, seq_length, _ = query.shape
        
        # 이전 레이어 특성 스택을 키/값으로 사용
        past_features = torch.cat([f.unsqueeze(1) for f in temporal_features[:-1]], dim=1)
        past_features = self.norm1(past_features)
        
        # 시간적 특성 스택 형태 변환: [batch_size, num_layers, seq_length, hidden_dim] -> [batch_size*seq_length, num_layers, hidden_dim]
        num_layers = past_features.size(1)
        past_features_reshaped = past_features.transpose(1, 2).reshape(batch_size * seq_length, num_layers, self.hidden_dim)
        query_reshaped = query.reshape(batch_size * seq_length, 1, self.hidden_dim)
        
        # 크로스 어텐션으로 다양한 시간 스케일 정보 통합
        context, _ = self.cross_attention(
            query=query_reshaped,
            key=past_features_reshaped,
            value=past_features_reshaped
        )
        
        # 원래 형태로 복원: [batch_size*seq_length, 1, hidden_dim] -> [batch_size, seq_length, hidden_dim]
        context = context.reshape(batch_size, seq_length, self.hidden_dim)
        
        # 게이트된 퓨전
        combined = torch.cat([query, context], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)
        
        fused = gate * transform + (1 - gate) * query
        fused = self.dropout(fused)
        
        # 잔차 연결 및 정규화
        fused = self.norm2(fused + query)
        
        return fused
