import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union


class HybridTemporalBlock(nn.Module):
    """
    Hybrid temporal block combining transformer and LSTM/GRU
    
    Designed to effectively capture various temporal patterns in financial time series data
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
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Transformer attention heads
            dropout: Dropout rate
            rnn_type: RNN type ('lstm' or 'gru')
            use_bidirectional: Whether to use bidirectional RNN
            transformer_first: Whether to apply transformer first
            layer_norm_eps: Layer normalization epsilon
            use_gated_connection: Whether to use gated skip connection
        """
        super(HybridTemporalBlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_first = transformer_first
        self.use_gated_connection = use_gated_connection
        
        # Transformer layer
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
        
        # RNN layer
        num_directions = 2 if use_bidirectional else 1
        self.rnn_output_dim = hidden_dim * num_directions
        
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=0  # No dropout for single layer
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=0  # No dropout for single layer
            )
        
        # Output projection (convert RNN output dimension to original dimension)
        self.output_projection = nn.Linear(self.rnn_output_dim, input_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        
        # Gated connection (learnable weights to combine outputs)
        if use_gated_connection:
            self.gate = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Sigmoid()
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        transformer_mask: Optional[torch.Tensor] = None,
        rnn_hidden: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input sequence [batch_size, seq_length, input_dim]
            transformer_mask: Transformer attention mask (optional)
            rnn_hidden: RNN initial hidden state (optional)
            
        Returns:
            output: Processed sequence [batch_size, seq_length, input_dim]
            hidden: RNN final hidden state (for LSTM, this is a (h, c) tuple)
        """
        if self.transformer_first:
            # Transformer -> RNN order
            transformer_out = self.transformer_layer(x, src_mask=transformer_mask)
            transformer_out = self.norm1(transformer_out)
            
            rnn_out, hidden = self.rnn(transformer_out, rnn_hidden)
            rnn_out = self.output_projection(rnn_out)
            rnn_out = self.dropout(rnn_out)
            rnn_out = self.norm2(rnn_out)
            
            first_out, second_out = transformer_out, rnn_out
        else:
            # RNN -> Transformer order
            rnn_out, hidden = self.rnn(x, rnn_hidden)
            rnn_out = self.output_projection(rnn_out)
            rnn_out = self.dropout(rnn_out)
            rnn_out = self.norm1(rnn_out)
            
            transformer_out = self.transformer_layer(rnn_out, src_mask=transformer_mask)
            transformer_out = self.norm2(transformer_out)
            
            first_out, second_out = rnn_out, transformer_out
        
        # Combine outputs
        if self.use_gated_connection:
            # Gated connection
            gate_input = torch.cat([first_out, second_out], dim=-1)
            gate = self.gate(gate_input)
            output = gate * first_out + (1 - gate) * second_out
        else:
            # Simple residual connection
            output = first_out + second_out
        
        return output, hidden


class DeepSeekHybridModel(nn.Module):
    """
    DeepSeek-R1 based hybrid financial trading model
    
    Combines transformer and recurrent networks to capture diverse temporal patterns
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
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            num_hybrid_layers: Number of hybrid blocks
            num_heads: Transformer attention heads
            dropout: Dropout rate
            rnn_type: RNN type ('lstm' or 'gru')
            use_bidirectional: Whether to use bidirectional RNN
            max_seq_length: Maximum sequence length
            use_temporal_fusion: Whether to use multi-timescale fusion
        """
        super(DeepSeekHybridModel, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.max_seq_length = max_seq_length
        self.use_temporal_fusion = use_temporal_fusion
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Hybrid block stack
        self.hybrid_blocks = nn.ModuleList()
        for i in range(num_hybrid_layers):
            # Alternate transformer/RNN priority
            transformer_first = (i % 2 == 0)
            
            block = HybridTemporalBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,  # Half dimension for bidirectional
                num_heads=num_heads,
                dropout=dropout,
                rnn_type=rnn_type,
                use_bidirectional=use_bidirectional,
                transformer_first=transformer_first
            )
            self.hybrid_blocks.append(block)
        
        # Multi-timescale fusion
        if use_temporal_fusion:
            self.temporal_fusion = TemporalFusionDecoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization"""
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
            x: Input time series data [batch_size, seq_length, input_dim]
            mask: Attention mask (optional)
            
        Returns:
            dict: Model outputs (policy logits, value estimation, etc.)
        """
        batch_size, seq_length, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # Apply positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_length, hidden_dim]
        
        # Process through hybrid blocks
        hidden_states = None
        temporal_features = []
        
        for i, block in enumerate(self.hybrid_blocks):
            # Store intermediate features (for multi-timescale fusion)
            if i > 0:
                temporal_features.append(x)
            
            # Process through hybrid block
            x, hidden_states = block(x, transformer_mask=mask, rnn_hidden=hidden_states)
        
        # Add final features
        temporal_features.append(x)
        
        # Multi-timescale fusion
        if self.use_temporal_fusion and len(temporal_features) > 1:
            x = self.temporal_fusion(temporal_features)
        
        # Use last sequence token (current state)
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return {
            "policy_logits": policy_logits,
            "value": value,
            "features": x
        }


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
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
    Multi-timescale fusion decoder
    
    Integrates temporal patterns captured at different layers
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
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Cross-attention heads
            dropout: Dropout rate
        """
        super(TemporalFusionDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated fusion network
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            temporal_features: List of features from different layers [batch_size, seq_length, hidden_dim]
            
        Returns:
            fused_features: Integrated features [batch_size, seq_length, hidden_dim]
        """
        # Use last layer features as query
        query = self.norm1(temporal_features[-1])
        batch_size, seq_length, _ = query.shape
        
        # Stack previous layer features as key/value
        past_features = torch.cat([f.unsqueeze(1) for f in temporal_features[:-1]], dim=1)
        past_features = self.norm1(past_features)
        
        # Reshape temporal feature stack: [batch_size, num_layers, seq_length, hidden_dim] -> [batch_size*seq_length, num_layers, hidden_dim]
        num_layers = past_features.size(1)
        past_features_reshaped = past_features.transpose(1, 2).reshape(batch_size * seq_length, num_layers, self.hidden_dim)
        query_reshaped = query.reshape(batch_size * seq_length, 1, self.hidden_dim)
        
        # Cross-attention to integrate different timescale information
        context, _ = self.cross_attention(
            query=query_reshaped,
            key=past_features_reshaped,
            value=past_features_reshaped
        )
        
        # Restore shape: [batch_size*seq_length, 1, hidden_dim] -> [batch_size, seq_length, hidden_dim]
        context = context.reshape(batch_size, seq_length, self.hidden_dim)
        
        # Gated fusion
        combined = torch.cat([query, context], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)
        
        fused = gate * transform + (1 - gate) * query
        fused = self.dropout(fused)
        
        # Residual connection and normalization
        fused = self.norm2(fused + query)
        
        return fused
