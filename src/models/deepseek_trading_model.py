import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class DeepSeekTradingModel(nn.Module):
    """
    DeepSeek-R1 based financial trading model
    
    Uses transformer architecture to process time series financial data,
    optimized for the GRPO (Generalized Reward-Penalty Optimization) algorithm.
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
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of multihead attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            use_gru_overlay: Whether to use GRU overlay
            use_feature_attention: Whether to use feature attention mechanism
            use_regime_detection: Whether to use market regime detection mechanism
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
        
        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (preserving temporal information)
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Transformer encoder
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
        
        # GRU overlay (enhancing temporal dependencies)
        if use_gru_overlay:
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Feature attention mechanism
        if use_feature_attention:
            self.feature_attention = FeatureAttention(
                feature_dim=hidden_dim,
                num_heads=num_heads // 2  # Reduced number of heads
            )
        
        # Market regime detector
        if use_regime_detection:
            self.regime_detector = RegimeDetector(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_regimes=4
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
        
        # Uncertainty head (state value variance estimation)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive values
        )
    
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
            dict: Model outputs (policy logits, value estimation, uncertainty estimation, etc.)
        """
        batch_size, seq_length, _ = x.shape
        
        # Feature embedding
        x = self.feature_embedding(x)  # [batch_size, seq_length, hidden_dim]
        
        # Apply positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_length, hidden_dim]
        
        # Transformer encoder
        x_transformer = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # GRU overlay (optional)
        if self.use_gru_overlay:
            x_gru, _ = self.gru(x_transformer)
            x = x_gru + x_transformer  # Residual connection
        else:
            x = x_transformer
        
        # Use last sequence token (current state)
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Apply feature attention (optional)
        attention_weights = None
        if self.use_feature_attention:
            x, attention_weights = self.feature_attention(x)
        
        # Market regime detection (optional)
        regime_probs = None
        if self.use_regime_detection:
            regime_probs = self.regime_detector(x)
        
        # Policy and value heads
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
        """Convert policy logits to probabilities"""
        return F.softmax(policy_logits, dim=-1)
    
    def get_action(
        self,
        policy_logits: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Select action (probabilistic or deterministic)"""
        if deterministic:
            return torch.argmax(policy_logits, dim=-1)
        
        # Temperature scaling
        scaled_logits = policy_logits / temperature
        action_probs = F.softmax(scaled_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        return action_dist.sample()

# Required auxiliary modules
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
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
    """Feature attention mechanism to compute importance between features"""
    
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
            weighted_features: Features with applied weights
            attention_weights: Feature importance weights
        """
        # Expand dimensions [batch_size, 1, feature_dim]
        x_expanded = x.unsqueeze(1)
        
        # Self-attention
        attn_output, _ = self.multihead_attn(x_expanded, x_expanded, x_expanded)
        attn_output = attn_output.squeeze(1)  # [batch_size, feature_dim]
        
        # Calculate feature importance
        attention_weights = self.feature_importance(attn_output)
        
        # Apply weights
        weighted_features = x * attention_weights
        
        return weighted_features, attention_weights

class RegimeDetector(nn.Module):
    """Market regime (state) detector module"""
    
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
            regime_probs: Probabilities for each regime [batch_size, num_regimes]
        """
        return self.detector(x)
