#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 기반 GRPO(Generalized Reward-Penalty Optimization) 에이전트

이 모듈은 DeepSeek-R1 아키텍처를 활용한 GRPO 에이전트의 구현을 제공합니다.
트랜스포머 기반 시계열 처리, 특성 어텐션, 분포형 가치 네트워크 등의
고급 기능을 통합하여 금융 시장 데이터의 복잡한 패턴을 효과적으로 학습합니다.
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PositionalEncoding(nn.Module):
    """
    시계열 데이터의 위치 정보를 인코딩하는 클래스

    트랜스포머 모델에서 시퀀스 내 각 위치에 대한 정보를 제공합니다.
    사인과 코사인 함수를 사용하여 다양한 주파수의 위치 정보를 생성합니다.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: 모델의 임베딩 차원
            dropout: 드롭아웃 비율
            max_len: 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록 (최적화 대상에서 제외)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        위치 인코딩 적용

        Args:
            x: 입력 텐서 [배치 크기, 시퀀스 길이, 임베딩 차원]
        
        Returns:
            위치 인코딩이 적용된 텐서
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalPooling(nn.Module):
    """시계열 특성 집계 모듈"""
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: 은닉 차원
        """
        super(TemporalPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        
        # 시간 축에 대한 어텐션
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        시간 축을 따라 특성 집계

        Args:
            x: 입력 텐서 [배치 크기, 시퀀스 길이, 은닉 차원]
        
        Returns:
            집계된 특성 [배치 크기, 은닉 차원]
        """
        # 시간 어텐션 가중치 계산
        attn_weights = self.temporal_attention(x)  # [배치, 시퀀스, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 가중 평균 계산
        weighted_sum = torch.sum(x * attn_weights, dim=1)  # [배치, 은닉]
        
        return weighted_sum


class TemporalEncoder(nn.Module):
    """
    시계열 데이터를 인코딩하는 트랜스포머 기반 모듈

    시계열 데이터의 시간적 패턴을 효과적으로 포착하는 트랜스포머 인코더입니다.
    자기 주의(self-attention) 메커니즘을 활용하여 시퀀스 내 장/단기 의존성을 모델링합니다.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 200
    ):
        """
        Args:
            feature_dim: 입력 특성 차원
            hidden_dim: 모델 은닉 차원
            num_layers: 인코더 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            max_len: 최대 시퀀스 길이
        """
        super(TemporalEncoder, self).__init__()
        
        # 특성 임베딩
        self.feature_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout, max_len)
        
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
        
        # 시간적 특성 집계
        self.temporal_pooling = TemporalPooling(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        시계열 데이터 인코딩

        Args:
            x: 입력 시계열 데이터 [배치 크기, 시퀀스 길이, 특성 차원]
        
        Returns:
            인코딩된 시계열 특성 [배치 크기, 은닉 차원]
        """
        # 특성 임베딩
        x = self.feature_embedding(x)
        
        # 위치 인코딩
        x = self.positional_encoding(x)
        
        # 트랜스포머 인코더
        x = self.transformer_encoder(x)
        
        # 시간적 특성 집계
        x = self.temporal_pooling(x)
        
        return x


class FeatureAttentionModule(nn.Module):
    """
    특성 어텐션 모듈

    다양한 기술적 지표 간의 중요도를 동적으로 가중치화하는 모듈입니다.
    멀티헤드 어텐션 메커니즘을 활용하여 다양한 특성 간의 관계를 모델링합니다.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_heads: int = 4):
        """
        Args:
            feature_dim: 입력 특성 차원
            hidden_dim: 모델 은닉 차원
            num_heads: 어텐션 헤드 수
        """
        super(FeatureAttentionModule, self).__init__()
        
        # 특성 투영 레이어
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 멀티헤드 어텐션
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 특성 중요도 가중치
        self.feature_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        특성 어텐션 적용

        Args:
            features: 입력 특성 [배치 크기, 특성 차원]
        
        Returns:
            weighted_features: 가중치가 적용된 특성 [배치 크기, 특성 차원]
            importance_weights: 특성 중요도 가중치 [배치 크기, 특성 차원]
        """
        # 특성 투영
        projected = self.feature_projection(features).unsqueeze(1)  # [배치, 1, 은닉]
        
        # 셀프 어텐션
        attn_output, _ = self.multihead_attn(projected, projected, projected)
        attn_output = attn_output.squeeze(1)  # [배치, 은닉]
        
        # 특성 중요도 계산
        importance_weights = self.feature_importance(attn_output)  # [배치, 특성]
        
        # 가중치 적용
        weighted_features = features * importance_weights
        
        return weighted_features, importance_weights


class DeepSeekPolicyNetwork(nn.Module):
    """
    DeepSeek-R1 기반 정책 네트워크

    트랜스포머 기반 정책 네트워크로, 시장 상태에 따른 행동 확률을 생성합니다.
    인코더-디코더 구조를 활용하여 복잡한 시장 패턴을 모델링합니다.
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
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            hidden_dim: 모델 은닉 차원
            num_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(DeepSeekPolicyNetwork, self).__init__()
        
        # 상태 임베딩
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # 트랜스포머 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 행동 프로젝션 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        정책 네트워크 순전파

        Args:
            state_features: 상태 특성 [배치 크기, 상태 차원]
        
        Returns:
            action_probs: 행동 확률 [배치 크기, 행동 차원]
        """
        # 상태 임베딩
        embedded_state = self.state_embedding(state_features).unsqueeze(1)  # [배치, 1, 은닉]
        
        # 트랜스포머 인코더
        transformed = self.transformer_encoder(embedded_state)
        
        # 행동 헤드
        action_logits = self.action_head(transformed.squeeze(1))  # [배치, 행동]
        
        # 행동 확률
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs


class DistributionalValueNetwork(nn.Module):
    """
    분포형 가치 네트워크

    상태-행동 가치의 불확실성을 명시적으로 모델링하는 분포형 가치 네트워크입니다.
    Q-값 대신 가치 분포를 예측하여 리스크를 더 효과적으로 포착합니다.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            hidden_dim: 모델 은닉 차원
            num_atoms: 분포를 표현할 원자 수
            v_min: 가치 분포의 최소값
            v_max: 가치 분포의 최대값
            num_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(DistributionalValueNetwork, self).__init__()
        
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        # 상태-행동 임베딩
        self.state_action_embedding = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # 트랜스포머 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 가치 분포 헤드
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atoms)
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        action_onehot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        가치 네트워크 순전파

        Args:
            state: 상태 [배치 크기, 상태 차원]
            action_onehot: 원핫 인코딩된 행동 [배치 크기, 행동 차원]
        
        Returns:
            value_dist: 가치 분포 [배치 크기, num_atoms]
            expected_value: 기대 가치 [배치 크기, 1]
        """
        # 상태-행동 쌍 결합
        sa_pair = torch.cat([state, action_onehot], dim=-1)
        
        # 임베딩
        embedded = self.state_action_embedding(sa_pair).unsqueeze(1)  # [배치, 1, 은닉]
        
        # 트랜스포머 통과
        transformed = self.transformer(embedded).squeeze(1)  # [배치, 은닉]
        
        # 가치 분포 로짓
        logits = self.value_head(transformed)  # [배치, num_atoms]
        
        # 가치 분포 확률
        value_dist = F.softmax(logits, dim=-1)
        
        # 기대 가치 계산
        expected_value = torch.sum(value_dist * self.support.unsqueeze(0), dim=1, keepdim=True)
        
        return value_dist, expected_value


class MetaController(nn.Module):
    """
    메타 컨트롤러

    시장 레짐 변화를 감지하고 전략을 동적으로 조정하는 메타 컨트롤러입니다.
    다양한 시장 조건에 따라 보상 스케일, 페널티 스케일, 탐색 온도 등을 조정합니다.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_regimes: int = 4
    ):
        """
        Args:
            state_dim: 상태 차원
            hidden_dim: 모델 은닉 차원
            num_regimes: 시장 레짐 수
        """
        super(MetaController, self).__init__()
        
        # 시장 레짐 감지기
        self.regime_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_regimes)
        )
        
        # 레짐별 파라미터 조정 네트워크
        self.parameter_adapter = nn.Sequential(
            nn.Linear(state_dim + num_regimes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  # [reward_scale, penalty_scale, exploration_temp]
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        메타 컨트롤러 순전파

        Args:
            state: 상태 [배치 크기, 상태 차원]
        
        Returns:
            regime_probs: 시장 레짐 확률 [배치 크기, num_regimes]
            reward_scale: 보상 스케일 [배치 크기]
            penalty_scale: 페널티 스케일 [배치 크기]
            exploration_temp: 탐색 온도 [배치 크기]
        """
        # 시장 레짐 로짓 계산
        regime_logits = self.regime_detector(state)  # [배치, num_regimes]
        
        # 시장 레짐 확률
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # 상태와 레짐 확률 결합
        combined = torch.cat([state, regime_probs], dim=-1)
        
        # 파라미터 조정값 계산
        params = self.parameter_adapter(combined)
        
        # 개별 파라미터 추출 및 제약 적용
        reward_scale = torch.sigmoid(params[:, 0]) * 2.0  # [0, 2] 범위
        penalty_scale = torch.sigmoid(params[:, 1]) * 1.0  # [0, 1] 범위
        exploration_temp = torch.sigmoid(params[:, 2]) * 5.0  # [0, 5] 범위
        
        return regime_probs, reward_scale, penalty_scale, exploration_temp


class DeepSeekGRPONetwork(nn.Module):
    """
    DeepSeek-R1 기반 GRPO 네트워크

    모든 컴포넌트를 통합한 DeepSeek-R1 GRPO 네트워크입니다.
    시계열 인코더, 특성 어텐션, 정책 네트워크, 가치 네트워크, 메타 컨트롤러를
    포함합니다.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_length: int,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            seq_length: 시퀀스 길이
            hidden_dim: 모델 은닉 차원
            num_transformer_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(DeepSeekGRPONetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # 네트워크 컴포넌트 초기화
        self.temporal_encoder = TemporalEncoder(
            feature_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.feature_attention = FeatureAttentionModule(
            feature_dim=state_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.policy_network = DeepSeekPolicyNetwork(
            state_dim=hidden_dim * 2,  # 시간 특성 + 어텐션 특성
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers // 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.value_network = DistributionalValueNetwork(
            state_dim=hidden_dim * 2,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers // 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.meta_controller = MetaController(
            state_dim=hidden_dim * 2,
            hidden_dim=hidden_dim
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        네트워크 순전파

        Args:
            state: 현재 상태 [배치 크기, 상태 차원]
            history: 상태 히스토리 [배치 크기, 시퀀스 길이, 상태 차원] (없으면 None)
        
        Returns:
            outputs: 출력 딕셔너리 (action_probs, value_dist, expected_value, regime_probs, 등)
        """
        batch_size = state.size(0)
        
        # 히스토리가 없는 경우 더미 생성
        if history is None:
            device = state.device
            history = torch.zeros(batch_size, self.seq_length, self.state_dim, device=device)
        
        # 시계열 인코딩
        temporal_features = self.temporal_encoder(history)
        
        # 특성 어텐션
        weighted_features, importance_weights = self.feature_attention(state)
        
        # 특성 결합
        combined_features = torch.cat([temporal_features, weighted_features], dim=-1)
        
        # 메타 컨트롤러
        regime_probs, reward_scale, penalty_scale, exploration_temp = self.meta_controller(combined_features)
        
        # 정책 네트워크
        action_probs = self.policy_network(combined_features)
        
        # 출력 딕셔너리 구성
        outputs = {
            'action_probs': action_probs,
            'regime_probs': regime_probs,
            'reward_scale': reward_scale,
            'penalty_scale': penalty_scale,
            'exploration_temp': exploration_temp,
            'importance_weights': importance_weights
        }
        
        return outputs
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        actions: torch.Tensor, 
        history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        행동 평가

        Args:
            state: 현재 상태 [배치 크기, 상태 차원]
            actions: 행동 인덱스 [배치 크기]
            history: 상태 히스토리 [배치 크기, 시퀀스 길이, 상태 차원] (없으면 None)
        
        Returns:
            outputs: 출력 딕셔너리 (action_probs, log_probs, value_dist, expected_value, entropy, 등)
        """
        # 기본 순전파
        outputs = self.forward(state, history)
        
        # 행동 확률 및 로그 확률
        action_probs = outputs['action_probs']
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # 선택된 행동의 원핫 인코딩
        action_onehot = F.one_hot(actions, self.action_dim).float()
        
        # 가치 네트워크 (상태-행동 쌍에 대한 가치 분포)
        batch_size = state.size(0)
        device = state.device
        
        # 히스토리가 없는 경우 더미 생성
        if history is None:
            history = torch.zeros(batch_size, self.seq_length, self.state_dim, device=device)
        
        # 시계열 인코딩
        temporal_features = self.temporal_encoder(history)
        
        # 특성 어텐션
        weighted_features, _ = self.feature_attention(state)
        
        # 특성 결합
        combined_features = torch.cat([temporal_features, weighted_features], dim=-1)
        
        # 가치 분포 및 기대 가치
        value_dist, expected_value = self.value_network(combined_features, action_onehot)
        
        # 추가 출력
        outputs.update({
            'log_probs': log_probs,
            'entropy': entropy,
            'value_dist': value_dist,
            'expected_value': expected_value
        })
        
        return outputs


class DeepSeekGRPOAgent:
    """
    DeepSeek-R1 기반 GRPO 에이전트

    DeepSeek-R1 아키텍처를 활용한 GRPO(Generalized Reward-Penalty Optimization) 에이전트입니다.
    트랜스포머 기반 시계열 처리, 특성 어텐션, 분포형 가치 네트워크 등의 고급 기능을 통합하여
    금융 시장 데이터의 복잡한 패턴을 효과적으로 학습합니다.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_length: int,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        penalty_scale: float = 0.5,
        kl_coef: float = 0.01,
        entropy_coef: float = 0.005,
        device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            seq_length: 시퀀스 길이
            hidden_dim: 모델 은닉 차원
            num_transformer_layers: 트랜스포머 레이어 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            lr: 학습률
            gamma: 할인 계수
            reward_scale: 보상 스케일
            penalty_scale: 페널티 스케일
            kl_coef: KL 발산 계수
            entropy_coef: 엔트로피 계수
            device: 학습 장치
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.penalty_scale = penalty_scale
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        
        # 네트워크 초기화
        self.network = DeepSeekGRPONetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 경험 버퍼
        self.reset_buffers()
        
        # 이전 정책 (KL 발산 계산용)
        self.old_action_probs = None
    
    def reset_buffers(self) -> None:
        """경험 버퍼 초기화"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.histories = []
        self.next_histories = []
    
    def select_action(
        self, 
        state: np.ndarray, 
        history: Optional[np.ndarray] = None, 
        deterministic: bool = False
    ) -> int:
        """
        정책에 따라 행동 선택

        Args:
            state: 현재 상태
            history: 상태 히스토리 (없으면 None)
            deterministic: 결정론적 행동 선택 여부
        
        Returns:
            선택된 행동 인덱스
        """
        with torch.no_grad():
            # 상태를 텐서로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 히스토리가 있으면 텐서로 변환
            if history is not None:
                history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
            else:
                history_tensor = None
            
            # 네트워크 순전파
            outputs = self.network(state_tensor, history_tensor)
            
            action_probs = outputs['action_probs']
            exploration_temp = outputs['exploration_temp']
            
            if deterministic:
                # 결정론적 행동 (최대 확률)
                action = torch.argmax(action_probs, dim=-1)
            else:
                # 확률적 행동 (탐색 온도 적용)
                if len(exploration_temp.shape) > 0:
                    temp = exploration_temp[0].item()
                else:
                    temp = exploration_temp.item()
                
                # 온도 스케일링
                if temp != 1.0:
                    scaled_probs = action_probs ** (1.0 / temp)
                    scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)
                else:
                    scaled_probs = action_probs
                
                # 확률적 샘플링
                dist = Categorical(scaled_probs)
                action = dist.sample()
        
        return action.item()
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool, 
        history: Optional[np.ndarray] = None
    ) -> None:
        """
        경험 저장

        Args:
            state: 현재 상태
            action: 행동
            reward: 보상
            next_state: 다음 상태
            done: 종료 여부
            history: 상태 히스토리 (없으면 None)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if history is not None:
            # 현재 히스토리 저장
            self.histories.append(history)
            
            # 다음 히스토리 계산 (현재 히스토리에서 가장 오래된 상태 제거하고 next_state 추가)
            next_history = np.concatenate([history[1:], np.expand_dims(state, axis=0)], axis=0)
            self.next_histories.append(next_history)
    
    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        GRPO 알고리즘으로 정책 업데이트

        Args:
            batch_size: 배치 크기 (None이면 전체 버퍼 사용)
        
        Returns:
            지표 딕셔너리 (policy_loss, value_loss, kl_div, entropy, 등)
        """
        # 버퍼가 비어있으면 업데이트 건너뛰기
        if len(self.states) == 0:
            return {}
        
        # 배치 크기 설정
        if batch_size is None or batch_size >= len(self.states):
            indices = range(len(self.states))
        else:
            indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        # 데이터 준비
        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        
        # 히스토리 데이터 준비
        if hasattr(self, 'histories') and len(self.histories) > 0:
            histories = np.array([self.histories[i] for i in indices])
            next_histories = np.array([self.next_histories[i] for i in indices])
        else:
            histories = None
            next_histories = None
        
        # 텐서 변환
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        if histories is not None:
            histories_tensor = torch.FloatTensor(histories).to(self.device)
            next_histories_tensor = torch.FloatTensor(next_histories).to(self.device)
        else:
            histories_tensor = None
            next_histories_tensor = None
        
        # 현재 정책 평가
        current_outputs = self.network.evaluate_actions(states_tensor, actions_tensor, histories_tensor)
        
        log_probs = current_outputs['log_probs']
        entropy = current_outputs['entropy']
        expected_value = current_outputs['expected_value']
        action_probs = current_outputs['action_probs']
        reward_scale = current_outputs['reward_scale']
        penalty_scale = current_outputs['penalty_scale']
        
        # 다음 상태의 가치 추정
        with torch.no_grad():
            next_outputs = self.network(next_states_tensor, next_histories_tensor)
            next_action_probs = next_outputs['action_probs']
            
            # 각 행동에 대한 다음 상태 가치 계산
            next_values = []
            for i in range(self.action_dim):
                action_indices = torch.full_like(actions_tensor, i)
                action_onehot = F.one_hot(action_indices, self.action_dim).float()
                _, value = self.network.value_network(
                    torch.cat([
                        self.network.temporal_encoder(next_histories_tensor),
                        self.network.feature_attention(next_states_tensor)[0]
                    ], dim=-1),
                    action_onehot
                )
                next_values.append(value)
            
            next_values = torch.cat(next_values, dim=1)  # [배치 크기, 행동 차원]
            
            # 다음 상태 기대 가치
            next_value = (next_values * next_action_probs).sum(dim=1, keepdim=True)
            
            # 타겟 가치
            target_value = rewards_tensor + self.gamma * next_value * (1 - dones_tensor)
        
        # 이점(advantage) 계산
        advantages = target_value - expected_value
        
        # KL 발산 계산
        kl_div = torch.zeros(1, device=self.device)
        if self.old_action_probs is not None:
            old_probs = self.old_action_probs[indices] if len(self.old_action_probs) > len(indices) else self.old_action_probs
            old_probs = torch.FloatTensor(old_probs).to(self.device)
            kl_div = (old_probs * torch.log(old_probs / (action_probs + 1e-10) + 1e-10)).sum(dim=1).mean()
        
        # 정책 손실 계산 (보상-페널티 분리)
        positive_mask = advantages > 0
        negative_mask = ~positive_mask
        
        policy_loss = torch.zeros(1, device=self.device)
        
        if positive_mask.any():
            reward_loss = -log_probs[positive_mask] * advantages[positive_mask] * reward_scale[positive_mask]
            policy_loss = policy_loss + reward_loss.mean()
        
        if negative_mask.any():
            penalty_loss = log_probs[negative_mask] * advantages[negative_mask].abs() * penalty_scale[negative_mask]
            policy_loss = policy_loss + penalty_loss.mean()
        
        # 가치 손실
        value_loss = F.mse_loss(expected_value, target_value.detach())
        
        # 총 손실
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean() + self.kl_coef * kl_div
        
        # 그래디언트 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # 그래디언트 클리핑
        self.optimizer.step()
        
        # 이전 정책 저장
        self.old_action_probs = action_probs.detach().cpu().numpy()
        
        # 버퍼 초기화
        self.reset_buffers()
        
        # 지표 반환
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.mean().item(),
            'mean_reward': rewards.mean(),
            'mean_advantage': advantages.mean().item(),
            'mean_reward_scale': reward_scale.mean().item(),
            'mean_penalty_scale': penalty_scale.mean().item()
        }
    
    def save(self, path: str) -> None:
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'seq_length': self.seq_length,
            'gamma': self.gamma,
            'reward_scale': self.reward_scale,
            'penalty_scale': self.penalty_scale,
            'kl_coef': self.kl_coef,
            'entropy_coef': self.entropy_coef
        }, path)
    
    def load(self, path: str) -> None:
        """
        모델 로드

        Args:
            path: 로드 경로
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 하이퍼파라미터 로드
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.reward_scale = checkpoint.get('reward_scale', self.reward_scale)
        self.penalty_scale = checkpoint.get('penalty_scale', self.penalty_scale)
        self.kl_coef = checkpoint.get('kl_coef', self.kl_coef)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
