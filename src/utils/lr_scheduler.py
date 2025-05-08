#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
학습률 스케줄링 유틸리티

이 모듈은 DeepSeek-R1 GRPO 에이전트 훈련을 위한 다양한 학습률 스케줄링 전략을 제공합니다.
워밍업, 코사인 감소, 단계적 감소, 사이클릭 및 원-사이클 스케줄링 등을 지원합니다.
"""

import math
from typing import List, Optional, Union, Callable

import numpy as np
import torch
from torch.optim import Optimizer


class LRScheduler:
    """학습률 스케줄러 기본 클래스"""
    
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()
    
    def state_dict(self):
        """스케줄러 상태 딕셔너리 반환"""
        return {'last_epoch': self.last_epoch}
    
    def load_state_dict(self, state_dict):
        """스케줄러 상태 로드"""
        self.last_epoch = state_dict['last_epoch']
    
    def get_lr(self):
        """현재 학습률 반환 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def step(self, epoch=None):
        """학습률 업데이트"""
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch
        
        values = self.get_lr()
        
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
            param_group['lr'] = lr
        
        return values


class WarmupCosineDecayLR(LRScheduler):
    """워밍업 후 코사인 감쇠 스케줄러"""
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 최적화기 객체
            warmup_steps: 워밍업 단계 수
            total_steps: 전체 학습 단계 수
            min_lr: 최소 학습률 (기본값: 0.0)
            last_epoch: 마지막 에폭 (기본값: -1)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """현재 학습률 계산"""
        if self.last_epoch < 0:
            return self.base_lrs
        
        if self.last_epoch < self.warmup_steps:
            # 워밍업 단계
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [lr * alpha for lr in self.base_lrs]
        
        # 코사인 감쇠
        progress = float(self.last_epoch - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (lr - self.min_lr) * cosine_decay 
            for lr in self.base_lrs
        ]


class WarmupLinearDecayLR(LRScheduler):
    """워밍업 후 선형 감쇠 스케줄러"""
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 최적화기 객체
            warmup_steps: 워밍업 단계 수
            total_steps: 전체 학습 단계 수
            min_lr: 최소 학습률 (기본값: 0.0)
            last_epoch: 마지막 에폭 (기본값: -1)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """현재 학습률 계산"""
        if self.last_epoch < 0:
            return self.base_lrs
        
        if self.last_epoch < self.warmup_steps:
            # 워밍업 단계
            alpha = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [lr * alpha for lr in self.base_lrs]
        
        # 선형 감쇠
        progress = float(self.last_epoch - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        linear_decay = 1.0 - progress
        
        return [
            self.min_lr + (lr - self.min_lr) * linear_decay 
            for lr in self.base_lrs
        ]


class CyclicLR(LRScheduler):
    """사이클릭 학습률 스케줄러"""
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[int], float]] = None,
        scale_mode: str = 'cycle',
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 최적화기 객체
            base_lr: 기본 학습률
            max_lr: 최대 학습률
            step_size_up: 상승 단계 수
            step_size_down: 하강 단계 수
            mode: 스케줄링 모드 ('triangular', 'triangular2', 'exp_range')
            gamma: 감마 계수 (mode='exp_range'일 때 사용)
            scale_fn: 사용자 정의 스케일링 함수
            scale_mode: 스케일링 모드 ('cycle' 또는 'iterations')
            last_epoch: 마지막 에폭
        """
        if not isinstance(base_lr, list) and not isinstance(base_lr, tuple):
            self.base_lrs = [base_lr] * len(optimizer.param_groups)
        else:
            self.base_lrs = list(base_lr)
        
        if not isinstance(max_lr, list) and not isinstance(max_lr, tuple):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
        
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        
        # 스케일링 함수 설정
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1.0 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
            else:
                raise ValueError(f'지원되지 않는 모드: {self.mode}')
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """현재 학습률 계산"""
        if self.last_epoch < 0:
            return self.base_lrs
        
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        
        if x <= self.step_size_up / self.total_size:
            # 상승 단계
            scale_factor = x * self.total_size / self.step_size_up
        else:
            # 하강 단계
            scale_factor = (x - self.step_size_up / self.total_size) * self.total_size / self.step_size_down
            scale_factor = 1 - scale_factor
        
        # 스케일링 계수
        if self.scale_mode == 'cycle':
            lr_scale = self.scale_fn(cycle)
        else:
            lr_scale = self.scale_fn(self.last_epoch)
        
        return [
            base_lr + (max_lr - base_lr) * scale_factor * lr_scale
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
        ]


class OneCycleLR(LRScheduler):
    """원 사이클 학습률 스케줄러"""
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        max_lr: Union[float, List[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 최적화기 객체
            max_lr: 최대 학습률
            total_steps: 전체 학습 단계 수
            pct_start: 상승 단계가 차지하는 비율
            anneal_strategy: 감쇠 전략 ('cos' 또는 'linear')
            div_factor: 초기 학습률 = max_lr / div_factor
            final_div_factor: 최종 학습률 = max_lr / final_div_factor
            last_epoch: 마지막 에폭
        """
        if not isinstance(max_lr, list) and not isinstance(max_lr, tuple):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
        
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # 단계 크기 계산
        self.step_size_up = int(self.total_steps * self.pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        
        # 초기 및 최종 학습률 계산
        self.base_lrs = [max_lr / self.div_factor for max_lr in self.max_lrs]
        self.final_lrs = [max_lr / self.final_div_factor for max_lr in self.max_lrs]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """현재 학습률 계산"""
        if self.last_epoch < 0:
            return self.base_lrs
        
        if self.last_epoch < self.step_size_up:
            # 상승 단계
            progress = float(self.last_epoch) / float(self.step_size_up)
            
            if self.anneal_strategy == 'cos':
                # 코사인 상승
                cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
                return [
                    base_lr + (max_lr - base_lr) * cos_progress
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
                ]
            else:
                # 선형 상승
                return [
                    base_lr + (max_lr - base_lr) * progress
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
                ]
        else:
            # 하강 단계
            progress = float(self.last_epoch - self.step_size_up) / float(self.step_size_down)
            
            if self.anneal_strategy == 'cos':
                # 코사인 하강
                cos_progress = 0.5 * (1 + math.cos(math.pi * progress))
                return [
                    max_lr + (final_lr - max_lr) * cos_progress
                    for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)
                ]
            else:
                # 선형 하강
                return [
                    max_lr + (final_lr - max_lr) * progress
                    for max_lr, final_lr in zip(self.max_lrs, self.final_lrs)
                ]


class NoamLR(LRScheduler):
    """Noam 학습률 스케줄러 (Transformer 논문에서 사용)"""
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        model_size: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 최적화기 객체
            model_size: 모델 차원 크기
            warmup_steps: 워밍업 단계 수
            factor: 스케일링 계수
            last_epoch: 마지막 에폭
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """현재 학습률 계산"""
        if self.last_epoch < 0:
            return self.base_lrs
        
        step = self.last_epoch + 1
        scale = self.factor * (self.model_size ** (-0.5)) * min(
            step ** (-0.5), step * (self.warmup_steps ** (-1.5))
        )
        
        return [scale for _ in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs
) -> LRScheduler:
    """지정된 이름의 스케줄러 반환"""
    name = name.lower()
    
    if name == 'warmup_cosine':
        return WarmupCosineDecayLR(optimizer, **kwargs)
    elif name == 'warmup_linear':
        return WarmupLinearDecayLR(optimizer, **kwargs)
    elif name == 'cyclic':
        return CyclicLR(optimizer, **kwargs)
    elif name == 'one_cycle':
        return OneCycleLR(optimizer, **kwargs)
    elif name == 'noam':
        return NoamLR(optimizer, **kwargs)
    else:
        raise ValueError(f"지원되지 않는 스케줄러: {name}")
