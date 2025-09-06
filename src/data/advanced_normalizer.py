import os
import pickle
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class AdaptiveFeatureNormalizer:
    """
    금융 시계열 데이터를 위한 적응형 특성 정규화 클래스
    
    여러 정규화 방법을 특성 유형에 따라 적용하고, 실시간 업데이트 가능
    """
    
    def __init__(
        self,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        normalization_methods: Optional[Dict[str, str]] = None,
        lookback_periods: Optional[Dict[str, int]] = None,
        clip_outliers: bool = True,
        clip_threshold: float = 3.0
    ):
        """
        Args:
            feature_groups: 특성 그룹 사전 {'그룹명': [특성1, 특성2, ...]}
            normalization_methods: 정규화 방법 사전 {'그룹명': '방법'}
                지원 방법: 'zscore', 'minmax', 'robust', 'quantile', 'log', 'tanh'
            lookback_periods: 룩백 기간 사전 {'그룹명': 기간}
            clip_outliers: 이상치 클리핑 여부
            clip_threshold: 이상치 클리핑 임계값 (표준편차 단위)
        """
        # 특성 그룹 설정
        self.feature_groups = feature_groups or {}
        
        # 정규화 방법 설정
        self.normalization_methods = normalization_methods or {}
        
        # 기본값 설정
        self.default_method = 'zscore'
        
        # 룩백 기간 설정
        self.lookback_periods = lookback_periods or {}
        self.default_lookback = 100
        
        # 이상치 처리 설정
        self.clip_outliers = clip_outliers
        self.clip_threshold = clip_threshold
        
        # 정규화 통계 저장
        self.stats = {}
        
        # 종합 특성 맵
        self.feature_to_group = {}
        for group, features in self.feature_groups.items():
            for feature in features:
                self.feature_to_group[feature] = group
    
    def fit(self, data: pd.DataFrame, mode: str = 'expand') -> 'AdaptiveFeatureNormalizer':
        """
        정규화 파라미터 학습
        
        Args:
            data: 학습 데이터
            mode: 'expand' (기존 통계 확장) 또는 'replace' (기존 통계 대체)
            
        Returns:
            self: 자기 자신
        """
        # 모든 특성 처리
        for column in data.columns:
            # 특성 그룹 확인
            group = self.feature_to_group.get(column, 'default')
            
            # 정규화 방법 결정
            method = self.normalization_methods.get(group, self.default_method)
            
            # 룩백 기간 결정
            lookback = self.lookback_periods.get(group, self.default_lookback)
            
            # 학습 데이터 준비 (최근 데이터만 사용)
            if lookback > 0 and len(data) > lookback:
                train_data = data[column].iloc[-lookback:]
            else:
                train_data = data[column]
            
            # 결측치 제외
            train_data = train_data.dropna()
            
            # 통계 계산
            column_stats = {}
            
            if method == 'zscore':
                column_stats['mean'] = train_data.mean()
                column_stats['std'] = train_data.std() if train_data.std() > 0 else 1.0
                
            elif method == 'minmax':
                column_stats['min'] = train_data.min()
                column_stats['max'] = train_data.max()
                if column_stats['min'] == column_stats['max']:
                    column_stats['min'] = column_stats['min'] - 0.5
                    column_stats['max'] = column_stats['max'] + 0.5
                
            elif method == 'robust':
                column_stats['median'] = train_data.median()
                column_stats['q1'] = train_data.quantile(0.25)
                column_stats['q3'] = train_data.quantile(0.75)
                column_stats['iqr'] = column_stats['q3'] - column_stats['q1']
                if column_stats['iqr'] == 0:
                    column_stats['iqr'] = 1.0
                
            elif method == 'quantile':
                for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                    column_stats[f'q{int(q*100)}'] = train_data.quantile(q)
                
            elif method == 'log':
                # 로그 변환을 위한 최소값 확인
                min_val = train_data.min()
                column_stats['offset'] = abs(min_val) + 1 if min_val <= 0 else 0
                
            elif method == 'tanh':
                column_stats['mean'] = train_data.mean()
                column_stats['std'] = train_data.std() if train_data.std() > 0 else 1.0
            
            # 이상치 처리를 위한 통계
            if self.clip_outliers:
                if 'mean' not in column_stats:
                    column_stats['mean'] = train_data.mean()
                if 'std' not in column_stats:
                    column_stats['std'] = train_data.std() if train_data.std() > 0 else 1.0
            
            # 통계 저장 또는 업데이트
            if column not in self.stats or mode == 'replace':
                self.stats[column] = {
                    'method': method,
                    'stats': column_stats
                }
            else:
                # 기존 통계와 확장 모드일 경우, 지수 이동 평균으로 업데이트
                alpha = 0.05  # 업데이트 비율
                for key, value in column_stats.items():
                    if key in self.stats[column]['stats']:
                        self.stats[column]['stats'][key] = (
                            (1 - alpha) * self.stats[column]['stats'][key] + alpha * value
                        )
                    else:
                        self.stats[column]['stats'][key] = value
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화 변환
        
        Args:
            data: 정규화할 데이터
            
        Returns:
            normalized_data: 정규화된 데이터
        """
        # 결과 데이터프레임 초기화
        normalized_data = pd.DataFrame(index=data.index)
        
        # 각 컬럼 정규화
        for column in data.columns:
            if column in self.stats:
                # 정규화 방법 및 통계 불러오기
                method = self.stats[column]['method']
                stats = self.stats[column]['stats']
                
                # 원본 데이터 복사
                normalized_col = data[column].copy()
                
                # 정규화 적용
                if method == 'zscore':
                    normalized_col = (normalized_col - stats['mean']) / stats['std']
                    
                elif method == 'minmax':
                    normalized_col = (normalized_col - stats['min']) / (stats['max'] - stats['min'])
                    
                elif method == 'robust':
                    normalized_col = (normalized_col - stats['median']) / stats['iqr']
                    
                elif method == 'quantile':
                    # 백분위수 기반 정규화 (0-1 범위로 매핑)
                    for i in range(len(normalized_col)):
                        val = normalized_col.iloc[i]
                        if pd.isna(val):
                            continue
                            
                        # 백분위수 판별
                        if val <= stats['q1']:
                            norm_val = 0.01 + 0.04 * (val - stats['q1']) / (stats['q1'] - stats['q1'])
                        elif val <= stats['q5']:
                            norm_val = 0.05 + 0.05 * (val - stats['q5']) / (stats['q5'] - stats['q1'])
                        elif val <= stats['q10']:
                            norm_val = 0.1 + 0.15 * (val - stats['q10']) / (stats['q10'] - stats['q5'])
                        elif val <= stats['q25']:
                            norm_val = 0.25 + 0.25 * (val - stats['q25']) / (stats['q25'] - stats['q10'])
                        elif val <= stats['q50']:
                            norm_val = 0.5 + 0.25 * (val - stats['q50']) / (stats['q50'] - stats['q25'])
                        elif val <= stats['q75']:
                            norm_val = 0.75 + 0.15 * (val - stats['q75']) / (stats['q75'] - stats['q50'])
                        elif val <= stats['q90']:
                            norm_val = 0.9 + 0.05 * (val - stats['q90']) / (stats['q90'] - stats['q75'])
                        elif val <= stats['q95']:
                            norm_val = 0.95 + 0.04 * (val - stats['q95']) / (stats['q95'] - stats['q90'])
                        else:
                            norm_val = 0.99 + 0.01 * (val - stats['q99']) / (stats['q99'] - stats['q95'])
                            norm_val = min(norm_val, 1.0)
                            
                        normalized_col.iloc[i] = norm_val
                        
                elif method == 'log':
                    # 로그 변환
                    normalized_col = np.log1p(normalized_col + stats['offset'])
                    
                elif method == 'tanh':
                    # 쌍곡탄젠트 변환 (소프트 클리핑)
                    normalized_col = np.tanh((normalized_col - stats['mean']) / (stats['std'] * 2))
                
                # 이상치 클리핑
                if self.clip_outliers:
                    if method not in ['minmax', 'quantile', 'tanh']:  # 이미 범위가 제한된 경우 제외
                        lower_bound = stats['mean'] - self.clip_threshold * stats['std']
                        upper_bound = stats['mean'] + self.clip_threshold * stats['std']
                        normalized_col = normalized_col.clip(
                            lower=lower_bound, upper=upper_bound
                        )
                
                # 결과 저장
                normalized_data[column] = normalized_col
            else:
                # 통계가 없는 경우 원본 복사
                normalized_data[column] = data[column]
        
        return normalized_data
    
    def fit_transform(self, data: pd.DataFrame, mode: str = 'expand') -> pd.DataFrame:
        """
        학습 후 데이터 정규화 변환
        
        Args:
            data: 정규화할 데이터
            mode: 'expand' 또는 'replace'
            
        Returns:
            normalized_data: 정규화된 데이터
        """
        self.fit(data, mode=mode)
        return self.transform(data)
    
    def inverse_transform(self, normalized_data: pd.DataFrame) -> pd.DataFrame:
        """
        정규화 역변환
        
        Args:
            normalized_data: 정규화된 데이터
            
        Returns:
            original_data: 원본 스케일의 데이터
        """
        # 결과 데이터프레임 초기화
        original_data = pd.DataFrame(index=normalized_data.index)
        
        # 각 컬럼 역변환
        for column in normalized_data.columns:
            if column in self.stats:
                # 정규화 방법 및 통계 불러오기
                method = self.stats[column]['method']
                stats = self.stats[column]['stats']
                
                # 정규화된 데이터 복사
                original_col = normalized_data[column].copy()
                
                # 역변환 적용
                if method == 'zscore':
                    original_col = original_col * stats['std'] + stats['mean']
                    
                elif method == 'minmax':
                    original_col = original_col * (stats['max'] - stats['min']) + stats['min']
                    
                elif method == 'robust':
                    original_col = original_col * stats['iqr'] + stats['median']
                    
                elif method == 'quantile':
                    # 백분위수 기반 역변환 (근사값)
                    for i in range(len(original_col)):
                        val = original_col.iloc[i]
                        if pd.isna(val):
                            continue
                            
                        # 백분위수 판별
                        if val <= 0.01:
                            orig_val = stats['q1']
                        elif val <= 0.05:
                            orig_val = stats['q1'] + (val - 0.01) * (stats['q5'] - stats['q1']) / 0.04
                        elif val <= 0.1:
                            orig_val = stats['q5'] + (val - 0.05) * (stats['q10'] - stats['q5']) / 0.05
                        elif val <= 0.25:
                            orig_val = stats['q10'] + (val - 0.1) * (stats['q25'] - stats['q10']) / 0.15
                        elif val <= 0.5:
                            orig_val = stats['q25'] + (val - 0.25) * (stats['q50'] - stats['q25']) / 0.25
                        elif val <= 0.75:
                            orig_val = stats['q50'] + (val - 0.5) * (stats['q75'] - stats['q50']) / 0.25
                        elif val <= 0.9:
                            orig_val = stats['q75'] + (val - 0.75) * (stats['q90'] - stats['q75']) / 0.15
                        elif val <= 0.95:
                            orig_val = stats['q90'] + (val - 0.9) * (stats['q95'] - stats['q90']) / 0.05
                        elif val <= 0.99:
                            orig_val = stats['q95'] + (val - 0.95) * (stats['q99'] - stats['q95']) / 0.04
                        else:
                            orig_val = stats['q99']
                            
                        original_col.iloc[i] = orig_val
                        
                elif method == 'log':
                    # 로그 역변환
                    original_col = np.expm1(original_col) - stats['offset']
                    
                elif method == 'tanh':
                    # 쌍곡탄젠트 역변환
                    original_col = np.arctanh(original_col) * (stats['std'] * 2) + stats['mean']
                
                # 결과 저장
                original_data[column] = original_col
            else:
                # 통계가 없는 경우 그대로 복사
                original_data[column] = normalized_data[column]
        
        return original_data
    
    def update_statistics(self, new_data: pd.DataFrame, alpha: float = 0.05) -> None:
        """
        온라인 모드로 정규화 통계 업데이트
        
        Args:
            new_data: 새로운 데이터
            alpha: 업데이트 비율 (0-1)
        """
        # 각 컬럼 통계 업데이트
        for column in new_data.columns:
            if column in self.stats:
                # 정규화 방법 및 기존 통계 불러오기
                method = self.stats[column]['method']
                stats = self.stats[column]['stats']
                
                # 결측치 제외
                column_data = new_data[column].dropna()
                
                if len(column_data) == 0:
                    continue
                
                # 통계 업데이트
                if method == 'zscore':
                    # 평균 및 표준편차 업데이트
                    new_mean = column_data.mean()
                    new_std = column_data.std() if column_data.std() > 0 else 1.0
                    
                    stats['mean'] = (1 - alpha) * stats['mean'] + alpha * new_mean
                    stats['std'] = (1 - alpha) * stats['std'] + alpha * new_std
                    
                elif method == 'minmax':
                    # 최소/최대값 업데이트 (점진적으로 확장)
                    new_min = column_data.min()
                    new_max = column_data.max()
                    
                    if new_min < stats['min']:
                        stats['min'] = new_min
                    
                    if new_max > stats['max']:
                        stats['max'] = new_max
                    
                elif method == 'robust':
                    # 중앙값 및 IQR 업데이트
                    new_median = column_data.median()
                    new_q1 = column_data.quantile(0.25)
                    new_q3 = column_data.quantile(0.75)
                    new_iqr = new_q3 - new_q1
                    
                    if new_iqr == 0:
                        new_iqr = 1.0
                    
                    stats['median'] = (1 - alpha) * stats['median'] + alpha * new_median
                    stats['q1'] = (1 - alpha) * stats['q1'] + alpha * new_q1
                    stats['q3'] = (1 - alpha) * stats['q3'] + alpha * new_q3
                    stats['iqr'] = (1 - alpha) * stats['iqr'] + alpha * new_iqr
                    
                elif method == 'quantile':
                    # 백분위수 업데이트
                    for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
                        new_q = column_data.quantile(q)
                        stats[f'q{int(q*100)}'] = (1 - alpha) * stats[f'q{int(q*100)}'] + alpha * new_q
                    
                elif method == 'log':
                    # 오프셋 업데이트
                    new_min = column_data.min()
                    if new_min <= 0:
                        new_offset = abs(new_min) + 1
                        stats['offset'] = max(stats['offset'], new_offset)
                    
                elif method == 'tanh':
                    # 평균 및 표준편차 업데이트
                    new_mean = column_data.mean()
                    new_std = column_data.std() if column_data.std() > 0 else 1.0
                    
                    stats['mean'] = (1 - alpha) * stats['mean'] + alpha * new_mean
                    stats['std'] = (1 - alpha) * stats['std'] + alpha * new_std
    
    def save(self, path: str) -> None:
        """
        정규화 상태 저장
        
        Args:
            path: 저장 경로
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'feature_groups': self.feature_groups,
                'normalization_methods': self.normalization_methods,
                'lookback_periods': self.lookback_periods,
                'clip_outliers': self.clip_outliers,
                'clip_threshold': self.clip_threshold,
                'stats': self.stats,
                'feature_to_group': self.feature_to_group
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'AdaptiveFeatureNormalizer':
        """
        정규화 상태 로드
        
        Args:
            path: 로드 경로
            
        Returns:
            normalizer: 로드된 정규화기
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        normalizer = cls(
            feature_groups=data['feature_groups'],
            normalization_methods=data['normalization_methods'],
            lookback_periods=data['lookback_periods'],
            clip_outliers=data['clip_outliers'],
            clip_threshold=data['clip_threshold']
        )
        
        normalizer.stats = data['stats']
        normalizer.feature_to_group = data['feature_to_group']
        
        return normalizer


class MultiResolutionNormalizer:
    """
    다중 해상도 정규화기
    
    여러 시간 스케일에 따라 다른 정규화 전략 적용
    """
    
    def __init__(
        self,
        time_scales: List[int],
        feature_groups: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            time_scales: 시간 스케일 리스트 (예: [1, 5, 20, 60])
            feature_groups: 특성 그룹 사전 {'그룹명': [특성1, 특성2, ...]}
        """
        self.time_scales = time_scales
        self.feature_groups = feature_groups or {}
        
        # 각 시간 스케일별 정규화기
        self.normalizers = {}
        
        # 각 그룹별 최적 정규화 방법 설정
        self.group_methods = {}
        self._set_default_methods()
    
    def _set_default_methods(self) -> None:
        """기본 정규화 방법 설정"""
        # 특성 그룹별 정규화 방법
        default_methods = {
            'price': 'log',         # 가격 데이터
            'volume': 'log',        # 거래량 데이터
            'returns': 'tanh',      # 수익률 데이터
            'oscillators': 'minmax',  # RSI 등 범위가 제한된 오실레이터
            'momentum': 'zscore',   # 모멘텀 지표
            'volatility': 'log',    # 변동성 지표
            'macro': 'robust',      # 거시경제 지표
            'sentiment': 'minmax',  # 감성 지표
            'default': 'zscore'     # 기본값
        }
        
        # 시간 스케일별 룩백 기간 조정
        for scale in self.time_scales:
            methods = default_methods.copy()
            
            # 시간 스케일에 따른 정규화 방법 미세 조정
            if scale < 5:  # 초단기
                methods['returns'] = 'robust'  # 이상치에 강건한 방법
                methods['volume'] = 'robust'   # 이상치에 강건한 방법
            elif scale > 20:  # 장기
                methods['oscillators'] = 'zscore'  # 장기적으로는 분포 중요
            
            normalizer = AdaptiveFeatureNormalizer(
                feature_groups=self.feature_groups,
                normalization_methods=methods,
                lookback_periods={
                    'price': max(100, scale * 5),
                    'volume': max(100, scale * 5),
                    'returns': max(250, scale * 10),
                    'oscillators': max(100, scale * 4),
                    'momentum': max(100, scale * 4),
                    'volatility': max(100, scale * 5),
                    'macro': max(50, scale * 3),
                    'sentiment': max(50, scale * 2),
                    'default': max(100, scale * 5)
                },
                clip_outliers=True,
                clip_threshold=3.0 if scale < 10 else 4.0  # 장기 데이터는 더 넓은 범위 허용
            )
            
            self.normalizers[scale] = normalizer
    
    def fit(self, data: pd.DataFrame, time_scale: int) -> 'MultiResolutionNormalizer':
        """
        특정 시간 스케일에 대한 정규화기 학습
        
        Args:
            data: 학습 데이터
            time_scale: 시간 스케일
            
        Returns:
            self: 자기 자신
        """
        if time_scale in self.normalizers:
            self.normalizers[time_scale].fit(data)
        else:
            # 가장 가까운 스케일 선택
            closest_scale = min(self.time_scales, key=lambda x: abs(x - time_scale))
            
            # 새 정규화기 생성 및 학습
            self.normalizers[time_scale] = self.normalizers[closest_scale]
            self.normalizers[time_scale].fit(data)
            
            # 시간 스케일 리스트 업데이트
            if time_scale not in self.time_scales:
                self.time_scales.append(time_scale)
                self.time_scales.sort()
        
        return self
    
    def transform(self, data: pd.DataFrame, time_scale: int) -> pd.DataFrame:
        """
        특정 시간 스케일에 대한 데이터 정규화
        
        Args:
            data: 정규화할 데이터
            time_scale: 시간 스케일
            
        Returns:
            normalized_data: 정규화된 데이터
        """
        if time_scale in self.normalizers:
            return self.normalizers[time_scale].transform(data)
        else:
            # 가장 가까운 스케일 선택
            closest_scale = min(self.time_scales, key=lambda x: abs(x - time_scale))
            return self.normalizers[closest_scale].transform(data)
    
    def fit_transform(self, data: pd.DataFrame, time_scale: int) -> pd.DataFrame:
        """
        특정 시간 스케일에 대한 학습 및 정규화
        
        Args:
            data: 정규화할 데이터
            time_scale: 시간 스케일
            
        Returns:
            normalized_data: 정규화된 데이터
        """
        self.fit(data, time_scale)
        return self.transform(data, time_scale)
    
    def inverse_transform(self, normalized_data: pd.DataFrame, time_scale: int) -> pd.DataFrame:
        """
        특정 시간 스케일에 대한 정규화 역변환
        
        Args:
            normalized_data: 정규화된 데이터
            time_scale: 시간 스케일
            
        Returns:
            original_data: 원본 스케일의 데이터
        """
        if time_scale in self.normalizers:
            return self.normalizers[time_scale].inverse_transform(normalized_data)
        else:
            # 가장 가까운 스케일 선택
            closest_scale = min(self.time_scales, key=lambda x: abs(x - time_scale))
            return self.normalizers[closest_scale].inverse_transform(normalized_data)
    
    def update_statistics(self, new_data: pd.DataFrame, time_scale: int, alpha: float = 0.05) -> None:
        """
        특정 시간 스케일에 대한 정규화 통계 업데이트
        
        Args:
            new_data: 새로운 데이터
            time_scale: 시간 스케일
            alpha: 업데이트 비율 (0-1)
        """
        if time_scale in self.normalizers:
            self.normalizers[time_scale].update_statistics(new_data, alpha)
        else:
            # 가장 가까운 스케일 선택
            closest_scale = min(self.time_scales, key=lambda x: abs(x - time_scale))
            self.normalizers[closest_scale].update_statistics(new_data, alpha)
    
    def save(self, path: str) -> None:
        """
        정규화 상태 저장
        
        Args:
            path: 저장 경로
        """
        # 디렉토리 생성
        os.makedirs(path, exist_ok=True)

        # 메타 정보 저장
        meta_path = os.path.join(path, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'time_scales': self.time_scales,
                'feature_groups': self.feature_groups,
                'group_methods': self.group_methods
            }, f)
        
        # 각 정규화기 저장
        for scale, normalizer in self.normalizers.items():
            scale_path = os.path.join(path, f'scale_{scale}.pkl')
            normalizer.save(scale_path)
    
    @classmethod
    def load(cls, path: str) -> 'MultiResolutionNormalizer':
        """
        정규화 상태 로드
        
        Args:
            path: 로드 경로
            
        Returns:
            normalizer: 로드된 정규화기
        """
        # 메타 정보 로드
        meta_path = os.path.join(path, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        # 인스턴스 생성
        multi_normalizer = cls(
            time_scales=meta['time_scales'],
            feature_groups=meta['feature_groups']
        )
        
        multi_normalizer.group_methods = meta.get('group_methods', {})
        
        # 각 정규화기 로드
        for scale in meta['time_scales']:
            scale_path = os.path.join(path, f'scale_{scale}.pkl')
            if os.path.exists(scale_path):
                multi_normalizer.normalizers[scale] = AdaptiveFeatureNormalizer.load(scale_path)
        
        return multi_normalizer


class AdaptiveNormalizationLayer(nn.Module):
    """
    적응형 정규화 레이어
    
    신경망 내에서 특성의 동적 정규화 수행
    """
    
    def __init__(
        self,
        num_features: int,
        normalization_type: str = 'batch',
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        feature_groups: Optional[List[List[int]]] = None
    ):
        """
        Args:
            num_features: 특성 수
            normalization_type: 정규화 유형 ('batch', 'layer', 'instance', 'group', 'adaptive')
            momentum: 이동 평균 모멘텀
            affine: 학습 가능한 감마/베타 파라미터 사용 여부
            track_running_stats: 실행 중 통계 추적 여부
            feature_groups: 특성 그룹 리스트 (GroupNorm 및 AdaptiveNorm용)
        """
        super(AdaptiveNormalizationLayer, self).__init__()
        
        self.num_features = num_features
        self.normalization_type = normalization_type
        self.feature_groups = feature_groups
        
        # 정규화 레이어 선택
        if normalization_type == 'batch':
            self.norm = nn.BatchNorm1d(
                num_features, 
                momentum=momentum, 
                affine=affine, 
                track_running_stats=track_running_stats
            )
            
        elif normalization_type == 'layer':
            self.norm = nn.LayerNorm(
                num_features,
                elementwise_affine=affine
            )
            
        elif normalization_type == 'instance':
            self.norm = nn.InstanceNorm1d(
                num_features, 
                momentum=momentum, 
                affine=affine, 
                track_running_stats=track_running_stats
            )
            
        elif normalization_type == 'group':
            # 특성 그룹이 없으면 단일 그룹으로 처리
            num_groups = len(feature_groups) if feature_groups else 1
            self.norm = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_features,
                affine=affine
            )
            
        elif normalization_type == 'adaptive':
            # 적응형 정규화 레이어 (여러 정규화 방법의 가중 결합)
            self.batch_norm = nn.BatchNorm1d(
                num_features, 
                momentum=momentum, 
                affine=False, 
                track_running_stats=track_running_stats
            )
            self.layer_norm = nn.LayerNorm(
                num_features,
                elementwise_affine=False
            )
            
            # 각 특성별 정규화 방법 가중치 (학습 가능)
            self.norm_weights = nn.Parameter(torch.ones(2, num_features) / 2)
            
            # 감마/베타 파라미터 (affine 변환)
            if affine:
                self.gamma = nn.Parameter(torch.ones(num_features))
                self.beta = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter('gamma', None)
                self.register_parameter('beta', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 특성 [batch_size, num_features] 또는 [batch_size, num_features, seq_length]
            
        Returns:
            normalized: 정규화된 특성
        """
        # 입력 형태 확인
        if x.dim() == 3:
            # [batch_size, num_features, seq_length] -> [batch_size, num_features]
            # 마지막 시점만 사용
            x = x[:, :, -1]
        
        if self.normalization_type == 'adaptive':
            # 각 정규화 방법 적용
            batch_normalized = self.batch_norm(x)
            layer_normalized = self.layer_norm(x)
            
            # 정규화 방법 가중치
            norm_weights = torch.softmax(self.norm_weights, dim=0)
            
            # 가중 결합
            normalized = (
                norm_weights[0].unsqueeze(0) * batch_normalized +
                norm_weights[1].unsqueeze(0) * layer_normalized
            )
            
            # 감마/베타 파라미터 적용
            if self.gamma is not None:
                normalized = normalized * self.gamma + self.beta
        
        elif self.normalization_type == 'group' and self.feature_groups:
            # 특성 그룹별 정규화 (마스킹 필요)
            normalized = x.clone()
            
            for group_idx, feature_indices in enumerate(self.feature_groups):
                mask = torch.zeros_like(x)
                mask[:, feature_indices] = 1
                
                # 그룹별 정규화 적용
                group_input = x * mask
                group_norm = self.norm(group_input)
                
                # 결과 결합
                normalized = normalized * (1 - mask) + group_norm * mask
        
        else:
            # 표준 정규화 적용
            normalized = self.norm(x)
        
        return normalized


def create_default_feature_groups() -> Dict[str, List[str]]:
    """
    일반적인 금융 특성 그룹 생성
    
    Returns:
        feature_groups: 특성 그룹 사전
    """
    feature_groups = {
        # 가격 관련 특성
        'price': [
            'Open', 'High', 'Low', 'Close', 'Adj Close',
            'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA200',
            'EMA5', 'EMA10', 'EMA20', 'EMA50', 'EMA200',
            'VWAP', 'TypicalPrice', 'HLC3'
        ],
        
        # 거래량 관련 특성
        'volume': [
            'Volume', 'VolumeEMA', 'OBV', 'VPT', 'ACVI',
            'ForceIndex2', 'MFI', 'EnhancedMFI', 'RVWRSI'
        ],
        
        # 수익률 관련 특성
        'returns': [
            'Returns1D', 'Returns5D', 'Returns10D', 'Returns20D',
            'LogReturns1D', 'LogReturns5D', 'LogReturns10D',
            'ExcessReturns', 'AbnormalReturns'
        ],
        
        # 오실레이터 지표
        'oscillators': [
            'RSI', 'ARSI', 'FisherRSI', 'StochK', 'StochD', '%K', '%D',
            'WaveTrend1', 'WaveTrend2', 'TTMTrend', 'RMI'
        ],
        
        # 모멘텀 지표
        'momentum': [
            'MACD', 'MACDSignal', 'MACDHist', 'ROC', 'KST', 'KSTSignal',
            'MTFMomentum', 'Impulse', 'SqueezeMomentum'
        ],
        
        # 변동성 지표
        'volatility': [
            'ATR', 'BBWidth', 'ThresholdVol', 'YangZhangVol', 'GARCHVol',
            'ImpliedVolProxy', 'VolCrunch', 'MarketEfficiency'
        ],
        
        # 거시경제 지표
        'macro': [
            'Macro_InterestRate', 'Macro_Inflation', 'Macro_GDP_Growth',
            'Macro_UnemploymentRate', 'Macro_YieldCurveSlope', 'Macro_RealInterestRate',
            'Macro_LeadingEconomicIndex', 'Macro_PMI'
        ],
        
        # 감성 지표
        'sentiment': [
            'News_sentiment_score', 'News_positive_count', 'News_negative_count',
            'News_economic_count', 'News_corporate_count',
            'ImpliedVol', 'FearGreedIndex', 'VolatilityRiskPremium'
        ],
        
        # 패턴 및 군집 지표
        'patterns': [
            'ClusterScore', 'LocalOutlierScore', 'ARResiduals',
            'TFDivergence', 'AdaptiveTL'
        ],
        
        # 캘린더 특성
        'calendar': [
            'Calendar_DayOfWeek', 'Calendar_MonthProgress', 'Calendar_YearProgress',
            'Calendar_MonthStart', 'Calendar_MonthEnd', 'Calendar_QuarterEnd',
            'IntradayPattern'
        ]
    }
    
    return feature_groups