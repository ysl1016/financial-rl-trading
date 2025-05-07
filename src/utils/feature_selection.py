import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class FeatureImportanceAnalyzer:
    """
    특성 중요도 분석 및 특성 선택을 위한 유틸리티 클래스
    
    다양한 특성 중요도 측정 방법과 특성 선택 알고리즘을 제공
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Args:
            data: 입력 데이터 (특성 및 타깃 포함)
            target_col: 타깃 변수 컬럼명
        """
        self.data = data.copy()
        self.features = data.drop(columns=[target_col])
        self.target = data[target_col]
        
        # 특성 이름 저장
        self.feature_names = self.features.columns.tolist()
        
        # 중요도 분석 결과 저장
        self.importance_results = {}
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        상관관계 기반 특성 중요도 분석
        
        Returns:
            importance_df: 특성 중요도 DataFrame
        """
        # 타깃과의 상관관계 계산
        correlations = []
        
        for col in self.feature_names:
            if np.issubdtype(self.features[col].dtype, np.number):
                # 피어슨 상관계수
                pearson_corr = self.features[col].corr(self.target)
                
                # 스피어만 순위 상관계수
                spearman_corr = self.features[col].corr(self.target, method='spearman')
                
                # 절대값 기준 중요도
                abs_correlation = abs(pearson_corr)
                
                correlations.append({
                    'feature': col,
                    'pearson_correlation': pearson_corr,
                    'spearman_correlation': spearman_corr,
                    'abs_correlation': abs_correlation
                })
        
        # DataFrame 생성 및 정렬
        importance_df = pd.DataFrame(correlations)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        # 중요도 결과 저장
        self.importance_results['correlation'] = importance_df
        
        return importance_df
    
    def mutual_information(self) -> pd.DataFrame:
        """
        상호 정보량 기반 특성 중요도 분석
        
        Returns:
            importance_df: 특성 중요도 DataFrame
        """
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # 타깃 변수 타입에 따라 적절한 함수 선택
        if self.target.dtype == bool or pd.api.types.is_categorical_dtype(self.target):
            # 분류 문제
            mi_func = mutual_info_classif
        else:
            # 회귀 문제
            mi_func = mutual_info_regression
        
        # 특성 행렬 준비
        X = self.features.select_dtypes(include=[np.number]).fillna(0)
        y = self.target.values
        
        # 상호 정보량 계산
        mi_scores = mi_func(X, y, random_state=42)
        
        # 결과 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'mutual_information': mi_scores,
            'normalized_mi': mi_scores / np.max(mi_scores)
        })
        
        # 중요도 기준 정렬
        importance_df = importance_df.sort_values('mutual_information', ascending=False)
        
        # 중요도 결과 저장
        self.importance_results['mutual_information'] = importance_df
        
        return importance_df
    
    def permutation_importance(self, model, X_val, y_val, n_repeats=10) -> pd.DataFrame:
        """
        순열 중요도 기반 특성 중요도 분석
        
        Args:
            model: 학습된 모델
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 타깃
            n_repeats: 반복 횟수
            
        Returns:
            importance_df: 특성 중요도 DataFrame
        """
        from sklearn.inspection import permutation_importance
        
        # 순열 중요도 계산
        result = permutation_importance(
            model, X_val, y_val, 
            n_repeats=n_repeats, 
            random_state=42
        )
        
        # 결과 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': X_val.columns.tolist(),
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
            'normalized_importance': result.importances_mean / np.max(result.importances_mean)
        })
        
        # 중요도 기준 정렬
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        # 중요도 결과 저장
        self.importance_results['permutation'] = importance_df
        
        return importance_df
    
    def recursive_feature_elimination(self, model, X, y, step=1) -> pd.DataFrame:
        """
        재귀적 특성 제거 기반 특성 중요도 분석
        
        Args:
            model: 모델 인스턴스
            X: 특성 데이터
            y: 타깃 데이터
            step: 각 단계에서 제거할 특성 수
            
        Returns:
            importance_df: 특성 중요도 DataFrame
        """
        from sklearn.feature_selection import RFE
        
        # RFE 적용
        rfe = RFE(estimator=model, n_features_to_select=1, step=step)
        rfe.fit(X, y)
        
        # 결과 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'rank': rfe.ranking_,
            'selected': rfe.support_
        })
        
        # 랭킹 기준 정렬
        importance_df = importance_df.sort_values('rank')
        
        # 중요도 결과 저장
        self.importance_results['rfe'] = importance_df
        
        return importance_df
    
    def select_features(self, method='combined', threshold=0.05, top_n=None) -> List[str]:
        """
        중요도 분석 결과에 기반한 특성 선택
        
        Args:
            method: 특성 선택 방법 ('correlation', 'mutual_information', 'permutation', 'rfe', 'combined')
            threshold: 선택 임계값
            top_n: 선택할 상위 특성 수
            
        Returns:
            selected_features: 선택된 특성 리스트
        """
        if method == 'combined' and len(self.importance_results) > 1:
            # 여러 방법의 결과를 결합
            combined_ranks = {}
            
            # 각 방법별 순위 계산
            for method_name, result_df in self.importance_results.items():
                feature_col = 'feature'
                
                if method_name == 'correlation':
                    rank_col = 'abs_correlation'
                elif method_name == 'mutual_information':
                    rank_col = 'mutual_information'
                elif method_name == 'permutation':
                    rank_col = 'importance_mean'
                elif method_name == 'rfe':
                    rank_col = 'rank'
                    # RFE는 낮은 값이 더 중요하므로 역순 처리
                    result_df['rank_score'] = result_df['rank'].max() - result_df['rank'] + 1
                    rank_col = 'rank_score'
                
                # 순위 정규화 (0-1 범위)
                if rank_col in result_df.columns:
                    max_val = result_df[rank_col].max()
                    min_val = result_df[rank_col].min()
                    
                    if max_val > min_val:
                        result_df['normalized_score'] = (result_df[rank_col] - min_val) / (max_val - min_val)
                    else:
                        result_df['normalized_score'] = 1.0
                
                    # 순위 합산
                    for _, row in result_df.iterrows():
                        feature = row[feature_col]
                        score = row['normalized_score']
                        
                        if feature in combined_ranks:
                            combined_ranks[feature] += score
                        else:
                            combined_ranks[feature] = score
            
            # 결합된 점수로 특성 정렬
            sorted_features = sorted(
                combined_ranks.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 상위 N개 특성 선택 또는 임계값 기준 선택
            if top_n is not None:
                selected_features = [feature for feature, _ in sorted_features[:top_n]]
            else:
                max_score = max(score for _, score in sorted_features)
                threshold_score = max_score * threshold
                selected_features = [
                    feature for feature, score in sorted_features 
                    if score >= threshold_score
                ]
            
        elif method in self.importance_results:
            # 단일 방법 기반 선택
            result_df = self.importance_results[method]
            
            if method == 'correlation':
                score_col = 'abs_correlation'
            elif method == 'mutual_information':
                score_col = 'mutual_information'
            elif method == 'permutation':
                score_col = 'importance_mean'
            elif method == 'rfe':
                selected_features = result_df[result_df['selected'] == True]['feature'].tolist()
                return selected_features
            
            # 상위 N개 특성 선택 또는 임계값 기준 선택
            if top_n is not None:
                selected_features = result_df.nlargest(top_n, score_col)['feature'].tolist()
            else:
                max_score = result_df[score_col].max()
                threshold_score = max_score * threshold
                selected_features = result_df[result_df[score_col] >= threshold_score]['feature'].tolist()
        
        else:
            # 분석 결과가 없는 경우 원본 특성 반환
            selected_features = self.feature_names
        
        return selected_features


class AdaptiveFeatureSelector(nn.Module):
    """
    신경망 기반 적응형 특성 선택기
    
    입력 특성의 중요도를 학습하고 동적으로 특성을 선택
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        l1_regularization: float = 0.01,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            hidden_dim: 히든 레이어 차원
            output_dim: 출력 차원
            dropout: 드롭아웃 비율
            l1_regularization: L1 정규화 강도
            feature_names: 특성 이름 리스트 (선택적)
        """
        super(AdaptiveFeatureSelector, self).__init__()
        
        self.input_dim = input_dim
        self.l1_regularization = l1_regularization
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        
        # 특성 중요도 가중치 (학습 가능)
        self.feature_importance = nn.Parameter(torch.ones(input_dim))
        
        # 특성 변환 네트워크
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 특성 [batch_size, input_dim]
            
        Returns:
            transformed: 변환된 특성 [batch_size, output_dim]
            importance_weights: 특성 중요도 가중치 [input_dim]
            importance_penalty: L1 정규화 페널티
        """
        # 소프트맥스로 중요도 가중치 정규화
        importance_weights = torch.softmax(self.feature_importance, dim=0)
        
        # 가중치 적용
        weighted_features = x * importance_weights.unsqueeze(0)
        
        # 특성 변환
        transformed = self.feature_transform(weighted_features)
        
        # L1 정규화 페널티 계산
        importance_penalty = self.l1_regularization * torch.norm(importance_weights, p=1)
        
        return transformed, importance_weights, importance_penalty
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        학습된 특성 중요도 반환
        
        Returns:
            importance_df: 특성 중요도 DataFrame
        """
        # 중요도 가중치 추출
        with torch.no_grad():
            importance_weights = torch.softmax(self.feature_importance, dim=0).cpu().numpy()
        
        # DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_weights
        })
        
        # 중요도 기준 정렬
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_top_features(self, top_n: int = 10) -> List[str]:
        """
        상위 N개 중요 특성 반환
        
        Args:
            top_n: 반환할 특성 수
            
        Returns:
            top_features: 상위 중요 특성 리스트
        """
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        return top_features


class AttentionFeatureSelector(nn.Module):
    """
    어텐션 기반 특성 선택기
    
    상태에 따라 다른 특성에 집중하는 동적 특성 선택 메커니즘
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            context_dim: 컨텍스트 특성 차원 (예: 시장 상태)
            hidden_dim: 히든 레이어 차원
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
            feature_names: 특성 이름 리스트 (선택적)
        """
        super(AttentionFeatureSelector, self).__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        
        # 특성 임베딩
        self.feature_embedding = nn.Linear(1, hidden_dim)
        
        # 컨텍스트 변환
        self.context_transform = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 다중 헤드 크로스 어텐션
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 입력 특성 [batch_size, input_dim]
            context: 컨텍스트 정보 [batch_size, context_dim]
            
        Returns:
            weighted_features: 가중치가 적용된 특성 [batch_size, input_dim]
            attention_weights: 특성별 어텐션 가중치 [batch_size, input_dim]
        """
        batch_size = features.size(0)
        
        # 각 특성을 개별 차원으로 변환
        # [batch_size, input_dim] -> [batch_size, input_dim, 1]
        features_expanded = features.unsqueeze(-1)
        
        # 각 특성을 임베딩
        # [batch_size, input_dim, 1] -> [batch_size, input_dim, hidden_dim]
        feature_embeddings = self.feature_embedding(features_expanded)
        
        # 컨텍스트 변환
        # [batch_size, context_dim] -> [batch_size, 1, hidden_dim]
        context_embedding = self.context_transform(context).unsqueeze(1)
        
        # 크로스 어텐션 적용 (컨텍스트를 쿼리로 사용)
        attn_output, attention_weights = self.cross_attention(
            query=context_embedding,
            key=feature_embeddings,
            value=feature_embeddings
        )
        
        # 어텐션 가중치 추출 및 형태 변환
        # [batch_size, 1, input_dim] -> [batch_size, input_dim]
        attention_weights = attention_weights.squeeze(1)
        
        # 가중치 적용
        weighted_features = features * attention_weights
        
        return weighted_features, attention_weights
    
    def get_importance_by_context(self, features: torch.Tensor, contexts: List[torch.Tensor]) -> pd.DataFrame:
        """
        다양한 컨텍스트에서의 특성 중요도 계산
        
        Args:
            features: 샘플 특성 [batch_size, input_dim]
            contexts: 컨텍스트 텐서 리스트 [context_dim]
            
        Returns:
            importance_df: 컨텍스트별 특성 중요도 DataFrame
        """
        results = []
        
        with torch.no_grad():
            # 각 컨텍스트에 대해 중요도 계산
            for i, context in enumerate(contexts):
                # 컨텍스트 확장
                # [context_dim] -> [batch_size, context_dim]
                context_expanded = context.unsqueeze(0).expand(features.size(0), -1)
                
                # 어텐션 가중치 계산
                _, attention_weights = self.forward(features, context_expanded)
                
                # 배치 평균 가중치
                avg_weights = attention_weights.mean(dim=0).cpu().numpy()
                
                # 결과 저장
                for j, weight in enumerate(avg_weights):
                    results.append({
                        'context_id': i,
                        'feature': self.feature_names[j],
                        'importance': weight
                    })
        
        # DataFrame 생성
        importance_df = pd.DataFrame(results)
        
        return importance_df


class DynamicFeatureEnsemble(nn.Module):
    """
    동적 특성 앙상블
    
    시장 상황에 따라 다양한 특성 선택기를 동적으로 앙상블하는 메커니즘
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_selectors: int = 3,
        ensemble_method: str = 'adaptive_weighting',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            context_dim: 컨텍스트 차원
            hidden_dim: 히든 레이어 차원
            num_selectors: 특성 선택기 수
            ensemble_method: 앙상블 방법 ('voting', 'stacking', 'adaptive_weighting')
            dropout: 드롭아웃 비율
        """
        super(DynamicFeatureEnsemble, self).__init__()
        
        self.input_dim = input_dim
        self.num_selectors = num_selectors
        self.ensemble_method = ensemble_method
        
        # 특성 선택기 그룹
        self.feature_selectors = nn.ModuleList([
            AdaptiveFeatureSelector(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                l1_regularization=0.01 * (i + 1)  # 다양한 강도의 정규화
            )
            for i in range(num_selectors)
        ])
        
        if ensemble_method == 'adaptive_weighting':
            # 컨텍스트 기반 선택기 가중치 네트워크
            self.selector_weighting = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_selectors),
                nn.Softmax(dim=-1)
            )
        
        elif ensemble_method == 'stacking':
            # 스태킹 메타 학습기
            self.meta_learner = nn.Sequential(
                nn.Linear(hidden_dim * num_selectors, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: 입력 특성 [batch_size, input_dim]
            context: 컨텍스트 정보 [batch_size, context_dim] (선택적)
            
        Returns:
            ensemble_features: 앙상블된 특성 [batch_size, hidden_dim]
            outputs: 추가 출력 딕셔너리
        """
        batch_size = features.size(0)
        selector_outputs = []
        selector_importance = []
        
        # 각 선택기 적용
        for selector in self.feature_selectors:
            transformed, importance, _ = selector(features)
            selector_outputs.append(transformed)
            selector_importance.append(importance)
        
        # 앙상블 방법에 따른 통합
        if self.ensemble_method == 'voting':
            # 단순 평균
            ensemble_features = torch.stack(selector_outputs).mean(dim=0)
            
            # 앙상블 가중치 (균등)
            ensemble_weights = torch.ones(
                batch_size, self.num_selectors, 
                device=features.device
            ) / self.num_selectors
            
        elif self.ensemble_method == 'stacking':
            # 스태킹 (모든 출력 연결 후 메타 학습기 적용)
            stacked_features = torch.cat(selector_outputs, dim=-1)
            ensemble_features = self.meta_learner(stacked_features)
            
            # 가중치 없음 (메타 학습기가 통합)
            ensemble_weights = torch.ones(
                batch_size, self.num_selectors, 
                device=features.device
            ) / self.num_selectors
            
        elif self.ensemble_method == 'adaptive_weighting' and context is not None:
            # 컨텍스트 기반 선택기 가중치 계산
            selector_weights = self.selector_weighting(context)  # [batch_size, num_selectors]
            
            # 가중 합산
            ensemble_features = torch.zeros_like(selector_outputs[0])
            
            for i, output in enumerate(selector_outputs):
                weight = selector_weights[:, i:i+1]  # [batch_size, 1]
                ensemble_features += output * weight
            
            ensemble_weights = selector_weights
            
        else:
            # 기본값: 단순 평균
            ensemble_features = torch.stack(selector_outputs).mean(dim=0)
            
            # 앙상블 가중치 (균등)
            ensemble_weights = torch.ones(
                batch_size, self.num_selectors, 
                device=features.device
            ) / self.num_selectors
        
        # 결과 반환
        outputs = {
            "selector_importance": torch.stack(selector_importance),  # [num_selectors, input_dim]
            "ensemble_weights": ensemble_weights  # [batch_size, num_selectors]
        }
        
        return ensemble_features, outputs