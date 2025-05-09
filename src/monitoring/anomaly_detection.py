import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """모델 드리프트 감지 결과"""
    detected: bool = False
    drift_score: float = 0.0
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    p_value: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class AnomalyDetectionResult:
    """이상 감지 결과"""
    detected: bool = False
    anomaly_score: float = 0.0
    anomaly_type: str = ""
    affected_features: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


class ModelDriftDetector:
    """
    모델 드리프트 감지 클래스.
    시간이 지남에 따라 모델 입력 또는 출력 분포의 변화를 감지합니다.
    """
    
    def __init__(self, 
                reference_data: pd.DataFrame,
                feature_columns: List[str],
                window_size: int = 100,
                drift_threshold: float = 0.05,
                method: str = "ks_test"):
        """
        ModelDriftDetector 초기화
        
        Args:
            reference_data: 기준 데이터 (정상 분포로 간주)
            feature_columns: 분석할 특성 열 이름
            window_size: 현재 윈도우 크기 (최근 데이터 포인트 수)
            drift_threshold: 드리프트 감지 임계값 (p-value)
            method: 드리프트 감지 방법 ('ks_test', 'kl_divergence')
        """
        self.reference_data = reference_data[feature_columns].copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.method = method
        
        # 현재 윈도우 데이터
        self.current_window = deque(maxlen=window_size)
        
        # 드리프트 이력
        self.drift_history = []
        
        # 특성별 참조 통계
        self.reference_stats = self._compute_reference_stats()
        
        logger.info(f"ModelDriftDetector 초기화 완료: {len(feature_columns)} 개 특성, 방법: {method}")
    
    def _compute_reference_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        참조 데이터의 통계 계산
        
        Returns:
            특성별 통계 정보
        """
        stats = {}
        
        for col in self.feature_columns:
            col_data = self.reference_data[col].dropna().values
            
            stats[col] = {
                "mean": np.mean(col_data),
                "std": np.std(col_data),
                "min": np.min(col_data),
                "max": np.max(col_data),
                "median": np.median(col_data),
                "q1": np.percentile(col_data, 25),
                "q3": np.percentile(col_data, 75),
                "hist": np.histogram(col_data, bins=20, density=True)[0]  # 정규화된 히스토그램
            }
        
        return stats
    
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """
        새로운 관측 데이터 추가
        
        Args:
            observation: 관측 데이터 (feature_columns에 있는 키를 포함해야 함)
        """
        # 특성 추출
        features = {col: observation.get(col, np.nan) for col in self.feature_columns}
        
        # 현재 윈도우에 추가
        self.current_window.append(features)
    
    def check_drift(self) -> DriftDetectionResult:
        """
        현재 윈도우에서 드리프트 감지
        
        Returns:
            드리프트 감지 결과
        """
        # 현재 윈도우가 충분히 차 있는지 확인
        if len(self.current_window) < self.window_size * 0.5:  # 최소 50% 이상 차 있어야 함
            return DriftDetectionResult(
                detected=False,
                description="Not enough data in current window"
            )
        
        # 현재 윈도우 데이터를 DataFrame으로 변환
        current_df = pd.DataFrame(list(self.current_window))
        
        # 방법에 따른 드리프트 감지
        if self.method == "ks_test":
            return self._detect_drift_ks_test(current_df)
        elif self.method == "kl_divergence":
            return self._detect_drift_kl_divergence(current_df)
        else:
            logger.warning(f"알 수 없는 드리프트 감지 방법: {self.method}")
            return DriftDetectionResult(
                detected=False,
                description=f"Unknown drift detection method: {self.method}"
            )
    
    def _detect_drift_ks_test(self, current_df: pd.DataFrame) -> DriftDetectionResult:
        """
        Kolmogorov-Smirnov 테스트를 사용한 드리프트 감지
        
        Args:
            current_df: 현재 윈도우 데이터
            
        Returns:
            드리프트 감지 결과
        """
        # 결과 객체 초기화
        result = DriftDetectionResult(
            detected=False,
            drift_score=0.0,
            timestamp=datetime.now()
        )
        
        # 각 특성별로 KS 테스트 수행
        feature_p_values = {}
        
        for col in self.feature_columns:
            # 결측치 제거
            ref_data = self.reference_data[col].dropna().values
            cur_data = current_df[col].dropna().values
            
            if len(cur_data) < 10:  # 최소 10개 관측치 필요
                continue
            
            # KS 테스트 수행
            ks_statistic, p_value = stats.ks_2samp(ref_data, cur_data)
            
            # 결과 저장
            feature_p_values[col] = p_value
            result.feature_contributions[col] = 1.0 - p_value  # 기여도 = 1 - p_value
        
        # 전체 p-value 계산 (특성 중 최소값 사용)
        if feature_p_values:
            result.p_value = min(feature_p_values.values())
            
            # 드리프트 감지 여부 결정
            if result.p_value < self.drift_threshold:
                result.detected = True
                result.drift_score = 1.0 - result.p_value
                
                # 가장 기여도가 높은 특성 식별
                most_contributing = max(result.feature_contributions.items(), key=lambda x: x[1])
                result.description = (
                    f"Drift detected (p={result.p_value:.4f}), "
                    f"most significant feature: {most_contributing[0]} "
                    f"(contribution={most_contributing[1]:.4f})"
                )
            else:
                result.description = f"No drift detected (p={result.p_value:.4f})"
        else:
            result.description = "Not enough valid features for drift detection"
        
        # 드리프트 이력에 추가
        self.drift_history.append((datetime.now(), result))
        
        return result
    
    def _detect_drift_kl_divergence(self, current_df: pd.DataFrame) -> DriftDetectionResult:
        """
        KL Divergence를 사용한 드리프트 감지
        
        Args:
            current_df: 현재 윈도우 데이터
            
        Returns:
            드리프트 감지 결과
        """
        # 결과 객체 초기화
        result = DriftDetectionResult(
            detected=False,
            drift_score=0.0,
            timestamp=datetime.now()
        )
        
        # 각 특성별로 KL Divergence 계산
        feature_kl_divs = {}
        
        for col in self.feature_columns:
            # 결측치 제거
            cur_data = current_df[col].dropna().values
            
            if len(cur_data) < 10:  # 최소 10개 관측치 필요
                continue
            
            # 현재 데이터의 히스토그램 계산
            hist_curr, _ = np.histogram(cur_data, bins=20, density=True)
            
            # 참조 히스토그램 가져오기
            hist_ref = self.reference_stats[col]["hist"]
            
            # 0을 작은 값으로 대체하여 나눗셈 오류 방지
            hist_ref = np.maximum(hist_ref, 1e-10)
            hist_curr = np.maximum(hist_curr, 1e-10)
            
            # KL Divergence 계산
            kl_div = np.sum(hist_curr * np.log(hist_curr / hist_ref))
            
            # 결과 저장
            feature_kl_divs[col] = kl_div
            result.feature_contributions[col] = kl_div
        
        # 전체 드리프트 점수 계산 (평균 KL Divergence)
        if feature_kl_divs:
            avg_kl_div = np.mean(list(feature_kl_divs.values()))
            result.drift_score = avg_kl_div
            
            # p-value 추정 (KL Divergence에서 직접적인 p-value는 없음)
            # 경험적으로 KL Divergence > 0.5를 유의미한 차이로 간주
            result.p_value = np.exp(-avg_kl_div)  # 추정된 p-value
            
            # 드리프트 감지 여부 결정
            if avg_kl_div > 0.5:  # 임계값 0.5는 조정 가능
                result.detected = True
                
                # 가장 기여도가 높은 특성 식별
                most_contributing = max(result.feature_contributions.items(), key=lambda x: x[1])
                result.description = (
                    f"Drift detected (KL={avg_kl_div:.4f}), "
                    f"most significant feature: {most_contributing[0]} "
                    f"(contribution={most_contributing[1]:.4f})"
                )
            else:
                result.description = f"No drift detected (KL={avg_kl_div:.4f})"
        else:
            result.description = "Not enough valid features for drift detection"
        
        # 드리프트 이력에 추가
        self.drift_history.append((datetime.now(), result))
        
        return result
    
    def get_drift_history(self) -> List[Tuple[datetime, DriftDetectionResult]]:
        """
        드리프트 감지 이력 조회
        
        Returns:
            (타임스탬프, 감지 결과) 튜플 목록
        """
        return self.drift_history.copy()
    
    def reset_window(self) -> None:
        """현재 윈도우 데이터 초기화"""
        self.current_window.clear()
    
    def update_reference_data(self, new_reference_data: pd.DataFrame) -> None:
        """
        참조 데이터 업데이트
        
        Args:
            new_reference_data: 새 참조 데이터
        """
        self.reference_data = new_reference_data[self.feature_columns].copy()
        self.reference_stats = self._compute_reference_stats()
        
        logger.info("참조 데이터 및 통계 업데이트 완료")


class AnomalyDetector:
    """
    이상 감지 클래스.
    실시간 데이터에서 이상 패턴이나 특이점을 식별합니다.
    """
    
    def __init__(self, 
                normal_data: pd.DataFrame,
                feature_columns: List[str],
                window_size: int = 20,
                contamination: float = 0.05,
                method: str = "isolation_forest"):
        """
        AnomalyDetector 초기화
        
        Args:
            normal_data: 정상 데이터
            feature_columns: 분석할 특성 열 이름
            window_size: 감지 윈도우 크기
            contamination: 이상치 비율 예상치
            method: 이상 감지 방법 ('isolation_forest', 'zscore', 'robust_zscore')
        """
        self.normal_data = normal_data[feature_columns].copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.contamination = contamination
        self.method = method
        
        # 현재 윈도우 데이터
        self.current_window = deque(maxlen=window_size)
        
        # 이상 감지 이력
        self.anomaly_history = []
        
        # 감지기 초기화
        self._initialize_detector()
        
        logger.info(f"AnomalyDetector 초기화 완료: {len(feature_columns)} 개 특성, 방법: {method}")
    
    def _initialize_detector(self) -> None:
        """이상 감지기 초기화"""
        if self.method == "isolation_forest":
            # Isolation Forest 모델 학습
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.detector.fit(self.normal_data[self.feature_columns])
            
        elif self.method == "zscore" or self.method == "robust_zscore":
            # 통계 계산
            if self.method == "zscore":
                self.feature_means = self.normal_data[self.feature_columns].mean()
                self.feature_stds = self.normal_data[self.feature_columns].std()
            else:  # robust_zscore
                self.feature_medians = self.normal_data[self.feature_columns].median()
                self.feature_mads = self.normal_data[self.feature_columns].apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                # MAD가 0인 경우 작은 값으로 대체
                self.feature_mads = self.feature_mads.replace(0, 1e-6)
    
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """
        새로운 관측 데이터 추가
        
        Args:
            observation: 관측 데이터 (feature_columns에 있는 키를 포함해야 함)
        """
        # 특성 추출
        features = {col: observation.get(col, np.nan) for col in self.feature_columns}
        
        # 현재 윈도우에 추가
        self.current_window.append(features)
    
    def detect_anomalies(self) -> List[AnomalyDetectionResult]:
        """
        현재 윈도우에서 이상 감지
        
        Returns:
            이상 감지 결과 목록
        """
        # 현재 윈도우가 충분히 차 있는지 확인
        if len(self.current_window) < 3:  # 최소 3개 관측치 필요
            return [AnomalyDetectionResult(
                detected=False,
                description="Not enough data in current window"
            )]
        
        # 현재 윈도우 데이터를 DataFrame으로 변환
        current_df = pd.DataFrame(list(self.current_window))
        
        # 방법에 따른 이상 감지
        if self.method == "isolation_forest":
            return self._detect_anomalies_isolation_forest(current_df)
        elif self.method == "zscore":
            return self._detect_anomalies_zscore(current_df)
        elif self.method == "robust_zscore":
            return self._detect_anomalies_robust_zscore(current_df)
        else:
            logger.warning(f"알 수 없는 이상 감지 방법: {self.method}")
            return [AnomalyDetectionResult(
                detected=False,
                description=f"Unknown anomaly detection method: {self.method}"
            )]
    
    def _detect_anomalies_isolation_forest(self, current_df: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Isolation Forest를 사용한 이상 감지
        
        Args:
            current_df: 현재 윈도우 데이터
            
        Returns:
            이상 감지 결과 목록
        """
        results = []
        
        try:
            # 결측치 처리
            df_filled = current_df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')
            
            # 이상 점수 계산 (-1: 이상, 1: 정상)
            anomaly_scores = self.detector.decision_function(df_filled)
            predictions = self.detector.predict(df_filled)  # -1: 이상, 1: 정상
            
            # 각 관측치에 대해 결과 생성
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1:  # 이상 감지
                    # 영향받은 특성 식별
                    observation = df_filled.iloc[i]
                    feature_scores = {}
                    
                    for col in self.feature_columns:
                        col_data = self.normal_data[col].dropna()
                        z_score = (observation[col] - col_data.mean()) / (col_data.std() + 1e-9)
                        feature_scores[col] = abs(z_score)
                    
                    # 주요 기여 특성 (z-score 기준 상위 3개)
                    affected_features = sorted(
                        feature_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    result = AnomalyDetectionResult(
                        detected=True,
                        anomaly_score=-score,  # 점수 부호 반전 (높을수록 이상)
                        anomaly_type="point_anomaly",
                        affected_features=[f[0] for f in affected_features],
                        timestamp=datetime.now(),
                        description=(
                            f"Anomaly detected (score={-score:.4f}), "
                            f"affected features: {', '.join([f'{n} ({s:.2f})' for n, s in affected_features])}"
                        )
                    )
                    
                    results.append(result)
                    self.anomaly_history.append((datetime.now(), result))
            
            # 이상이 없는 경우
            if not results:
                result = AnomalyDetectionResult(
                    detected=False,
                    timestamp=datetime.now(),
                    description="No anomalies detected"
                )
                results.append(result)
        
        except Exception as e:
            logger.error(f"Isolation Forest 이상 감지 중 오류: {e}")
            results.append(AnomalyDetectionResult(
                detected=False,
                timestamp=datetime.now(),
                description=f"Error during anomaly detection: {str(e)}"
            ))
        
        return results
    
    def _detect_anomalies_zscore(self, current_df: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Z-score를 사용한 이상 감지
        
        Args:
            current_df: 현재 윈도우 데이터
            
        Returns:
            이상 감지 결과 목록
        """
        results = []
        threshold = 3.0  # 임계값 (3-sigma)
        
        try:
            # 결측치 처리
            df_filled = current_df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')
            
            # 각 행에 대해 Z-score 계산
            for i, row in df_filled.iterrows():
                # 각 특성의 Z-score 계산
                z_scores = {}
                for col in self.feature_columns:
                    mean = self.feature_means[col]
                    std = self.feature_stds[col]
                    if std > 0:
                        z_scores[col] = abs((row[col] - mean) / std)
                    else:
                        z_scores[col] = 0
                
                # 임계값을 초과하는 특성 확인
                outlier_features = {col: z for col, z in z_scores.items() if z > threshold}
                
                if outlier_features:
                    # 이상 점수 = 최대 Z-score
                    anomaly_score = max(outlier_features.values())
                    
                    # 주요 기여 특성 (z-score 기준 상위 3개)
                    affected_features = sorted(
                        outlier_features.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    result = AnomalyDetectionResult(
                        detected=True,
                        anomaly_score=anomaly_score,
                        anomaly_type="zscore_outlier",
                        affected_features=[f[0] for f in affected_features],
                        timestamp=datetime.now(),
                        description=(
                            f"Outlier detected (max z-score={anomaly_score:.4f}), "
                            f"affected features: {', '.join([f'{n} ({s:.2f})' for n, s in affected_features])}"
                        )
                    )
                    
                    results.append(result)
                    self.anomaly_history.append((datetime.now(), result))
            
            # 이상이 없는 경우
            if not results:
                result = AnomalyDetectionResult(
                    detected=False,
                    timestamp=datetime.now(),
                    description="No anomalies detected"
                )
                results.append(result)
        
        except Exception as e:
            logger.error(f"Z-score 이상 감지 중 오류: {e}")
            results.append(AnomalyDetectionResult(
                detected=False,
                timestamp=datetime.now(),
                description=f"Error during anomaly detection: {str(e)}"
            ))
        
        return results
    
    def _detect_anomalies_robust_zscore(self, current_df: pd.DataFrame) -> List[AnomalyDetectionResult]:
        """
        Robust Z-score (median, MAD)를 사용한 이상 감지
        
        Args:
            current_df: 현재 윈도우 데이터
            
        Returns:
            이상 감지 결과 목록
        """
        results = []
        threshold = 3.5  # 임계값 (MAD 기반)
        
        try:
            # 결측치 처리
            df_filled = current_df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')
            
            # 각 행에 대해 Robust Z-score 계산
            for i, row in df_filled.iterrows():
                # 각 특성의 Robust Z-score 계산
                robust_z_scores = {}
                for col in self.feature_columns:
                    median = self.feature_medians[col]
                    mad = self.feature_mads[col]
                    if mad > 0:
                        # 0.6745는 정규 분포에서 MAD를 표준 편차 추정치로 변환하는 상수
                        robust_z_scores[col] = abs((row[col] - median) / (mad * 1.4826))
                    else:
                        robust_z_scores[col] = 0
                
                # 임계값을 초과하는 특성 확인
                outlier_features = {col: z for col, z in robust_z_scores.items() if z > threshold}
                
                if outlier_features:
                    # 이상 점수 = 최대 Robust Z-score
                    anomaly_score = max(outlier_features.values())
                    
                    # 주요 기여 특성 (z-score 기준 상위 3개)
                    affected_features = sorted(
                        outlier_features.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    result = AnomalyDetectionResult(
                        detected=True,
                        anomaly_score=anomaly_score,
                        anomaly_type="robust_zscore_outlier",
                        affected_features=[f[0] for f in affected_features],
                        timestamp=datetime.now(),
                        description=(
                            f"Outlier detected (max robust z-score={anomaly_score:.4f}), "
                            f"affected features: {', '.join([f'{n} ({s:.2f})' for n, s in affected_features])}"
                        )
                    )
                    
                    results.append(result)
                    self.anomaly_history.append((datetime.now(), result))
            
            # 이상이 없는 경우
            if not results:
                result = AnomalyDetectionResult(
                    detected=False,
                    timestamp=datetime.now(),
                    description="No anomalies detected"
                )
                results.append(result)
        
        except Exception as e:
            logger.error(f"Robust Z-score 이상 감지 중 오류: {e}")
            results.append(AnomalyDetectionResult(
                detected=False,
                timestamp=datetime.now(),
                description=f"Error during anomaly detection: {str(e)}"
            ))
        
        return results
    
    def get_anomaly_history(self) -> List[Tuple[datetime, AnomalyDetectionResult]]:
        """
        이상 감지 이력 조회
        
        Returns:
            (타임스탬프, 감지 결과) 튜플 목록
        """
        return self.anomaly_history.copy()
    
    def reset_window(self) -> None:
        """현재 윈도우 데이터 초기화"""
        self.current_window.clear()
    
    def update_normal_data(self, new_normal_data: pd.DataFrame) -> None:
        """
        정상 데이터 업데이트 및 감지기 재학습
        
        Args:
            new_normal_data: 새 정상 데이터
        """
        self.normal_data = new_normal_data[self.feature_columns].copy()
        self._initialize_detector()
        
        logger.info("정상 데이터 업데이트 및 감지기 재학습 완료")
