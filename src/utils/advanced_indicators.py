import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
import statsmodels.api as sm
import talib


def calculate_advanced_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    고급 기술적 지표 계산
    
    Args:
        data: OHLCV 데이터가 포함된 DataFrame
             (Open, High, Low, Close, Volume 컬럼 필요)
             
    Returns:
        df: 계산된 고급 지표가 추가된 DataFrame
    """
    # 기본 데이터 추출
    df = pd.DataFrame(index=data.index)
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    volume = data['Volume']
    
    # 기존 지표 계산 (참고용)
    df = calculate_standard_indicators(data, df)
    
    # ===== 고급 모멘텀 지표 =====
    
    # 1. 적응형 RSI (ARSI)
    volatility = calculate_rolling_volatility(close, window=14)
    alpha = 2 / (1 + np.exp(volatility * 5)) # 변동성에 따른 적응형 알파
    df['ARSI'] = calculate_adaptive_rsi(close, alpha)
    
    # 2. Fisher Transform 적용 RSI
    df['FisherRSI'] = fisher_transform(df['RSI'] / 100.0)
    
    # 3. KST (Know Sure Thing) 오실레이터
    df['KST'] = calculate_kst(close)
    df['KSTSignal'] = df['KST'].rolling(window=9).mean()
    
    # 4. Relative Momentum Index
    df['RMI'] = calculate_rmi(close, period=20, momentum_period=5)
    
    # ===== 고급 변동성 지표 =====
    
    # 1. KAMA (Kaufman's Adaptive Moving Average)
    df['KAMA'] = talib.KAMA(close.values, timeperiod=14)
    
    # 2. MESA 적응형 이동평균 (대체 버전)
    df['MAMA'], df['FAMA'] = calculate_mesa_adaptive_ma(close)
    
    # 3. 임계값 변동성 (Threshold Volatility)
    df['ThresholdVol'] = calculate_threshold_volatility(close, window=20)
    
    # 4. Yang Zhang 변동성
    df['YangZhangVol'] = calculate_yang_zhang_volatility(open_price, high, low, close, window=20)
    
    # 5. 비선형 변동성 예측기 (GARCH 기반)
    df['GARCHVol'] = calculate_garch_volatility(close)
    
    # ===== 고급 추세 지표 =====
    
    # 1. TTM 추세 (TrendScore)
    df['TTMTrend'] = calculate_ttm_trend(close, high, low)
    
    # 2. 어댑티브 트렌드 라인 (Adaptive Trend Line)
    df['AdaptiveTL'] = calculate_adaptive_trendline(close)
    
    # 3. ZLEMA (Zero Lag Exponential Moving Average)
    df['ZLEMA'] = calculate_zlema(close, period=20)
    
    # 4. Impulse System
    df['Impulse'] = calculate_impulse_system(close)
    
    # ===== 고급 패턴 지표 =====
    
    # 1. Squeeze Momentum
    df['SqueezeMomentum'] = calculate_squeeze_momentum(close, high, low)
    
    # 2. 변동성 크런치 (Volatility Crunch)
    df['VolCrunch'] = calculate_volatility_crunch(close)
    
    # 3. 웨이브 추세 오실레이터 (Wave Trend Oscillator)
    df['WaveTrend1'], df['WaveTrend2'] = calculate_wave_trend(close, high, low)
    
    # ===== 거래량 기반 고급 지표 =====
    
    # 1. 상대 볼륨 가중 RSI
    df['RVWRSI'] = calculate_volume_weighted_rsi(close, volume)
    
    # 2. 머니 플로우 인덱스 (Money Flow Index) 개선버전
    df['EnhancedMFI'] = calculate_enhanced_mfi(close, high, low, volume)
    
    # 3. 적응형 복합 볼륨 지수 (Adaptive Composite Volume Index)
    df['ACVI'] = calculate_adaptive_composite_volume(close, volume)
    
    # ===== 머신러닝 기반 파생 지표 =====
    
    # 1. 국소 이상치 점수 (LOF 기반)
    df['LocalOutlierScore'] = calculate_local_outlier_score(close, volume)
    
    # 2. 클러스터 멤버십 점수
    df['ClusterScore'] = calculate_cluster_membership(close, high, low, volume)
    
    # 3. 자기회귀 잔차 (예측 오류)
    df['ARResiduals'] = calculate_ar_residuals(close)
    
    # ===== 금융공학 지표 =====
    
    # 1. 임플라이드 변동성 프록시
    df['ImpliedVolProxy'] = calculate_implied_vol_proxy(close, window=30)
    
    # 2. 조건부 분위 지표 (Conditional Quantile Indicator)
    df['CondQuantile'] = calculate_conditional_quantile(close)
    
    # 3. 시장 효율성 비율 (Market Efficiency Ratio)
    df['MarketEfficiency'] = calculate_market_efficiency_ratio(close, window=20)
    
    # ===== 멀티 타임프레임 지표 =====
    
    # 1. 타임프레임 불일치 지표 (Timeframe Divergence)
    df['TFDivergence'] = calculate_timeframe_divergence(close)
    
    # 2. 멀티 타임프레임 모멘텀
    df['MTFMomentum'] = calculate_mtf_momentum(close)
    
    # ===== 시간적 특성 =====
    
    # 1. 일중 패턴 (Intraday Pattern - 일별 데이터에서는 무시)
    if 'Time' in data.columns or isinstance(data.index, pd.DatetimeIndex):
        df['IntradayPattern'] = calculate_intraday_pattern(data)
    
    # 2. 캘린더 특성 (Calendar Features)
    if isinstance(data.index, pd.DatetimeIndex):
        calendar_features = calculate_calendar_features(data.index)
        for col, values in calendar_features.items():
            df[col] = values
    
    # ===== 정규화 =====
    
    # Z-점수 정규화
    for col in df.columns:
        if col not in ['IntradayPattern'] and not col.startswith('Calendar_'):
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-9)
    
    # 결측치 처리
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def calculate_standard_indicators(data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """기본 기술적 지표 계산 (참고용)"""
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    delta = close.diff()
    
    # RSI
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Force Index
    df['ForceIndex2'] = close.diff(2) * volume
    
    # Stochastic Oscillator
    low_min = low.rolling(window=14).min()
    high_max = high.rolling(window=14).max()
    df['%K'] = 100 * (close - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACDSignal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    window = 20
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    df['SMA20'] = sma
    df['BBUpper'] = sma + (std * 2)
    df['BBLower'] = sma - (std * 2)
    df['BBWidth'] = (df['BBUpper'] - df['BBLower']) / df['SMA20']
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # OBV
    df['OBV'] = (volume * (np.sign(delta))).cumsum()
    
    # ROC
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    
    return df


# ===== 고급 지표 계산 함수들 =====

def calculate_rolling_volatility(series: pd.Series, window: int = 14) -> pd.Series:
    """롤링 변동성 (표준 편차) 계산"""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window).std()


def calculate_adaptive_rsi(series: pd.Series, alpha: Union[float, pd.Series]) -> pd.Series:
    """적응형 알파를 사용한 RSI 계산"""
    delta = series.diff()
    
    # 초기 값 설정
    up = delta.copy()
    up[up < 0] = 0
    down = -delta.copy()
    down[down < 0] = 0
    
    # 적응형 평균
    if isinstance(alpha, pd.Series):
        # 벡터화된 연산 불가능, 루프 사용
        avg_up = pd.Series(index=up.index, dtype=float)
        avg_down = pd.Series(index=down.index, dtype=float)
        
        # 초기값 설정
        avg_up.iloc[0] = 0
        avg_down.iloc[0] = 0
        
        for i in range(1, len(up)):
            a = alpha.iloc[i] if not np.isnan(alpha.iloc[i]) else 0.1
            avg_up.iloc[i] = (1-a) * avg_up.iloc[i-1] + a * up.iloc[i]
            avg_down.iloc[i] = (1-a) * avg_down.iloc[i-1] + a * down.iloc[i]
    else:
        # 스칼라 알파, 벡터화된 연산 가능
        avg_up = up.ewm(alpha=alpha, adjust=False).mean()
        avg_down = down.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_up / (avg_down + 1e-9)  # 0으로 나누기 방지
    return 100 - (100 / (1 + rs))


def fisher_transform(series: pd.Series) -> pd.Series:
    """Fisher 변환 적용"""
    # [-1, 1] 범위로 제한
    clamped = series.clip(-0.999, 0.999)
    return 0.5 * np.log((1 + clamped) / (1 - clamped))


def calculate_kst(close: pd.Series) -> pd.Series:
    """KST (Know Sure Thing) 오실레이터 계산"""
    # 다양한 기간의 ROC (Rate of Change) 계산
    roc10 = ((close / close.shift(10)) - 1) * 100
    roc15 = ((close / close.shift(15)) - 1) * 100
    roc20 = ((close / close.shift(20)) - 1) * 100
    roc30 = ((close / close.shift(30)) - 1) * 100
    
    # 각 ROC의 이동 평균 계산
    ma10 = roc10.rolling(window=10).mean()
    ma15 = roc15.rolling(window=10).mean()
    ma20 = roc20.rolling(window=10).mean()
    ma30 = roc30.rolling(window=15).mean()
    
    # KST 계산 (가중치 적용)
    kst = ma10 + ma15 * 2 + ma20 * 3 + ma30 * 4
    
    return kst


def calculate_rmi(close: pd.Series, period: int = 20, momentum_period: int = 5) -> pd.Series:
    """Relative Momentum Index 계산"""
    # 모멘텀 계산
    momentum = close - close.shift(momentum_period)
    
    # 업 모멘텀과 다운 모멘텀
    up_momentum = momentum.copy()
    up_momentum[up_momentum < 0] = 0
    down_momentum = -momentum.copy()
    down_momentum[down_momentum < 0] = 0
    
    # 이동 평균
    avg_up = up_momentum.rolling(window=period).mean()
    avg_down = down_momentum.rolling(window=period).mean()
    
    # RMI 계산
    rmi = 100 - (100 / (1 + (avg_up / (avg_down + 1e-9))))
    
    return rmi


def calculate_mesa_adaptive_ma(close: pd.Series) -> tuple:
    """
    MESA 적응형 이동평균 (대체 구현)
    
    Returns:
        tuple: (MAMA, FAMA) - MESA 적응형 이동평균과 빠른 MAMA
    """
    # 대체 구현: 힐버트 변환 대신 위상 가중 이동평균 사용
    alpha = 0.0962
    fast_limit = 0.5
    slow_limit = 0.05
    
    # 필요한 중간 계산
    smooth = close.rolling(window=9).mean()
    detrender = pd.Series(0, index=close.index)
    smooth_period = pd.Series(0, index=close.index)
    phase = pd.Series(0, index=close.index)
    
    # 위상 계산 (단순화된 버전)
    for i in range(12, len(close)):
        detrender.iloc[i] = (0.0962 * close.iloc[i] + 0.5769 * close.iloc[i-2] - 0.5769 * close.iloc[i-4] - 0.0962 * close.iloc[i-6]) * (0.075 * smooth_period.iloc[i-1] + 0.54)
        
        # 단기 및 장기 주기 계산
        smooth_period_val = (detrender.iloc[i] ** 2 + detrender.iloc[i-1] ** 2).clip(1e-9) ** 0.5
        smooth_period.iloc[i] = 0.2 * smooth_period_val + 0.8 * smooth_period.iloc[i-1]
        
        phase_val = np.arctan2(detrender.iloc[i-3], detrender.iloc[i]) / (2 * np.pi)
        phase.iloc[i] = 0.1 * phase_val + 0.9 * phase.iloc[i-1]
    
    # 위상 기반 적응형 알파 계산
    dphase = phase.diff().abs()
    alpha = fast_limit * dphase * 2 + slow_limit
    alpha = alpha.clip(slow_limit, fast_limit)
    
    # MAMA 및 FAMA 계산
    mama = pd.Series(0, index=close.index)
    fama = pd.Series(0, index=close.index)
    
    mama.iloc[12] = close.iloc[12]
    fama.iloc[12] = close.iloc[12]
    
    for i in range(13, len(close)):
        mama.iloc[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * mama.iloc[i-1]
        fama.iloc[i] = 0.5 * alpha.iloc[i] * mama.iloc[i] + (1 - 0.5 * alpha.iloc[i]) * fama.iloc[i-1]
    
    return mama, fama


def calculate_threshold_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """임계값 변동성 - 지정된 임계값을 초과하는 가격 변화만 고려한 변동성"""
    returns = close.pct_change()
    abs_returns = returns.abs()
    
    # 임계값 계산 (롤링 평균 변동의 1.5배)
    threshold = abs_returns.rolling(window=window).mean() * 1.5
    
    # 임계값 초과 변동만 추출
    filtered_returns = returns.copy()
    mask = abs_returns < threshold
    filtered_returns[mask] = 0
    
    # 임계값 변동성 계산 (롤링 제곱합의 제곱근)
    threshold_vol = (filtered_returns ** 2).rolling(window=window).sum() ** 0.5
    
    return threshold_vol


def calculate_yang_zhang_volatility(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Yang-Zhang 변동성 추정기
    주간 변동(overnight volatility)과 장중 변동(trading hours volatility)을 조합
    """
    # 수익률 계산
    close_to_close = np.log(close / close.shift(1))
    open_to_open = np.log(open_price / open_price.shift(1))
    overnight_jump = np.log(open_price / close.shift(1))
    
    # 주간 변동성 (overnight volatility)
    overnight_vol = overnight_jump.rolling(window=window).var()
    
    # 장중 변동성 (trading hours volatility) - 단순화된 Garman-Klass 변동성 사용
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)
    trading_hours_vol = (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(window=window).mean()
    
    # Yang-Zhang 변동성 (주간 + 장중 변동성의 가중합)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yang_zhang_vol = overnight_vol + k * trading_hours_vol
    
    # 연율화
    yang_zhang_vol = np.sqrt(252 * yang_zhang_vol)
    
    return yang_zhang_vol


def calculate_garch_volatility(close: pd.Series, p: int = 1, q: int = 1) -> pd.Series:
    """
    GARCH 모델을 사용한 변동성 예측
    단순화된 버전으로 제공
    """
    # 로그 수익률
    log_returns = np.log(close / close.shift(1)).dropna()
    
    try:
        # GARCH 모델 피팅
        model = sm.tsa.GARCH(log_returns, vol='GARCH', p=p, q=q)
        results = model.fit(disp='off')
        
        # 조건부 변동성 예측
        conditional_vol = results.conditional_volatility
        
        # 원래 시리즈 길이에 맞게 조정
        garch_vol = pd.Series(index=close.index, dtype=float)
        garch_vol.iloc[log_returns.index.get_indexer(log_returns.index)] = conditional_vol
        
        # 앞쪽 결측치 처리
        garch_vol = garch_vol.ffill().bfill()
        
        # 연율화
        garch_vol = np.sqrt(252) * garch_vol
        
    except:
        # 모델 피팅 실패 시 단순 변동성 사용
        log_returns = np.log(close / close.shift(1))
        garch_vol = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    return garch_vol


def calculate_ttm_trend(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """
    TTM 추세 점수 (TrendScore)
    다양한 시간대의 이동평균을 결합한 추세 강도 측정
    """
    # 다양한 기간의 이동평균
    ma_periods = [8, 21, 34, 55, 89, 144]
    mas = [close.rolling(window=p).mean() for p in ma_periods]
    
    # 추세 점수 계산
    trend_score = pd.Series(0, index=close.index)
    
    # 1. 단기 추세: MA 교차 기반
    for i in range(len(ma_periods)-1):
        # 단기 MA가 장기 MA를 상회하면 양수 점수, 그 반대는 음수 점수
        trend_score += np.sign(mas[i] - mas[i+1])
    
    # 2. 볼린저 밴드 기반 점수
    bb_window = 20
    bb_sma = close.rolling(window=bb_window).mean()
    bb_std = close.rolling(window=bb_window).std()
    
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    
    # 가격이 상단/하단 밴드 대비 위치 (-1 ~ +1)
    bb_position = (close - bb_sma) / (bb_upper - bb_lower) * 2
    trend_score += bb_position
    
    # 3. ADX 추세 강도 (단순화된 버전)
    tr = pd.DataFrame([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ]).max()
    
    dx = ((high - high.shift(1)).abs() - (low - low.shift(1)).abs()).abs() / tr
    adx = dx.rolling(window=14).mean()
    
    # ADX를 0-1 범위로 정규화하고 추세 점수에 추가
    trend_score += adx / adx.rolling(window=100).max() * 2
    
    # 최종 점수 정규화 (-100 ~ +100)
    max_abs = trend_score.abs().rolling(window=100).max().clip(lower=1)
    normalized_trend = 100 * trend_score / max_abs
    
    return normalized_trend


def calculate_adaptive_trendline(close: pd.Series) -> pd.Series:
    """
    적응형 추세선
    가격 모멘텀과 변동성에 적응하는 동적 추세선
    """
    # 가격 모멘텀 계산
    momentum = close.pct_change(periods=5).rolling(window=10).mean()
    
    # 변동성 계산 
    volatility = close.pct_change().rolling(window=20).std()
    
    # 변동성 기반 알파 계산 (높은 변동성 = 낮은 알파 = 느린 적응)
    adaptive_alpha = (0.2 / (volatility * 10 + 0.2)).clip(0.01, 0.3)
    
    # 추세선 계산
    trendline = pd.Series(0, index=close.index)
    trendline.iloc[0] = close.iloc[0]
    
    for i in range(1, len(close)):
        alpha = adaptive_alpha.iloc[i] if not np.isnan(adaptive_alpha.iloc[i]) else 0.1
        trendline.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * trendline.iloc[i-1]
        
        # 모멘텀에 따른 추세 가속/감속
        if momentum.iloc[i] > 0:
            # 상승 추세에서 가속
            trendline.iloc[i] += momentum.iloc[i] * volatility.iloc[i] * close.iloc[i] * 0.1
        elif momentum.iloc[i] < 0:
            # 하락 추세에서 가속
            trendline.iloc[i] += momentum.iloc[i] * volatility.iloc[i] * close.iloc[i] * 0.1
    
    return trendline


def calculate_zlema(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Zero-Lag Exponential Moving Average (ZLEMA)
    지연을 최소화한 이동평균
    """
    lag = (period - 1) // 2
    
    # 에러 수정 (lag 값 제한)
    if lag <= 0:
        lag = 1
    
    # EMA 지연 보정항 계산
    ema_data = 2 * close - close.shift(lag)
    
    # ZLEMA 계산
    alpha = 2 / (period + 1)
    zlema = pd.Series(index=close.index, dtype=float)
    zlema.iloc[lag] = ema_data.iloc[lag]
    
    for i in range(lag + 1, len(close)):
        if np.isnan(ema_data.iloc[i]):
            zlema.iloc[i] = zlema.iloc[i-1]
        else:
            zlema.iloc[i] = alpha * ema_data.iloc[i] + (1 - alpha) * zlema.iloc[i-1]
    
    return zlema


def calculate_impulse_system(close: pd.Series) -> pd.Series:
    """
    Impulse System
    추세 방향과 모멘텀을 결합한 시스템
    """
    # EMA 계산
    ema13 = close.ewm(span=13).mean()
    ema8 = close.ewm(span=8).mean()
    
    # MACD 계산
    macd = ema8 - ema13
    macd_signal = macd.ewm(span=9).mean()
    
    # 임펄스 계산 (방향 + 모멘텀)
    impulse = pd.Series(0, index=close.index)
    
    # 상승 임펄스: 가격 > EMA13 및 MACD 상승
    long_impulse = (close > ema13) & (macd > macd_signal)
    
    # 하락 임펄스: 가격 < EMA13 및 MACD 하락
    short_impulse = (close < ema13) & (macd < macd_signal)
    
    # 임펄스 값 설정
    impulse[long_impulse] = 1
    impulse[short_impulse] = -1
    
    return impulse


def calculate_squeeze_momentum(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Squeeze Momentum Indicator
    존 카터의 TTM Squeeze 지표
    """
    # 볼린저 밴드 계산
    bb_length = 20
    bb_mult = 2
    
    sma = close.rolling(window=bb_length).mean()
    bb_std = close.rolling(window=bb_length).std()
    
    bb_upper = sma + bb_mult * bb_std
    bb_lower = sma - bb_mult * bb_std
    
    # 켈트너 채널 계산
    kc_length = 20
    kc_mult = 1.5
    
    tr = pd.DataFrame([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ]).max()
    
    atr = tr.rolling(window=kc_length).mean()
    
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr
    
    # 스퀴즈 감지 (볼린저 밴드가 켈트너 채널 내에 있을 때)
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # 스퀴즈 모멘텀 계산
    # 린우드 방식: 현재 가격 - 20일 이동평균의 중간값
    highest_high = high.rolling(window=kc_length).max()
    lowest_low = low.rolling(window=kc_length).min()
    
    mid = (highest_high + lowest_low) / 2
    momentum = close - (mid + sma) / 2
    
    # 스퀴즈 해제 시 모멘텀 증폭
    result = momentum.copy()
    result[~squeeze_on] = result[~squeeze_on] * 1.5
    
    return result


def calculate_volatility_crunch(close: pd.Series) -> pd.Series:
    """
    변동성 크런치 지표
    변동성이 급격히 감소하는 패턴 감지
    """
    # 장기 및 단기 변동성 계산
    vol_short = close.pct_change().rolling(window=5).std() * 100
    vol_long = close.pct_change().rolling(window=20).std() * 100
    
    # 변동성 비율
    vol_ratio = vol_short / vol_long
    
    # 변동성 크런치 계산 (단기 변동성이 장기 대비 큰 폭 감소시)
    crunch = 1 - vol_ratio
    
    # 크런치 강도 (z-점수 기반)
    crunch_z = (crunch - crunch.rolling(window=60).mean()) / crunch.rolling(window=60).std()
    
    return crunch_z


def calculate_wave_trend(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple:
    """
    Wave Trend Oscillator
    복합적인 방향성 지표
    """
    # 파라미터
    n1 = 10  # 에너지 채널 주기
    n2 = 21  # 평균 주기
    
    # 평균 가격
    ap = (high + low + close) / 3
    
    # 에너지 채널값 계산
    esa = ap.ewm(span=n1).mean()
    d = (ap - esa).abs().ewm(span=n1).mean()
    ci = (ap - esa) / (0.015 * d)
    
    # Wave Trend 계산
    wt1 = ci.ewm(span=n2).mean()
    wt2 = wt1.rolling(window=4).mean()
    
    return wt1, wt2


def calculate_volume_weighted_rsi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    상대 볼륨 가중 RSI
    볼륨을 고려한 RSI 변형
    """
    # 상대 볼륨 계산 (현재 볼륨 / 평균 볼륨)
    rel_volume = volume / volume.rolling(window=20).mean()
    
    # 가격 변화
    change = close.diff()
    
    # 볼륨 가중 상승/하락
    gain = change.copy()
    gain[gain < 0] = 0
    gain = gain * rel_volume
    
    loss = -change.copy()
    loss[loss < 0] = 0
    loss = loss * rel_volume
    
    # RSI 계산
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)  # 0으로 나누기 방지
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_enhanced_mfi(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """
    향상된 Money Flow Index
    전통적인 MFI에 비선형 변형 적용
    """
    # 일반적인 MFI 계산
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    money_flow_positive = raw_money_flow.copy()
    money_flow_positive[typical_price < typical_price.shift(1)] = 0
    
    money_flow_negative = raw_money_flow.copy()
    money_flow_negative[typical_price > typical_price.shift(1)] = 0
    
    mf_ratio = (money_flow_positive.rolling(window=14).sum() / 
               (money_flow_negative.rolling(window=14).sum() + 1e-9))
    
    mfi = 100 - (100 / (1 + mf_ratio))
    
    # 향상된 MFI - 비선형 변형
    # 1. 볼륨 스파이크 영향 감소
    vol_percentile = volume.rolling(window=50).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
    )
    
    # 2. 가격 변화량에 따른 가중치
    price_change = typical_price.pct_change().abs()
    price_change_weight = price_change / price_change.rolling(window=20).max()
    
    # 3. 비선형 변형 (극단 영역에서 더 빠른 반응)
    enhanced_mfi = mfi.copy()
    
    # 과매수 영역 (70-100)의 값 압축
    mask_overbought = (mfi > 70)
    enhanced_mfi[mask_overbought] = 70 + (mfi[mask_overbought] - 70) ** 0.5 * 30 ** 0.5
    
    # 과매도 영역 (0-30)의 값 압축
    mask_oversold = (mfi < 30)
    enhanced_mfi[mask_oversold] = 30 - (30 - mfi[mask_oversold]) ** 0.5 * 30 ** 0.5
    
    # 볼륨 및 가격 변화 스케일링 적용
    enhanced_mfi = enhanced_mfi * (0.5 + 0.5 * vol_percentile) * (0.5 + 0.5 * price_change_weight)
    
    return enhanced_mfi


def calculate_adaptive_composite_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    적응형 복합 볼륨 지수
    다양한 볼륨 지표를 조합한 종합 지표
    """
    # 1. 일반 OBV (On Balance Volume)
    change = close.diff()
    obv = (volume * np.sign(change)).cumsum()
    
    # 2. 역변동성 가중 볼륨 (변동성이 낮을 때 볼륨의 중요도 증가)
    volatility = close.pct_change().rolling(window=20).std()
    inv_vol_weight = 1 / (volatility + 0.001)
    inv_vol_weight = inv_vol_weight / inv_vol_weight.rolling(window=60).mean()
    vwobv = (volume * np.sign(change) * inv_vol_weight).cumsum()
    
    # 3. 추세 방향 가중 볼륨
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    trend = ema20 / ema50 - 1
    trend_weight = trend / trend.abs().rolling(window=60).max().clip(lower=0.01)
    twobv = (volume * np.sign(change) * (1 + trend_weight)).cumsum()
    
    # 4. 볼륨 서프라이즈 지표
    vol_mean = volume.rolling(window=20).mean()
    vol_std = volume.rolling(window=20).std()
    vol_surprise = (volume - vol_mean) / (vol_std + 1)
    vs_obv = (volume * np.sign(change) * (1 + vol_surprise * 0.2)).cumsum()
    
    # 지표 조합
    # 1. 각 지표 정규화
    norm_obv = (obv - obv.rolling(window=100).min()) / (obv.rolling(window=100).max() - obv.rolling(window=100).min() + 1e-9)
    norm_vwobv = (vwobv - vwobv.rolling(window=100).min()) / (vwobv.rolling(window=100).max() - vwobv.rolling(window=100).min() + 1e-9)
    norm_twobv = (twobv - twobv.rolling(window=100).min()) / (twobv.rolling(window=100).max() - twobv.rolling(window=100).min() + 1e-9)
    norm_vs_obv = (vs_obv - vs_obv.rolling(window=100).min()) / (vs_obv.rolling(window=100).max() - vs_obv.rolling(window=100).min() + 1e-9)
    
    # 2. 적응형 가중 결합
    # 가중치: 각 지표의 예측력에 비례 (단순화 버전)
    w_obv = 0.25
    w_vwobv = 0.25
    w_twobv = 0.25
    w_vs_obv = 0.25
    
    composite = (w_obv * norm_obv + 
                w_vwobv * norm_vwobv + 
                w_twobv * norm_twobv + 
                w_vs_obv * norm_vs_obv)
    
    return composite


def calculate_local_outlier_score(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    국소 이상치 점수
    간소화된 LOF(Local Outlier Factor) 기반 이상치 탐지
    """
    # 특성 준비
    returns = close.pct_change()
    log_volume = np.log1p(volume)
    
    # 정규화
    norm_returns = (returns - returns.rolling(window=30).mean()) / returns.rolling(window=30).std()
    norm_volume = (log_volume - log_volume.rolling(window=30).mean()) / log_volume.rolling(window=30).std()
    
    # 특성 결합
    features = pd.DataFrame({
        'returns': norm_returns,
        'volume': norm_volume
    })
    
    # 간소화된 LOF 계산 (정확한 LOF는 scikit-learn 필요)
    # 여기서는 단순화된 계산 사용
    
    # 1. 이동 평균과의 거리 계산
    mean_dist = ((features['returns'] ** 2) + (features['volume'] ** 2)) ** 0.5
    
    # 2. 편차의 롤링 표준편차와 비교
    std_dist = mean_dist.rolling(window=30).std()
    local_outlier_score = mean_dist / (std_dist + 0.1)
    
    return local_outlier_score


def calculate_cluster_membership(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """
    클러스터 멤버십 점수
    시장 패턴 클러스터링 기반 멤버십 점수
    """
    # 특성 추출
    returns = close.pct_change()
    high_low_ratio = (high - low) / close
    volume_change = volume.pct_change()
    
    # 정규화
    norm_returns = (returns - returns.rolling(window=30).mean()) / returns.rolling(window=30).std()
    norm_hl_ratio = (high_low_ratio - high_low_ratio.rolling(window=30).mean()) / high_low_ratio.rolling(window=30).std()
    norm_vol_change = (volume_change - volume_change.rolling(window=30).mean()) / volume_change.rolling(window=30).std()
    
    # 클러스터 프로토타입 (3개의 대표적 패턴)
    # 1. 강한 상승 트렌드
    c1 = np.array([1.5, 0.8, 1.0])
    
    # 2. 횡보 시장
    c2 = np.array([0.0, -0.5, -0.5])
    
    # 3. 강한 하락 트렌드
    c3 = np.array([-1.5, 1.5, 1.5])
    
    # 각 클러스터에 대한 거리 계산
    cluster_score = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if pd.notna(norm_returns.iloc[i]) and pd.notna(norm_hl_ratio.iloc[i]) and pd.notna(norm_vol_change.iloc[i]):
            # 현재 특성 벡터
            current = np.array([
                norm_returns.iloc[i], 
                norm_hl_ratio.iloc[i], 
                norm_vol_change.iloc[i]
            ])
            
            # 각 클러스터까지의 거리
            d1 = np.sum((current - c1) ** 2) ** 0.5
            d2 = np.sum((current - c2) ** 2) ** 0.5
            d3 = np.sum((current - c3) ** 2) ** 0.5
            
            # 가장 가까운 클러스터와 거리
            min_dist = min(d1, d2, d3)
            
            # 클러스터 멤버십 점수 (-1 ~ +1, -1: 하락 트렌드, 0: 횡보, +1: 상승 트렌드)
            if min_dist == d1:
                cluster_score.iloc[i] = 1.0 / (1.0 + min_dist * 0.5)
            elif min_dist == d2:
                cluster_score.iloc[i] = 0.0
            else:
                cluster_score.iloc[i] = -1.0 / (1.0 + min_dist * 0.5)
    
    # 점수 스무딩
    cluster_score = cluster_score.rolling(window=3).mean()
    
    return cluster_score


def calculate_ar_residuals(close: pd.Series) -> pd.Series:
    """
    자기회귀 모델 잔차
    AR(5) 모델의 예측 오류
    """
    log_returns = np.log(close / close.shift(1))
    
    # AR 모델 대신 간소화된 이동 평균 기반 예측 사용
    ma5 = log_returns.rolling(window=5).mean()
    ma20 = log_returns.rolling(window=20).mean()
    
    # 추세 기반 예측 (단순화된 AR(5) 대체)
    predicted = 2 * ma5 - ma20
    
    # 잔차 계산
    residuals = log_returns - predicted
    
    # 잔차 표준화
    std_residuals = residuals / residuals.rolling(window=30).std()
    
    return std_residuals


def calculate_implied_vol_proxy(close: pd.Series, window: int = 30) -> pd.Series:
    """
    임플라이드 변동성 프록시
    실제 옵션 데이터 없이 임플라이드 변동성 추정
    """
    # 실현 변동성 계산
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    
    # 변동성 예측 (EWMA 모델)
    lambda_param = 0.94
    ewma_vol = np.zeros(len(close))
    ewma_vol[0] = realized_vol.iloc[0] if not np.isnan(realized_vol.iloc[0]) else 0.15
    
    for i in range(1, len(close)):
        if np.isnan(log_returns.iloc[i-1]):
            ewma_vol[i] = ewma_vol[i-1]
        else:
            ewma_vol[i] = np.sqrt(lambda_param * ewma_vol[i-1]**2 + (1-lambda_param) * log_returns.iloc[i-1]**2) * np.sqrt(252)
    
    # 변동성 추세 및 스킴 측정
    vol_trend = realized_vol / realized_vol.shift(window // 2) - 1
    
    # 임플라이드 볼 프록시 = 실현 변동성 * (1 + 변동성 추세 * 0.5)
    iv_proxy = realized_vol * (1 + vol_trend * 0.5)
    
    # VIX와 유사한 스케일로 조정 (퍼센트)
    iv_proxy = iv_proxy * 100
    
    return iv_proxy


def calculate_conditional_quantile(close: pd.Series) -> pd.Series:
    """
    조건부 분위 지표
    현재 가격 움직임의 과거 분포 내 백분위
    """
    returns = close.pct_change()
    
    # 롤링 윈도우에서 현재 수익률의 백분위 계산
    cq = returns.rolling(window=100).apply(
        lambda x: stats.percentileofscore(x.iloc[:-1], x.iloc[-1]) / 100
        if len(x.iloc[:-1]) > 0 else 0.5
    )
    
    # 0.5 중심 조정 (0.5가 중립점)
    cq = cq * 2 - 1
    
    # 스무딩
    cq = cq.ewm(span=5).mean()
    
    return cq


def calculate_market_efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """
    시장 효율성 비율
    가격 이동 대비 순 변화의 비율
    """
    # 윈도우 내 순 가격 변화
    net_change = (close.shift(-window) / close - 1).abs()
    
    # 윈도우 내 총 가격 이동
    price_path = close.pct_change().abs().rolling(window=window).sum()
    
    # 효율성 비율 = 순 변화 / 총 이동 (0-1, 1이 완전 효율적)
    efficiency_ratio = net_change / (price_path + 1e-9)
    
    # 범위 제한
    efficiency_ratio = efficiency_ratio.clip(0, 1)
    
    return efficiency_ratio


def calculate_timeframe_divergence(close: pd.Series) -> pd.Series:
    """
    타임프레임 불일치 지표
    다른 타임프레임 간의 추세 불일치 감지
    """
    # 다양한 기간의 EMA
    ema10 = close.ewm(span=10).mean()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    
    # 기준 시점 대비 변화율
    change10 = close / ema10 - 1
    change20 = close / ema20 - 1
    change50 = close / ema50 - 1
    
    # 타임프레임 불일치 계산
    short_mid_div = (change10 - change20).abs()
    short_long_div = (change10 - change50).abs()
    mid_long_div = (change20 - change50).abs()
    
    # 종합 불일치 점수
    divergence = (short_mid_div + short_long_div + mid_long_div) / 3
    
    # 불일치 방향 (양수: 상향 발산, 음수: 하향 발산)
    direction = np.sign(change10 + change20 + change50)
    
    # 방향 감안 불일치 점수
    tf_divergence = divergence * direction
    
    # 스케일링 (지난 60일 최대치 대비)
    max_div = tf_divergence.abs().rolling(window=60).max().clip(lower=0.001)
    tf_divergence = tf_divergence / max_div
    
    return tf_divergence


def calculate_mtf_momentum(close: pd.Series) -> pd.Series:
    """
    멀티 타임프레임 모멘텀
    여러 타임프레임의 모멘텀을 종합한 지표
    """
    # 다양한 타임프레임의 모멘텀
    momentum1 = close.pct_change(periods=1)
    momentum5 = close.pct_change(periods=5)
    momentum10 = close.pct_change(periods=10)
    momentum20 = close.pct_change(periods=20)
    
    # 표준화
    z1 = (momentum1 - momentum1.rolling(window=100).mean()) / momentum1.rolling(window=100).std()
    z5 = (momentum5 - momentum5.rolling(window=100).mean()) / momentum5.rolling(window=100).std()
    z10 = (momentum10 - momentum10.rolling(window=100).mean()) / momentum10.rolling(window=100).std()
    z20 = (momentum20 - momentum20.rolling(window=100).mean()) / momentum20.rolling(window=100).std()
    
    # 가중치 (단기에서 장기로 감소)
    w1, w5, w10, w20 = 0.4, 0.3, 0.2, 0.1
    
    # 종합 멀티 타임프레임 모멘텀
    mtf_momentum = w1*z1 + w5*z5 + w10*z10 + w20*z20
    
    return mtf_momentum


def calculate_intraday_pattern(data: pd.DataFrame) -> pd.Series:
    """
    일중 패턴 특성
    하루 중 시간대에 따른 특성값
    """
    # 일별 데이터인 경우 더미 값 반환
    if not isinstance(data.index, pd.DatetimeIndex) or 'Time' not in data.columns:
        return pd.Series(0, index=data.index)
    
    # 시간 추출
    if 'Time' in data.columns:
        # 'Time' 컬럼이 있는 경우
        time_series = pd.to_datetime(data['Time']).dt.hour + data['Time'].dt.minute / 60
    else:
        # DatetimeIndex인 경우
        time_series = data.index.hour + data.index.minute / 60
    
    # 정규화된 시간 (0-1 범위, 9:30=0, 16:00=1 가정)
    market_open = 9.5  # 9:30 AM
    market_close = 16.0  # 4:00 PM
    
    normalized_time = (time_series - market_open) / (market_close - market_open)
    normalized_time = normalized_time.clip(0, 1)
    
    # 시간대별 패턴 (U자형 커브)
    pattern = 1 - 4 * (normalized_time - 0.5) ** 2
    
    return pattern


def calculate_calendar_features(dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """
    캘린더 특성 계산
    날짜 관련 다양한 특성 추출
    """
    features = {}
    
    # 월말/월초 지표
    features['Calendar_MonthStart'] = (dates.day <= 3).astype(float)
    features['Calendar_MonthEnd'] = (dates.day >= dates.days_in_month - 2).astype(float)
    
    # 분기말 지표
    is_quarter_end = (dates.month.isin([3, 6, 9, 12]) & (dates.day >= dates.days_in_month - 5)).astype(float)
    features['Calendar_QuarterEnd'] = is_quarter_end
    
    # 요일 (월=0, 금=4)
    days = dates.dayofweek
    features['Calendar_DayOfWeek'] = days / 4  # 0-1 정규화
    
    # 월중 위치 (0-1 정규화)
    features['Calendar_MonthProgress'] = (dates.day - 1) / (dates.days_in_month - 1)
    
    # 년중 위치 (0-1 정규화)
    day_of_year = dates.dayofyear
    is_leap_year = (dates.year % 4 == 0) & ((dates.year % 100 != 0) | (dates.year % 400 == 0))
    days_in_year = 365 + is_leap_year.astype(int)
    features['Calendar_YearProgress'] = (day_of_year - 1) / (days_in_year - 1)
    
    return features