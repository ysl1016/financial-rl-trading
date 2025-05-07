import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class MacroEconomicEncoder(nn.Module):
    """
    거시경제 지표를 처리하는 인코더 모듈
    
    금리, 인플레이션, 고용률 등의 거시경제 지표를 금융 트레이딩 모델에 통합
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 거시경제 지표 차원
            hidden_dim: 히든 레이어 차원
            output_dim: 출력 임베딩 차원
            num_layers: 네트워크 레이어 수
            dropout: 드롭아웃 비율
        """
        super(MacroEconomicEncoder, self).__init__()
        
        # 입력 임베딩과 정규화
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 히든 레이어
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout)
                )
            )
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 지표 중요도 가중치
        self.indicator_importance = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 거시경제 지표 [batch_size, input_dim]
            
        Returns:
            encoded: 인코딩된 거시경제 특성 [batch_size, output_dim]
            importance: 각 지표의 중요도 [batch_size, input_dim]
        """
        # 입력 임베딩
        h = self.input_embedding(x)
        h = self.layer_norm(h)
        
        # 히든 레이어 처리
        for layer in self.layers:
            h = h + layer(h)  # 잔차 연결
        
        # 지표 중요도 계산
        importance = self.indicator_importance(h)
        
        # 가중치 적용 및 출력 생성
        weighted_input = x * importance
        encoded = self.output_projection(h)
        
        return encoded, importance


class NewsSentimentAnalyzer(nn.Module):
    """
    뉴스 및 감성 데이터를 처리하는 모듈
    
    금융 뉴스, 소셜 미디어 감성 등의 비구조적 데이터를 트레이딩 모델에 통합
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained_embeddings: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 단어 임베딩 차원
            hidden_dim: 히든 레이어 차원
            output_dim: 출력 임베딩 차원
            num_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout: 드롭아웃 비율
            use_pretrained_embeddings: 사전 학습된 임베딩 사용 여부
            pretrained_embeddings: 사전 학습된 임베딩 텐서 (선택적)
        """
        super(NewsSentimentAnalyzer, self).__init__()
        
        # 단어 임베딩
        if use_pretrained_embeddings and pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,
                padding_idx=0
            )
        else:
            self.word_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(
            d_model=embedding_dim,
            dropout=dropout,
            max_len=512
        )
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(embedding_dim, output_dim)
        
        # 감성 분류 헤드 (훈련용)
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # 부정, 중립, 긍정
        )
        
        # 주제 분류 헤드 (훈련용)
        self.topic_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 10)  # 10개 금융 주제 범주
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sentiment: bool = False,
        return_topic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: 입력 토큰 ID [batch_size, seq_length]
            attention_mask: 어텐션 마스크 [batch_size, seq_length]
            return_sentiment: 감성 분류 결과 반환 여부
            return_topic: 주제 분류 결과 반환 여부
            
        Returns:
            dict: 뉴스 임베딩 및 추가 출력
        """
        # 토큰 임베딩
        embeddings = self.word_embedding(tokens)
        
        # 위치 인코딩 적용
        embeddings = self.positional_encoding(embeddings)
        
        # 어텐션 마스크 변환 (패딩 마스크)
        if attention_mask is not None:
            padding_mask = (~attention_mask.bool())
        else:
            padding_mask = None
        
        # 트랜스포머 인코더 통과
        encoded = self.transformer_encoder(
            src=embeddings,
            src_key_padding_mask=padding_mask
        )
        
        # [CLS] 토큰 임베딩 추출 (첫 번째 토큰)
        pooled = encoded[:, 0, :]
        
        # 출력 프로젝션
        news_embedding = self.output_projection(pooled)
        
        # 결과 딕셔너리 구성
        result = {"embedding": news_embedding}
        
        # 감성 분류 (선택적)
        if return_sentiment:
            sentiment_logits = self.sentiment_head(pooled)
            result["sentiment_logits"] = sentiment_logits
        
        # 주제 분류 (선택적)
        if return_topic:
            topic_logits = self.topic_head(pooled)
            result["topic_logits"] = topic_logits
        
        return result


class MultiModalFusion(nn.Module):
    """
    기술적 지표, 거시경제 지표, 뉴스 감성 데이터를 결합하는 퓨전 모듈
    """
    
    def __init__(
        self,
        technical_dim: int,
        macro_dim: int,
        news_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            technical_dim: 기술적 지표 임베딩 차원
            macro_dim: 거시경제 지표 임베딩 차원
            news_dim: 뉴스 임베딩 차원
            hidden_dim: 히든 레이어 차원
            output_dim: 출력 차원
            dropout: 드롭아웃 비율
        """
        super(MultiModalFusion, self).__init__()
        
        self.technical_dim = technical_dim
        self.macro_dim = macro_dim
        self.news_dim = news_dim
        
        # 각 모달리티 프로젝션
        self.technical_projection = nn.Linear(technical_dim, hidden_dim)
        self.macro_projection = nn.Linear(macro_dim, hidden_dim)
        self.news_projection = nn.Linear(news_dim, hidden_dim)
        
        # 크로스 모달 어텐션
        self.cross_attention = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_modalities=3,
            dropout=dropout
        )
        
        # 모달리티 가중치 (중요도) 계산
        self.modality_importance = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # 출력 프로젝션
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        technical_features: torch.Tensor,
        macro_features: Optional[torch.Tensor] = None,
        news_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            technical_features: 기술적 지표 특성 [batch_size, technical_dim]
            macro_features: 거시경제 지표 특성 [batch_size, macro_dim] (선택적)
            news_features: 뉴스 감성 특성 [batch_size, news_dim] (선택적)
            
        Returns:
            fused: 통합된 특성 [batch_size, output_dim]
            modality_weights: 각 모달리티의 가중치
        """
        batch_size = technical_features.size(0)
        
        # 기술적 지표 프로젝션
        technical_proj = self.technical_projection(technical_features)
        
        # 거시경제 지표 프로젝션 (없는 경우 0으로 채움)
        if macro_features is None:
            macro_proj = torch.zeros(batch_size, technical_proj.size(1), device=technical_features.device)
        else:
            macro_proj = self.macro_projection(macro_features)
        
        # 뉴스 감성 프로젝션 (없는 경우 0으로 채움)
        if news_features is None:
            news_proj = torch.zeros(batch_size, technical_proj.size(1), device=technical_features.device)
        else:
            news_proj = self.news_projection(news_features)
        
        # 모달리티 리스트
        modalities = [technical_proj, macro_proj, news_proj]
        
        # 크로스 모달 어텐션
        attended_modalities = self.cross_attention(modalities)
        
        # 모달리티 가중치 계산
        combined = torch.cat([
            attended_modalities[0],
            attended_modalities[1],
            attended_modalities[2]
        ], dim=-1)
        
        modality_weights = self.modality_importance(combined)
        
        # 가중 합산
        weighted_sum = (
            attended_modalities[0] * modality_weights[:, 0:1] +
            attended_modalities[1] * modality_weights[:, 1:2] +
            attended_modalities[2] * modality_weights[:, 2:3]
        )
        
        # 출력 프로젝션
        fused = self.output_projection(weighted_sum)
        
        # 모달리티 가중치 딕셔너리
        modality_weights_dict = {
            "technical": modality_weights[:, 0],
            "macro": modality_weights[:, 1],
            "news": modality_weights[:, 2]
        }
        
        return fused, modality_weights_dict


class CrossModalAttention(nn.Module):
    """
    여러 모달리티 간의 상호작용을 모델링하는 크로스모달 어텐션
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 히든 레이어 차원
            num_modalities: 모달리티 수
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # 모달리티 간 어텐션
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_modalities)
        ])
        
        # 출력 프로젝션
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_modalities)
        ])
    
    def forward(
        self,
        modalities: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            modalities: 모달리티 임베딩 리스트 [batch_size, hidden_dim]
            
        Returns:
            attended_modalities: 어텐션이 적용된 모달리티 리스트
        """
        batch_size = modalities[0].size(0)
        attended_features = []
        
        # 각 모달리티에 대해 다른 모든 모달리티에 어텐션 적용
        for i in range(self.num_modalities):
            # 현재 모달리티를 쿼리로 사용
            query = modalities[i].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 모든 모달리티를 키/값으로 연결
            key_value = torch.stack(modalities, dim=1)  # [batch_size, num_modalities, hidden_dim]
            
            # 크로스 어텐션 적용
            attended, _ = self.cross_attention[i](
                query=query,
                key=key_value,
                value=key_value
            )
            
            # [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
            attended = attended.squeeze(1)
            
            # 모든 모달리티와 현재 모달리티 연결
            concat_features = torch.cat([attended, modalities[i]], dim=-1)
            
            # 출력 프로젝션
            projected = self.output_projections[i](concat_features)
            
            attended_features.append(projected)
        
        return attended_features


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


# 데이터 처리 유틸리티 함수

def process_macro_economic_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    거시경제 데이터 처리 및 특성 추출
    
    Args:
        data: 거시경제 지표가 포함된 DataFrame
        
    Returns:
        processed_data: 처리된 거시경제 특성
    """
    df = pd.DataFrame(index=data.index)
    
    # 1. 기본 거시경제 지표 (예시)
    indicators = [
        'InterestRate', 'Inflation', 'UnemploymentRate', 
        'GDP_Growth', 'ConsumerSentiment', 'PMI',
        'RetailSales', 'IndustrialProduction', 'HousingStarts'
    ]
    
    # 기존 컬럼 복사
    for indicator in indicators:
        if indicator in data.columns:
            df[indicator] = data[indicator]
    
    # 2. 파생 지표 계산
    
    # 금리 곡선 기울기 (10년물 - 2년물)
    if 'InterestRate_10Y' in data.columns and 'InterestRate_2Y' in data.columns:
        df['YieldCurveSlope'] = data['InterestRate_10Y'] - data['InterestRate_2Y']
    
    # 실질 금리 (명목 금리 - 인플레이션)
    if 'InterestRate' in data.columns and 'Inflation' in data.columns:
        df['RealInterestRate'] = data['InterestRate'] - data['Inflation']
    
    # 경기 선행 지수
    if set(['RetailSales', 'ConsumerSentiment', 'PMI']).issubset(data.columns):
        # 간단한 경기 선행 지수 (정규화된 지표들의 평균)
        leading_indicators = ['RetailSales', 'ConsumerSentiment', 'PMI']
        normalized = data[leading_indicators].apply(lambda x: (x - x.mean()) / x.std())
        df['LeadingEconomicIndex'] = normalized.mean(axis=1)
    
    # 3. 변화율 계산
    for col in df.columns:
        if col in data.columns:
            # 월별 변화율
            df[f'{col}_MoM'] = df[col].pct_change()
            
            # 연간 변화율
            df[f'{col}_YoY'] = df[col].pct_change(periods=12)
    
    # 4. 추세 지표 계산
    for col in df.columns:
        if not col.endswith('_MoM') and not col.endswith('_YoY'):
            # 3개월 이동평균
            df[f'{col}_MA3'] = df[col].rolling(window=3).mean()
            
            # 트렌드 방향 (1: 상승, 0: 횡보, -1: 하락)
            df[f'{col}_Trend'] = np.sign(df[f'{col}_MA3'] - df[f'{col}_MA3'].shift(3))
    
    # 5. Z-점수 정규화
    for col in df.columns:
        if not col.endswith('_Trend'):
            mean = df[col].mean()
            std = df[col].std()
            df[f'{col}_Z'] = (df[col] - mean) / (std + 1e-9)
    
    # 결측치 처리
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def preprocess_news_data(news_texts: List[str], news_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    뉴스 데이터 전처리 및 기본 특성 추출
    
    Args:
        news_texts: 뉴스 텍스트 리스트
        news_dates: 뉴스 발행 날짜 리스트
        
    Returns:
        news_features: 뉴스 특성 DataFrame
    """
    import re
    from collections import Counter
    
    # 결과 DataFrame 초기화
    df = pd.DataFrame(index=pd.DatetimeIndex(news_dates))
    
    # 키워드 카운팅 및 감성 분석 (간단한 예시)
    daily_keyword_counts = {}
    
    # 긍정/부정 단어 사전 (예시)
    positive_words = {
        'increase', 'gain', 'growth', 'positive', 'rise', 'improve', 'recovery',
        'profit', 'bullish', 'optimistic', 'upward', 'surged', 'strong', 'successful'
    }
    
    negative_words = {
        'decrease', 'loss', 'decline', 'negative', 'fall', 'drop', 'weak', 
        'bearish', 'pessimistic', 'downward', 'plunged', 'struggled', 'failed'
    }
    
    # 키워드 세트 (예시)
    economic_keywords = {
        'inflation', 'recession', 'gdp', 'economy', 'fed', 'interest rate', 
        'unemployment', 'stimulus', 'fiscal', 'monetary policy'
    }
    
    corporate_keywords = {
        'earnings', 'revenue', 'profit', 'merger', 'acquisition', 'ceo', 
        'dividend', 'bankruptcy', 'debt', 'stock'
    }
    
    # 날짜별 처리
    for date in set(news_dates):
        date_texts = [text for text, d in zip(news_texts, news_dates) if d == date]
        
        # 모든 텍스트 조합
        combined_text = ' '.join(date_texts).lower()
        
        # 단어 빈도 계산
        words = re.findall(r'\b[a-z]+\b', combined_text)
        word_counts = Counter(words)
        
        # 긍정/부정 단어 카운팅
        positive_count = sum(word_counts[word] for word in positive_words if word in word_counts)
        negative_count = sum(word_counts[word] for word in negative_words if word in word_counts)
        
        # 감성 점수 계산 (-1 ~ +1)
        total_sentiment_words = positive_count + negative_count
        sentiment_score = 0
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # 키워드 그룹 카운팅
        economic_count = sum(word_counts[word] for word in economic_keywords if word in word_counts)
        corporate_count = sum(word_counts[word] for word in corporate_keywords if word in word_counts)
        
        # 특성 저장
        daily_keyword_counts[date] = {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'economic_count': economic_count,
            'corporate_count': corporate_count,
            'total_news_count': len(date_texts)
        }
    
    # DataFrame 생성
    daily_df = pd.DataFrame.from_dict(daily_keyword_counts, orient='index')
    
    # 결과 병합
    df = pd.concat([df, daily_df], axis=1)
    
    # 결측치 처리
    df = df.fillna(0)
    
    # 이동 평균 및 추세 계산
    for col in df.columns:
        # 3일 이동평균
        df[f'{col}_MA3'] = df[col].rolling(window=3).mean()
        
        # 7일 이동평균
        df[f'{col}_MA7'] = df[col].rolling(window=7).mean()
        
        # 감성 변화량
        df[f'{col}_change'] = df[col].diff()
    
    # 최종 결측치 처리
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def merge_market_and_external_data(
    market_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    news_data: pd.DataFrame,
    date_column: str = 'Date'
) -> pd.DataFrame:
    """
    시장 데이터, 거시경제 데이터, 뉴스 데이터 병합
    
    Args:
        market_data: OHLCV 및 기술적 지표 데이터
        macro_data: 거시경제 지표 데이터
        news_data: 뉴스 감성 데이터
        date_column: 날짜 컬럼명
        
    Returns:
        merged_data: 병합된 DataFrame
    """
    # 날짜 인덱스 확인 및 설정
    if date_column in market_data.columns:
        market_data = market_data.set_index(date_column)
    
    if date_column in macro_data.columns:
        macro_data = macro_data.set_index(date_column)
    
    if date_column in news_data.columns:
        news_data = news_data.set_index(date_column)
    
    # 날짜 인덱스 정렬
    market_data = market_data.sort_index()
    macro_data = macro_data.sort_index()
    news_data = news_data.sort_index()
    
    # 병합 (외부 조인)
    merged = market_data.join(
        [
            macro_data.add_prefix('Macro_'),
            news_data.add_prefix('News_')
        ],
        how='left'
    )
    
    # 결측치 처리
    # 1. 거시경제 데이터 - 전방향 채우기 (최신 정보로 유지)
    macro_cols = [col for col in merged.columns if col.startswith('Macro_')]
    merged[macro_cols] = merged[macro_cols].fillna(method='ffill')
    
    # 2. 뉴스 데이터 - 0으로 채우기 (뉴스가 없는 날은 중립)
    news_cols = [col for col in merged.columns if col.startswith('News_')]
    merged[news_cols] = merged[news_cols].fillna(0)
    
    # 3. 나머지 결측치 처리
    merged = merged.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return merged