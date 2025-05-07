import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.enhanced_processor import EnhancedDataProcessor

class TestEnhancedDataProcessor(unittest.TestCase):
    """EnhancedDataProcessor 클래스에 대한 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 셋업 - 테스트 데이터 생성"""
        # 테스트 데이터 경로
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # 가상의 주가 데이터 생성
        cls.create_mock_stock_data()
        
        # 캐시 디렉토리
        cls.cache_dir = os.path.join(cls.test_data_dir, 'cache')
        os.makedirs(cls.cache_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 - 임시 파일 정리"""
        # 실제 사용 시 주석 해제하여 테스트 데이터 정리
        # import shutil
        # shutil.rmtree(cls.test_data_dir)
        pass
    
    @classmethod
    def create_mock_stock_data(cls):
        """가상의 주가 데이터 생성"""
        # 날짜 범위 생성
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 랜덤 시드 설정
        np.random.seed(42)
        
        # 초기 가격 설정
        initial_price = 100.0
        
        # 주가 데이터 생성
        prices = [initial_price]
        for i in range(1, len(date_range)):
            # 랜덤 일일 변동폭 (-2% ~ +2%)
            daily_return = np.random.normal(0.0005, 0.015)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # 데이터프레임 생성
        stock_data = pd.DataFrame({
            'Date': date_range,
            'Open': prices * np.random.uniform(0.99, 1.0, len(prices)),
            'High': prices * np.random.uniform(1.01, 1.03, len(prices)),
            'Low': prices * np.random.uniform(0.97, 0.99, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(prices))
        })
        
        # 인덱스 설정
        stock_data.set_index('Date', inplace=True)
        
        # CSV 파일로 저장
        cls.test_data_path = os.path.join(cls.test_data_dir, 'test_stock_data.csv')
        stock_data.to_csv(cls.test_data_path)
        
        return stock_data
    
    def setUp(self):
        """각 테스트 전 설정"""
        self.processor = EnhancedDataProcessor(
            use_advanced_indicators=True,
            use_macro_data=True,
            use_news_data=True,
            normalization_method='adaptive',
            data_cache_dir=self.cache_dir
        )
        
        # 테스트 데이터 로드
        self.test_data = pd.read_csv(self.test_data_path, index_col='Date', parse_dates=True)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.processor.normalization_method, 'adaptive')
        self.assertTrue(self.processor.use_advanced_indicators)
        self.assertTrue(self.processor.use_macro_data)
        self.assertTrue(self.processor.use_news_data)
        self.assertEqual(self.processor.data_cache_dir, self.cache_dir)
    
    def test_download_stock_data(self):
        """주식 데이터 다운로드 테스트"""
        # 모의 다운로드 (실제 API를 호출하지 않고 테스트 데이터 사용)
        def mock_download(symbol, start_date, end_date, force_download):
            return self.test_data.copy()
        
        # 다운로드 함수를 모의 함수로 대체
        original_download = self.processor.download_stock_data
        self.processor.download_stock_data = mock_download
        
        try:
            # 테스트 수행
            data = self.processor.download_stock_data('TEST', '2022-01-01', '2023-01-01', False)
            
            # 검증
            self.assertIsInstance(data, pd.DataFrame)
            self.assertTrue('Open' in data.columns)
            self.assertTrue('High' in data.columns)
            self.assertTrue('Low' in data.columns)
            self.assertTrue('Close' in data.columns)
            self.assertTrue('Volume' in data.columns)
            self.assertEqual(len(data), len(self.test_data))
        finally:
            # 원래 함수 복원
            self.processor.download_stock_data = original_download
    
    def test_process_data(self):
        """데이터 처리 테스트"""
        # 모의 다운로드 함수
        def mock_download_stock(symbol, start_date, end_date, force_download=False):
            return self.test_data.copy()
        
        def mock_download_macro(indicators, start_date, end_date, source):
            # 가상의 거시경제 데이터
            data = pd.DataFrame(index=self.test_data.index)
            data['GDP'] = np.random.normal(3.0, 0.5, len(data))
            data['UNRATE'] = np.random.normal(5.0, 0.3, len(data))
            data['CPIAUCSL'] = np.random.normal(2.0, 0.2, len(data))
            return data
        
        def mock_download_news(symbol, start_date, end_date, source):
            # 가상의 뉴스 데이터
            data = pd.DataFrame(index=self.test_data.index)
            data['sentiment_score'] = np.random.normal(0.1, 0.3, len(data))
            data['positive_count'] = np.random.randint(0, 10, len(data))
            data['negative_count'] = np.random.randint(0, 8, len(data))
            return data
        
        # 원본 함수 저장
        original_stock_download = self.processor.download_stock_data
        original_macro_download = self.processor.download_macro_data
        original_news_download = self.processor.download_news_data
        
        # 모의 함수로 대체
        self.processor.download_stock_data = mock_download_stock
        self.processor.download_macro_data = mock_download_macro
        self.processor.download_news_data = mock_download_news
        
        try:
            # 테스트 수행
            processed_data = self.processor.process_data('TEST', '2022-01-01', '2023-01-01')
            
            # 검증
            self.assertIsInstance(processed_data, pd.DataFrame)
            
            # 기본 가격 데이터 확인
            self.assertTrue('Open' in processed_data.columns)
            self.assertTrue('Close' in processed_data.columns)
            
            # 기술적 지표 확인
            self.assertTrue(any(col.endswith('_norm') for col in processed_data.columns))
            
            # 정규화된 특성 확인
            norm_columns = [col for col in processed_data.columns if col.endswith('_norm')]
            self.assertGreater(len(norm_columns), 0)
            
            # 모든 열에 결측치가 없는지 확인
            self.assertTrue(processed_data.notna().all().all())
            
            # 데이터 길이 확인
            self.assertEqual(len(processed_data), len(self.test_data))
            
        finally:
            # 원래 함수 복원
            self.processor.download_stock_data = original_stock_download
            self.processor.download_macro_data = original_macro_download
            self.processor.download_news_data = original_news_download
    
    def test_feature_groups(self):
        """특성 그룹 설정 테스트"""
        # 특성 그룹 확인
        self.assertIsInstance(self.processor.feature_groups, dict)
        
        # 기본 그룹 포함 확인
        expected_groups = ['price', 'volume', 'oscillators', 'momentum', 'volatility']
        for group in expected_groups:
            self.assertIn(group, self.processor.feature_groups)
    
    def test_cache_functionality(self):
        """캐시 기능 테스트"""
        # 모의 다운로드 함수
        def mock_download(symbol, start_date, end_date, force_download=False):
            return self.test_data.copy()
        
        # 다운로드 함수를 모의 함수로 대체
        original_download = self.processor.download_stock_data
        self.processor.download_stock_data = mock_download
        
        try:
            # 캐시 경로 생성
            cache_path = self.processor._get_cache_path('TEST', '2022-01-01', '2023-01-01', '_raw')
            
            # 캐시 파일 존재 여부 확인
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            # 첫 번째 호출 (캐시 생성)
            data1 = self.processor.download_stock_data('TEST', '2022-01-01', '2023-01-01')
            
            # 캐시 파일 생성 확인
            self.assertTrue(os.path.exists(cache_path))
            
            # 두 번째 호출 (캐시 사용)
            data2 = self.processor.download_stock_data('TEST', '2022-01-01', '2023-01-01')
            
            # 두 결과가 동일한지 확인
            pd.testing.assert_frame_equal(data1, data2)
            
            # 강제 다운로드 테스트
            data3 = self.processor.download_stock_data('TEST', '2022-01-01', '2023-01-01', force_download=True)
            
            # 여전히 동일한지 확인 (모의 함수이므로 동일해야 함)
            pd.testing.assert_frame_equal(data1, data3)
            
        finally:
            # 원래 함수 복원
            self.processor.download_stock_data = original_download


if __name__ == '__main__':
    unittest.main()
