#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모든 테스트를 실행하는 스크립트
"""

import unittest
import sys
import os
import argparse
import logging

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로거 설정
logger = logging.getLogger('TestRunner')

def run_tests(test_pattern=None, verbosity=2):
    """
    지정된 패턴에 맞는 테스트 실행
    
    Args:
        test_pattern: 실행할 테스트 패턴 (None이면 모든 테스트)
        verbosity: 테스트 출력 상세도 (1=간략, 2=상세)
    
    Returns:
        bool: 모든 테스트 성공 여부
    """
    # 테스트 로더 생성
    loader = unittest.TestLoader()
    
    # 테스트 스위트 생성
    if test_pattern:
        logger.info(f"패턴 '{test_pattern}'에 맞는 테스트 실행 중...")
        suite = loader.discover('.', pattern=test_pattern)
    else:
        logger.info("모든 테스트 실행 중...")
        suite = loader.discover('.', pattern='test_*.py')
    
    # 테스트 실행기 생성
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # 테스트 실행
    result = runner.run(suite)
    
    # 결과 요약
    logger.info(f"실행된 테스트: {result.testsRun}")
    logger.info(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"실패: {len(result.failures)}")
    logger.info(f"오류: {len(result.errors)}")
    
    # 테스트 성공 여부 반환
    return len(result.failures) == 0 and len(result.errors) == 0

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="금융거래 강화학습 모델 테스트 실행")
    
    parser.add_argument(
        '-p', '--pattern',
        help='실행할 테스트 파일 패턴 (예: test_enhanced_processor.py)',
        default=None
    )
    
    parser.add_argument(
        '-v', '--verbosity',
        help='출력 상세도 (1=간략, 2=상세)',
        type=int,
        choices=[1, 2],
        default=2
    )
    
    parser.add_argument(
        '--log-level',
        help='로그 레벨',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    # 인수 파싱
    args = parse_arguments()
    
    # 로그 레벨 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    success = run_tests(args.pattern, args.verbosity)
    
    # 종료 코드 설정
    sys.exit(0 if success else 1)
