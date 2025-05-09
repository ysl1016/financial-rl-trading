import sys
import os
import unittest
import argparse

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_tests(test_type=None, verbose=True):
    """
    지정된 유형의 테스트를 실행합니다.
    
    Args:
        test_type (str, optional): 실행할 테스트 유형 ('unit', 'integration', 'regression', 'all')
        verbose (bool, optional): 상세 출력 여부
    
    Returns:
        bool: 모든 테스트 통과 여부
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if test_type == 'unit' or test_type == 'all':
        print("\n=== 단위 테스트 실행 중... ===")
        unit_tests = unittest.TestLoader().discover(current_dir, pattern="test_*_env.py")
        unit_tests.addTests(unittest.TestLoader().discover(current_dir, pattern="test_*_agent.py"))
        unit_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(unit_tests)
        
        if unit_result.failures or unit_result.errors:
            print("\n❌ 단위 테스트 실패")
            if test_type != 'all':
                return False
        else:
            print("\n✅ 단위 테스트 통과")
    
    if test_type == 'integration' or test_type == 'all':
        print("\n=== 통합 테스트 실행 중... ===")
        integration_tests = unittest.TestLoader().discover(current_dir, pattern="test_integration.py")
        integration_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(integration_tests)
        
        if integration_result.failures or integration_result.errors:
            print("\n❌ 통합 테스트 실패")
            if test_type != 'all':
                return False
        else:
            print("\n✅ 통합 테스트 통과")
    
    if test_type == 'regression' or test_type == 'all':
        print("\n=== 회귀 테스트 실행 중... ===")
        regression_tests = unittest.TestLoader().discover(current_dir, pattern="test_regression.py")
        regression_result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(regression_tests)
        
        if regression_result.failures or regression_result.errors:
            print("\n❌ 회귀 테스트 실패")
            if test_type != 'all':
                return False
        else:
            print("\n✅ 회귀 테스트 통과")
    
    if test_type == 'all':
        all_success = (not unit_result.failures and not unit_result.errors and
                      not integration_result.failures and not integration_result.errors and
                      not regression_result.failures and not regression_result.errors)
        print("\n=== 테스트 최종 결과 ===")
        if all_success:
            print("✅ 모든 테스트 통과")
        else:
            print("❌ 일부 테스트 실패")
        return all_success
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='금융 RL 트레이딩 모델 테스트 실행')
    parser.add_argument('--type', type=str, choices=['unit', 'integration', 'regression', 'all'],
                        default='all', help='실행할 테스트 유형 (기본값: all)')
    parser.add_argument('--quiet', action='store_true', help='상세 출력 비활성화')
    
    args = parser.parse_args()
    
    success = run_tests(test_type=args.type, verbose=not args.quiet)
    sys.exit(0 if success else 1)
