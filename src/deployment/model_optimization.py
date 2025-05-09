import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from ..models.grpo_agent import GRPOAgent, GRPONetwork

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    모델 최적화를 위한 클래스.
    훈련된 모델의 크기를 줄이고 추론 속도를 개선하는 기능을 제공합니다.
    """
    
    def __init__(self):
        """
        ModelOptimizer 초기화
        """
        pass
        
    def quantize_model(self, 
                      model: GRPOAgent, 
                      quantization_type: str = 'dynamic',
                      dtype: str = 'qint8') -> GRPOAgent:
        """
        모델을 양자화하여 크기를 줄이고 추론 속도를 개선합니다.
        
        Args:
            model: 양자화할 GRPOAgent 인스턴스
            quantization_type: 양자화 유형 ('static', 'dynamic', 'qat')
            dtype: 양자화 데이터 유형 ('qint8', 'quint8', 'qfloat16')
            
        Returns:
            양자화된 GRPOAgent 인스턴스
        """
        # 원본 모델 백업
        model.network.eval()
        original_state_dict = model.network.state_dict()
        
        if quantization_type == 'dynamic':
            # 동적 양자화
            try:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.default_dynamic_quant_observer,
                    weight=torch.quantization.default_weight_observer)
                
                # 양자화 모델 준비
                quantized_model = torch.quantization.quantize_dynamic(
                    model.network,
                    {torch.nn.Linear},  # 양자화할 모듈 유형
                    dtype=torch.qint8 if dtype == 'qint8' else torch.quint8
                )
                
                # 새 에이전트 생성 (양자화된 네트워크로)
                state_dim = model.network.policy[0].in_features
                quantized_agent = GRPOAgent(
                    state_dim=state_dim,
                    action_dim=model.action_dim,
                    hidden_dim=128,  # 기본값 사용
                    gamma=model.gamma,
                    reward_scale=model.reward_scale,
                    penalty_scale=model.penalty_scale,
                    device=model.device
                )
                
                # 새 에이전트에 양자화된 네트워크 할당
                quantized_agent.network = quantized_model
                
                logger.info(f"모델을 동적으로 양자화했습니다(dtype={dtype})")
                return quantized_agent
                
            except Exception as e:
                logger.error(f"동적 양자화 중 오류 발생: {e}")
                logger.info("원본 모델로 복원합니다.")
                # 원본 모델 복원 및 반환
                model.network.load_state_dict(original_state_dict)
                return model
                
        elif quantization_type == 'static':
            logger.warning("정적 양자화는 현재 구현되지 않았습니다. 원본 모델을 반환합니다.")
            return model
            
        elif quantization_type == 'qat':
            logger.warning("양자화 인식 훈련(QAT)은 현재 구현되지 않았습니다. 원본 모델을 반환합니다.")
            return model
            
        else:
            logger.warning(f"알 수 없는 양자화 유형: {quantization_type}. 원본 모델을 반환합니다.")
            return model
    
    def prune_model(self, 
                   model: GRPOAgent, 
                   pruning_method: str = 'magnitude',
                   amount: float = 0.2) -> GRPOAgent:
        """
        모델을 가지치기하여 크기를 줄이고 추론 속도를 개선합니다.
        
        Args:
            model: 가지치기할 GRPOAgent 인스턴스
            pruning_method: 가지치기 방법 ('magnitude', 'l1_unstructured', 'random')
            amount: 제거할 파라미터의 비율 (0.0 ~ 1.0)
            
        Returns:
            가지치기된 GRPOAgent 인스턴스
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            logger.warning("torch.nn.utils.prune 모듈을 가져올 수 없습니다. 원본 모델을 반환합니다.")
            return model
        
        # 원본 모델 백업
        model.network.eval()
        original_state_dict = model.network.state_dict()
        
        try:
            # 가지치기할 모듈 목록 (Linear 레이어)
            modules_to_prune = []
            
            # 정책 네트워크의 Linear 레이어 찾기
            for i, module in enumerate(model.network.policy):
                if isinstance(module, torch.nn.Linear):
                    modules_to_prune.append((model.network.policy, f'{i}'))
            
            # Q 추정기의 Linear 레이어 찾기
            for i, module in enumerate(model.network.q_estimator):
                if isinstance(module, torch.nn.Linear):
                    modules_to_prune.append((model.network.q_estimator, f'{i}'))
            
            # 가지치기 메서드 선택
            if pruning_method == 'magnitude':
                pruning_fn = prune.l1_unstructured
            elif pruning_method == 'l1_unstructured':
                pruning_fn = prune.l1_unstructured
            elif pruning_method == 'random':
                pruning_fn = prune.random_unstructured
            else:
                logger.warning(f"알 수 없는 가지치기 방법: {pruning_method}. 'magnitude'를 사용합니다.")
                pruning_fn = prune.l1_unstructured
            
            # 각 모듈 가지치기
            for module, name in modules_to_prune:
                pruning_fn(module=module[int(name)], name='weight', amount=amount)
                prune.remove(module=module[int(name)], name='weight')  # 가지치기 영구화
            
            logger.info(f"모델을 가지치기했습니다 (방법={pruning_method}, 비율={amount})")
            return model
            
        except Exception as e:
            logger.error(f"가지치기 중 오류 발생: {e}")
            logger.info("원본 모델로 복원합니다.")
            # 원본 모델 복원 및 반환
            model.network.load_state_dict(original_state_dict)
            return model
    
    def fuse_modules(self, model: GRPOAgent) -> GRPOAgent:
        """
        모델의 모듈을 융합하여 추론 속도를 개선합니다.
        예: Linear + ReLU -> LinearReLU
        
        Args:
            model: 모듈을 융합할 GRPOAgent 인스턴스
            
        Returns:
            모듈이 융합된 GRPOAgent 인스턴스
        """
        # 원본 모델 백업
        model.network.eval()
        original_state_dict = model.network.state_dict()
        
        try:
            # 융합할 모듈 쌍 찾기
            # 예: Linear + ReLU -> LinearReLU
            fused_network = GRPONetwork(
                state_dim=model.network.policy[0].in_features,
                action_dim=model.action_dim,
                hidden_dim=model.network.policy[2].in_features
            )
            
            # 원본 가중치 복사
            fused_network.load_state_dict(original_state_dict)
            
            # 융합된 네트워크 사용
            fused_agent = GRPOAgent(
                state_dim=model.network.policy[0].in_features,
                action_dim=model.action_dim,
                hidden_dim=model.network.policy[2].in_features,
                gamma=model.gamma,
                reward_scale=model.reward_scale,
                penalty_scale=model.penalty_scale,
                device=model.device
            )
            
            # 새 에이전트에 융합된 네트워크 할당
            fused_agent.network = fused_network
            
            logger.info("모델 모듈이 융합되었습니다.")
            return fused_agent
            
        except Exception as e:
            logger.error(f"모듈 융합 중 오류 발생: {e}")
            logger.info("원본 모델로 복원합니다.")
            # 원본 모델 복원 및 반환
            model.network.load_state_dict(original_state_dict)
            return model
    
    def optimize_for_inference(self, 
                             model: GRPOAgent,
                             optimization_level: str = 'medium') -> GRPOAgent:
        """
        모델을 추론에 최적화합니다. 여러 최적화 기법을 복합적으로 적용합니다.
        
        Args:
            model: 최적화할 GRPOAgent 인스턴스
            optimization_level: 최적화 수준 ('low', 'medium', 'high')
            
        Returns:
            최적화된 GRPOAgent 인스턴스
        """
        # 원본 모델 백업
        model.network.eval()
        original_state_dict = model.network.state_dict()
        
        try:
            optimized_model = model
            
            # 최적화 수준에 따라 적용할 기법 결정
            if optimization_level == 'low':
                # 가벼운 최적화
                pass  # 특별한 최적화 없음
                
            elif optimization_level == 'medium':
                # 중간 수준 최적화
                # 1. 모듈 융합
                optimized_model = self.fuse_modules(optimized_model)
                
            elif optimization_level == 'high':
                # 고수준 최적화
                # 1. 모듈 융합
                optimized_model = self.fuse_modules(optimized_model)
                # 2. 경량 가지치기 (10%)
                optimized_model = self.prune_model(optimized_model, amount=0.1)
                # 3. 동적 양자화
                optimized_model = self.quantize_model(optimized_model)
                
            else:
                logger.warning(f"알 수 없는 최적화 수준: {optimization_level}. 'medium'을 사용합니다.")
                # 중간 수준 최적화 적용
                optimized_model = self.fuse_modules(optimized_model)
            
            # 추론 모드로 설정
            optimized_model.network.eval()
            
            # JIT 컴파일 적용 (TorchScript)
            try:
                optimized_model.network = torch.jit.script(optimized_model.network)
                logger.info("JIT 컴파일이 적용되었습니다.")
            except Exception as e:
                logger.warning(f"JIT 컴파일 중 오류 발생: {e}")
            
            logger.info(f"모델이 추론을 위해 최적화되었습니다(수준={optimization_level}).")
            return optimized_model
            
        except Exception as e:
            logger.error(f"추론 최적화 중 오류 발생: {e}")
            logger.info("원본 모델로 복원합니다.")
            # 원본 모델 복원 및 반환
            model.network.load_state_dict(original_state_dict)
            return model
    
    def trace_for_mobile(self, 
                        model: GRPOAgent, 
                        output_path: str) -> str:
        """
        모바일 환경에서 사용하기 위해 모델을 추적하고 저장합니다.
        
        Args:
            model: 추적할 GRPOAgent 인스턴스
            output_path: 추적된 모델을 저장할 경로
            
        Returns:
            저장된 경로
        """
        try:
            # 추론 모드로 설정
            model.network.eval()
            
            # 더미 입력 생성
            state_dim = model.network.policy[0].in_features
            dummy_input = torch.randn(1, state_dim, device=model.device)
            
            # TorchScript로 추적
            traced_model = torch.jit.trace(model.network, dummy_input)
            
            # 저장
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            traced_model.save(output_path)
            
            logger.info(f"모바일용 추적 모델이 저장되었습니다: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"모바일 추적 중 오류 발생: {e}")
            return None
    
    def benchmark_inference(self, 
                          model: GRPOAgent, 
                          num_iterations: int = 1000,
                          batch_size: int = 1) -> Dict[str, float]:
        """
        모델의 추론 성능을 벤치마크합니다.
        
        Args:
            model: 벤치마크할 GRPOAgent 인스턴스
            num_iterations: 반복 횟수
            batch_size: 배치 크기
            
        Returns:
            벤치마크 결과 (평균 추론 시간, 처리량 등)
        """
        import time
        
        # 추론 모드로 설정
        model.network.eval()
        
        # 더미 입력 생성
        state_dim = model.network.policy[0].in_features
        dummy_input = torch.randn(batch_size, state_dim, device=model.device)
        
        # 준비 단계 (워밍업)
        for _ in range(10):
            with torch.no_grad():
                _ = model.network(dummy_input)
        
        # 벤치마크 시작
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model.network(dummy_input)
        
        end_time = time.time()
        
        # 결과 계산
        total_time = end_time - start_time
        avg_time_per_iter = total_time / num_iterations
        avg_time_per_sample = avg_time_per_iter / batch_size
        throughput = num_iterations * batch_size / total_time
        
        results = {
            'total_time_seconds': total_time,
            'avg_time_per_iteration_ms': avg_time_per_iter * 1000,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'throughput_samples_per_second': throughput,
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'device': str(model.device)
        }
        
        logger.info(f"추론 벤치마크 결과: 평균 시간 = {avg_time_per_sample*1000:.3f}ms/샘플, 처리량 = {throughput:.1f}샘플/초")
        return results
    
    def compare_models(self, 
                      original_model: GRPOAgent, 
                      optimized_model: GRPOAgent,
                      num_samples: int = 100) -> Dict[str, Any]:
        """
        원본 모델과 최적화된 모델을 비교합니다.
        
        Args:
            original_model: 원본 GRPOAgent 인스턴스
            optimized_model: 최적화된 GRPOAgent 인스턴스
            num_samples: 비교할 샘플 수
            
        Returns:
            비교 결과 (출력 차이, 성능 개선 등)
        """
        # 추론 모드로 설정
        original_model.network.eval()
        optimized_model.network.eval()
        
        # 더미 입력 생성
        state_dim = original_model.network.policy[0].in_features
        dummy_inputs = torch.randn(num_samples, state_dim, device=original_model.device)
        
        # 각 모델에 대한 추론 수행
        with torch.no_grad():
            original_outputs = []
            optimized_outputs = []
            
            for i in range(num_samples):
                original_output = original_model.network(dummy_inputs[i:i+1])
                optimized_output = optimized_model.network(dummy_inputs[i:i+1])
                
                original_outputs.append(original_output)
                optimized_outputs.append(optimized_output)
        
        # 출력 텐서 결합
        original_outputs = torch.cat(original_outputs, dim=0)
        optimized_outputs = torch.cat(optimized_outputs, dim=0)
        
        # 성능 비교
        original_benchmark = self.benchmark_inference(original_model, num_iterations=100)
        optimized_benchmark = self.benchmark_inference(optimized_model, num_iterations=100)
        
        # 출력 차이 계산
        abs_diff = torch.abs(original_outputs - optimized_outputs)
        mean_abs_diff = torch.mean(abs_diff).item()
        max_abs_diff = torch.max(abs_diff).item()
        
        # 모델 크기 계산 (대략적인 추정)
        original_size = sum(p.numel() * p.element_size() for p in original_model.network.parameters())
        optimized_size = sum(p.numel() * p.element_size() for p in optimized_model.network.parameters())
        
        # 결과 구성
        results = {
            'mean_absolute_difference': mean_abs_diff,
            'max_absolute_difference': max_abs_diff,
            'size_reduction_bytes': original_size - optimized_size,
            'size_reduction_percentage': (1 - optimized_size / original_size) * 100 if original_size > 0 else 0,
            'speed_improvement_percentage': 
                (1 - optimized_benchmark['avg_time_per_sample_ms'] / original_benchmark['avg_time_per_sample_ms']) * 100,
            'original_benchmark': original_benchmark,
            'optimized_benchmark': optimized_benchmark
        }
        
        logger.info(f"모델 비교 결과: 평균 절대 차이 = {mean_abs_diff:.6f}, 크기 감소 = {results['size_reduction_percentage']:.1f}%, 속도 개선 = {results['speed_improvement_percentage']:.1f}%")
        return results
