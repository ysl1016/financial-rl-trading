import os
import json
import torch
import datetime
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from ..models.grpo_agent import GRPOAgent
from ..models.trading_env import TradingEnv

logger = logging.getLogger(__name__)

class ModelPackager:
    """
    모델 직렬화 및 역직렬화를 위한 클래스.
    훈련된 모델과 메타데이터를 저장하고 로드하는 기능을 제공합니다.
    """
    
    def __init__(self, base_dir: str = "./models"):
        """
        ModelPackager 초기화
        
        Args:
            base_dir: 모델이 저장될 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, 
                  model: GRPOAgent, 
                  env_config: Dict[str, Any],
                  model_name: str,
                  version: str = None,
                  metadata: Dict[str, Any] = None) -> str:
        """
        모델과 관련 메타데이터를 저장합니다.
        
        Args:
            model: 저장할 GRPOAgent 인스턴스
            env_config: 환경 설정 데이터
            model_name: 모델 이름
            version: 모델 버전 (지정하지 않으면 자동 생성)
            metadata: 추가 메타데이터
            
        Returns:
            저장된 모델의 경로
        """
        # 버전 생성 (지정되지 않은 경우)
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 저장 디렉토리 생성
        model_dir = self.base_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 파일 경로
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        metadata_path = model_dir / "metadata.json"
        
        # 모델 저장
        model_state = {
            'network_state_dict': model.network.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'model_config': {
                'state_dim': model.network.policy[0].in_features,
                'action_dim': model.action_dim,
                'hidden_dim': model.network.policy[2].in_features,
                'reward_scale': model.reward_scale,
                'penalty_scale': model.penalty_scale,
                'gamma': model.gamma,
            }
        }
        torch.save(model_state, model_path)
        
        # 환경 설정 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(env_config, f, indent=2, ensure_ascii=False)
        
        # 메타데이터 저장
        if metadata is None:
            metadata = {}
            
        # 기본 메타데이터 추가
        metadata.update({
            'created_at': datetime.datetime.now().isoformat(),
            'model_name': model_name,
            'version': version,
            'framework_version': torch.__version__,
        })
        
        # 모델 체크섬 추가
        with open(model_path, 'rb') as f:
            model_checksum = hashlib.md5(f.read()).hexdigest()
            metadata['model_checksum'] = model_checksum
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"모델이 저장되었습니다: {model_dir}")
        return str(model_dir)
    
    def load_model(self, 
                  model_name: str, 
                  version: str = "latest",
                  device: str = None) -> Tuple[GRPOAgent, Dict[str, Any], Dict[str, Any]]:
        """
        저장된 모델과 메타데이터를 로드합니다.
        
        Args:
            model_name: 로드할 모델 이름
            version: 로드할 모델 버전 ("latest"면 최신 버전)
            device: 모델을 로드할 장치 (None이면 자동 감지)
            
        Returns:
            (로드된 모델, 환경 설정, 메타데이터) 튜플
        """
        # 장치 설정
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 최신 버전 찾기 (필요한 경우)
        model_base_dir = self.base_dir / model_name
        if not model_base_dir.exists():
            raise FileNotFoundError(f"모델 '{model_name}'을 찾을 수 없습니다.")
            
        if version == "latest":
            versions = [d.name for d in model_base_dir.iterdir() if d.is_dir()]
            if not versions:
                raise FileNotFoundError(f"모델 '{model_name}'의 버전을 찾을 수 없습니다.")
            
            # 날짜 기반 버전 형식이면 날짜순으로 정렬
            try:
                versions.sort(reverse=True)  # 가장 최근 버전 선택
            except Exception:
                pass
                
            version = versions[0]
        
        # 모델 디렉토리 및 파일 경로
        model_dir = model_base_dir / version
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        metadata_path = model_dir / "metadata.json"
        
        # 파일 확인
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        
        # 모델 인스턴스 생성
        agent = GRPOAgent(
            state_dim=model_config.get('state_dim'),
            action_dim=model_config.get('action_dim'),
            hidden_dim=model_config.get('hidden_dim', 128),
            lr=0.0003,  # 추론 시에는 학습률이 중요하지 않음
            gamma=model_config.get('gamma', 0.99),
            reward_scale=model_config.get('reward_scale', 1.0),
            penalty_scale=model_config.get('penalty_scale', 0.5),
            device=device
        )
        
        # 가중치 로드
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 추론 모드로 설정
        agent.network.eval()
        
        # 환경 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            env_config = json.load(f)
            
        # 메타데이터 로드
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # 체크섬 검증
        if 'model_checksum' in metadata:
            with open(model_path, 'rb') as f:
                computed_checksum = hashlib.md5(f.read()).hexdigest()
                
            if computed_checksum != metadata['model_checksum']:
                logger.warning(f"모델 체크섬 불일치: {computed_checksum} != {metadata['model_checksum']}")
        
        logger.info(f"모델 로드 완료: {model_name}/{version}")
        return agent, env_config, metadata
        
    def export_torchscript(self, 
                          model: GRPOAgent, 
                          model_name: str,
                          version: str,
                          output_path: str = None) -> str:
        """
        모델을 TorchScript 형식으로 변환하여 저장합니다.
        이를 통해 Python 의존성 없이 C++에서 모델을 로드할 수 있습니다.
        
        Args:
            model: 변환할 GRPOAgent 인스턴스
            model_name: 모델 이름
            version: 모델 버전
            output_path: 출력 경로 (지정하지 않으면 자동 생성)
            
        Returns:
            저장된 TorchScript 모델의 경로
        """
        # 모델을 평가 모드로 전환
        model.network.eval()
        
        # 스크립트 모드로 변환 (TorchScript)
        scripted_model = torch.jit.script(model.network)
        
        # 저장 경로 결정
        if output_path is None:
            output_dir = self.base_dir / model_name / version / "export"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model.pt"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        scripted_model.save(output_path)
        logger.info(f"TorchScript 모델이 저장되었습니다: {output_path}")
        
        return str(output_path)
    
    def export_onnx(self,
                   model: GRPOAgent,
                   model_name: str,
                   version: str,
                   output_path: str = None) -> str:
        """
        모델을 ONNX 형식으로 변환하여 저장합니다.
        이를 통해 다양한 딥러닝 프레임워크에서 모델을 사용할 수 있습니다.
        
        Args:
            model: 변환할 GRPOAgent 인스턴스
            model_name: 모델 이름
            version: 모델 버전
            output_path: 출력 경로 (지정하지 않으면 자동 생성)
            
        Returns:
            저장된 ONNX 모델의 경로
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("ONNX 변환을 위해 'onnx'와 'onnxruntime' 패키지를 설치하세요.")
        
        # 모델을 평가 모드로 전환
        model.network.eval()
        
        # 더미 입력 생성
        state_dim = model.network.policy[0].in_features
        dummy_input = torch.randn(1, state_dim, device=model.device)
        
        # 저장 경로 결정
        if output_path is None:
            output_dir = self.base_dir / model_name / version / "export"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "model.onnx"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ONNX로 내보내기
        torch.onnx.export(
            model.network,                        # 모델
            dummy_input,                         # 더미 입력
            output_path,                         # 출력 경로
            export_params=True,                  # 가중치 내보내기
            opset_version=12,                    # ONNX 버전
            do_constant_folding=True,            # 상수 폴딩 최적화
            input_names=['input'],               # 입력 이름
            output_names=['output'],             # 출력 이름
            dynamic_axes={                        # 동적 축 (배치 차원)
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 모델 검증
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX 모델이 저장되었습니다: {output_path}")
        return str(output_path)


class ModelVersionManager:
    """
    모델 버전 관리를 위한 클래스.
    모델의 버전 기록, 비교, 롤백 등의 기능을 제공합니다.
    """
    
    def __init__(self, base_dir: str = "./models"):
        """
        ModelVersionManager 초기화
        
        Args:
            base_dir: 모델이 저장된 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_models(self) -> List[str]:
        """
        사용 가능한 모든 모델 이름 목록을 반환합니다.
        
        Returns:
            모델 이름 목록
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        특정 모델의 모든 버전 정보를 반환합니다.
        
        Args:
            model_name: 모델 이름
            
        Returns:
            버전 정보 목록 (생성 시간, 메타데이터 등 포함)
        """
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"모델 '{model_name}'을 찾을 수 없습니다.")
            
        versions = []
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir():
                continue
                
            version_info = {
                'version': version_dir.name,
                'path': str(version_dir)
            }
            
            # 메타데이터 파일이 있으면 읽기
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    version_info.update(metadata)
            
            versions.append(version_info)
            
        # 날짜순으로 정렬 (최신 순)
        versions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return versions
    
    def get_version_info(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        특정 모델의 특정 버전에 대한 정보를 반환합니다.
        
        Args:
            model_name: 모델 이름
            version: 모델 버전
            
        Returns:
            버전 정보 (메타데이터 포함)
        """
        model_dir = self.base_dir / model_name
        version_dir = model_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"모델 '{model_name}'의 버전 '{version}'을 찾을 수 없습니다.")
            
        version_info = {
            'version': version,
            'path': str(version_dir)
        }
        
        # 메타데이터 파일이 있으면 읽기
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                version_info.update(metadata)
                
        return version_info
    
    def get_latest_version(self, model_name: str) -> str:
        """
        특정 모델의 최신 버전을 반환합니다.
        
        Args:
            model_name: 모델 이름
            
        Returns:
            최신 버전 문자열
        """
        versions = self.list_versions(model_name)
        if not versions:
            raise FileNotFoundError(f"모델 '{model_name}'의 버전을 찾을 수 없습니다.")
            
        return versions[0]['version']
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        특정 모델의 특정 버전을 삭제합니다.
        
        Args:
            model_name: 모델 이름
            version: 모델 버전
            
        Returns:
            성공 여부
        """
        model_dir = self.base_dir / model_name
        version_dir = model_dir / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"모델 '{model_name}'의 버전 '{version}'을 찾을 수 없습니다.")
            
        try:
            shutil.rmtree(version_dir)
            logger.info(f"모델 버전이 삭제되었습니다: {model_name}/{version}")
            return True
        except Exception as e:
            logger.error(f"모델 버전 삭제 중 오류 발생: {e}")
            return False
    
    def compare_versions(self, 
                        model_name: str, 
                        version1: str, 
                        version2: str) -> Dict[str, Any]:
        """
        두 모델 버전을 비교합니다.
        
        Args:
            model_name: 모델 이름
            version1: 첫 번째 버전
            version2: 두 번째 버전
            
        Returns:
            두 버전 간 차이점을 담은 딕셔너리
        """
        # 각 버전 정보 가져오기
        info1 = self.get_version_info(model_name, version1)
        info2 = self.get_version_info(model_name, version2)
        
        # 비교 결과
        comparison = {
            'version1': version1,
            'version2': version2,
            'differences': {}
        }
        
        # 메타데이터 차이점 비교
        all_keys = set(info1.keys()) | set(info2.keys())
        for key in all_keys:
            if key in ['version', 'path']:
                continue
                
            if key not in info1:
                comparison['differences'][key] = {'type': 'added', 'value': info2[key]}
            elif key not in info2:
                comparison['differences'][key] = {'type': 'removed', 'value': info1[key]}
            elif info1[key] != info2[key]:
                comparison['differences'][key] = {
                    'type': 'changed',
                    'from': info1[key],
                    'to': info2[key]
                }
        
        return comparison
