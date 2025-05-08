#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
분산 학습 유틸리티

이 모듈은 DeepSeek-R1 GRPO 에이전트의 분산 훈련을 위한 유틸리티 함수들을 제공합니다.
다중 GPU 환경에서의 데이터 병렬 처리, 동기화, 그래디언트 통합 등을 지원합니다.
"""

import os
import random
import socket
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Sampler, DistributedSampler


def setup_distributed_environment(
    rank: int,
    world_size: int,
    master_addr: str = 'localhost',
    master_port: str = '12355',
    backend: str = 'nccl'
) -> None:
    """
    분산 학습 환경 설정
    
    Args:
        rank: 현재 프로세스의 랭크
        world_size: 전체 프로세스 수
        master_addr: 마스터 노드 주소
        master_port: 마스터 노드 포트
        backend: 분산 백엔드 ('nccl' 또는 'gloo')
    """
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 프로세스 그룹 초기화
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # 현재 GPU 설정
    if backend == 'nccl':
        torch.cuda.set_device(rank)


def cleanup_distributed_environment() -> None:
    """분산 학습 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_available_port(port_range: Tuple[int, int] = (10000, 20000)) -> int:
    """
    사용 가능한 포트 찾기
    
    Args:
        port_range: 포트 범위 (시작, 끝)
    
    Returns:
        사용 가능한 포트 번호
    """
    min_port, max_port = port_range
    while True:
        port = random.randint(min_port, max_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue


def is_distributed() -> bool:
    """분산 환경 여부 확인"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """현재 프로세스 랭크 반환"""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """전체 프로세스 수 반환"""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """메인 프로세스 여부 확인"""
    return get_rank() == 0


def synchronize() -> None:
    """모든 프로세스 동기화"""
    if is_distributed():
        dist.barrier()


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True
) -> Dict[str, torch.Tensor]:
    """
    딕셔너리의 텐서 값을 모든 프로세스에서 축소
    
    Args:
        input_dict: 축소할 딕셔너리
        average: 평균 계산 여부
    
    Returns:
        축소된 딕셔너리
    """
    if not is_distributed():
        return input_dict
    
    # 딕셔너리를 텐서로 변환
    names = []
    values = []
    for k, v in sorted(input_dict.items()):
        if isinstance(v, torch.Tensor):
            names.append(k)
            values.append(v)
    
    if not values:
        return input_dict
    
    # 텐서를 단일 텐서로 연결
    values = torch.stack(values, dim=0)
    
    # 축소
    dist.reduce(values, dst=0)
    
    # 메인 프로세스에서 평균 계산
    if average and is_main_process():
        values /= get_world_size()
    
    # 결과 딕셔너리 생성
    if is_main_process():
        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict
    else:
        return input_dict


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    world_size: Optional[int] = None
) -> torch.Tensor:
    """
    텐서를 모든 프로세스에서 축소
    
    Args:
        tensor: 축소할 텐서
        op: 축소 연산
        world_size: 프로세스 수 (None이면 자동 감지)
    
    Returns:
        축소된 텐서
    """
    if not is_distributed():
        return tensor
    
    if world_size is None:
        world_size = get_world_size()
    
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=op)
    
    if op == dist.ReduceOp.SUM:
        reduced_tensor.div_(world_size)
    
    return reduced_tensor


def all_reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True
) -> Dict[str, torch.Tensor]:
    """
    딕셔너리의 텐서 값을 모든 프로세스에서 올-리듀스
    
    Args:
        input_dict: 축소할 딕셔너리
        average: 평균 계산 여부
    
    Returns:
        축소된 딕셔너리
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    result_dict = {}
    
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            result_dict[k] = all_reduce_tensor(v, dist.ReduceOp.SUM, world_size if average else None)
        else:
            result_dict[k] = v
    
    return result_dict


def broadcast_value(
    value: Union[torch.Tensor, float, int, bool],
    src: int = 0
) -> Union[torch.Tensor, float, int, bool]:
    """
    값을 모든 프로세스에 브로드캐스트
    
    Args:
        value: 브로드캐스트할 값
        src: 소스 랭크
    
    Returns:
        브로드캐스트된 값
    """
    if not is_distributed():
        return value
    
    if isinstance(value, torch.Tensor):
        dist.broadcast(value, src=src)
        return value
    else:
        # 스칼라 값을 텐서로 변환
        if is_main_process():
            tensor = torch.tensor([value], dtype=torch.float32).cuda()
        else:
            tensor = torch.tensor([0.0], dtype=torch.float32).cuda()
        
        dist.broadcast(tensor, src=src)
        
        # 값 타입 변환
        if isinstance(value, float):
            return tensor.item()
        elif isinstance(value, int):
            return int(tensor.item())
        elif isinstance(value, bool):
            return bool(tensor.item())
        else:
            return tensor.item()


def prepare_model_for_ddp(
    model: torch.nn.Module,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True
) -> DDP:
    """
    모델을 DDP로 래핑
    
    Args:
        model: 래핑할 모델
        find_unused_parameters: 사용되지 않는 매개변수 찾기 여부
        gradient_as_bucket_view: 그래디언트 버킷 뷰 사용 여부
    
    Returns:
        DDP로 래핑된 모델
    """
    if not is_distributed():
        warnings.warn("분산 환경이 아니므로 DDP를 사용하지 않습니다.")
        return model
    
    # 현재 장치 설정
    local_rank = get_rank()
    model = model.to(f"cuda:{local_rank}")
    
    # DDP 래핑
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view
    )
    
    return ddp_model


def prepare_batch_for_device(
    batch: Union[torch.Tensor, List, Dict],
    device: torch.device
) -> Union[torch.Tensor, List, Dict]:
    """
    배치 데이터를 장치로 이동
    
    Args:
        batch: 이동할 배치 데이터
        device: 대상 장치
    
    Returns:
        장치로 이동된 배치 데이터
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [prepare_batch_for_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: prepare_batch_for_device(v, device) for k, v in batch.items()}
    else:
        return batch


class DistributedTimeSeriesDataset(Dataset):
    """분산 환경을 위한 시계열 데이터셋"""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        step: int = 1
    ):
        """
        Args:
            data: 시계열 데이터 (샘플 수, 특성 차원)
            seq_length: 시퀀스 길이
            step: 시퀀스 생성 단계 크기
        """
        self.data = data
        self.seq_length = seq_length
        self.step = step
        
        # 시퀀스 인덱스 생성
        self.indices = list(range(0, len(data) - seq_length, step))
    
    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 항목 반환"""
        start_idx = self.indices[idx]
        
        # 시퀀스 및 다음 시점 값 추출
        sequence = self.data[start_idx:start_idx + self.seq_length]
        next_value = self.data[start_idx + self.seq_length]
        
        return sequence, next_value


def get_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42
) -> Sampler:
    """
    분산 환경을 위한 샘플러 생성
    
    Args:
        dataset: 대상 데이터셋
        shuffle: 셔플 여부
        seed: 무작위 시드
    
    Returns:
        분산 샘플러 또는 None
    """
    if is_distributed():
        return DistributedSampler(
            dataset=dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            seed=seed
        )
    return None


def set_random_seed(seed: int) -> None:
    """
    분산 환경에서의 난수 발생기 시드 설정
    
    Args:
        seed: 설정할 시드
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 분산 환경에서 재현성 보장
    if is_distributed():
        # 기본 시드에 랭크 추가하여 다른 랭크에서 다른 시퀀스 생성
        rank = get_rank()
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
    
    # CUDA 결정성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def launch_distributed_job(
    main_fn,
    num_gpus_per_node: int,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = 'localhost',
    master_port: Optional[str] = None,
    backend: str = 'nccl',
    **kwargs
) -> None:
    """
    분산 작업 실행
    
    Args:
        main_fn: 메인 함수
        num_gpus_per_node: 노드당 GPU 수
        num_nodes: 전체 노드 수
        node_rank: 현재 노드 랭크
        master_addr: 마스터 노드 주소
        master_port: 마스터 노드 포트 (None이면 자동 감지)
        backend: 분산 백엔드
        **kwargs: 메인 함수에 전달할 추가 인수
    """
    world_size = num_gpus_per_node * num_nodes
    
    if world_size <= 1:
        # 단일 GPU 작업
        main_fn(rank=0, world_size=1, **kwargs)
        return
    
    # 마스터 포트 설정
    if master_port is None:
        master_port = str(get_available_port())
    
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 시작 메소드 설정
    mp.set_start_method('spawn', force=True)
    
    # 분산 작업 시작
    processes = []
    for local_rank in range(num_gpus_per_node):
        global_rank = node_rank * num_gpus_per_node + local_rank
        
        p = mp.Process(
            target=_distributed_worker,
            args=(main_fn, backend, world_size, global_rank),
            kwargs=kwargs
        )
        p.start()
        processes.append(p)
    
    # 모든 프로세스 종료 대기
    for p in processes:
        p.join()


def _distributed_worker(main_fn, backend, world_size, rank, **kwargs):
    """분산 작업자 함수"""
    # GPU 설정
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # 프로세스 그룹 초기화
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    
    # 메인 함수 실행
    try:
        main_fn(rank=rank, world_size=world_size, **kwargs)
    except Exception as e:
        print(f"랭크 {rank}에서 오류 발생: {e}")
        raise
    finally:
        # 프로세스 그룹 정리
        dist.destroy_process_group()
