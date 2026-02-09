"""
Base Scheduler Module

Defines the abstract base class for request schedulers and common data structures.
"""

import time
import logging
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    LENGTH_BASED = "length_based"


@dataclass
class SchedulerRequest:
    request_id: str
    question_id: str
    prompt: str
    input_tokens: int
    sequence_category: str
    actual_total_tokens: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    target_instance_id: Optional[str] = None
    target_tp_degree: Optional[int] = None
    priority: int = 0
    max_tokens: int = 512
    temperature: float = 0.7
    
    def get_queue_time(self) -> float:
        if self.scheduled_at:
            return self.scheduled_at - self.created_at
        return time.time() - self.created_at


@dataclass
class SchedulingResult:
    request_id: str
    instance_id: str
    tp_degree: int
    success: bool
    queue_time: float
    reason: str = ""


@dataclass
class SchedulerStatistics:
    total_requests: int = 0
    successful_schedules: int = 0
    failed_schedules: int = 0
    avg_queue_time: float = 0.0
    max_queue_time: float = 0.0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_tp_degree: Dict[int, int] = field(default_factory=dict)
    by_instance: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_schedules / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_schedules": self.successful_schedules,
            "failed_schedules": self.failed_schedules,
            "success_rate": self.success_rate,
            "avg_queue_time": self.avg_queue_time,
            "max_queue_time": self.max_queue_time,
            "by_category": self.by_category,
            "by_tp_degree": self.by_tp_degree,
            "by_instance": self.by_instance
        }


class BaseScheduler(ABC):
    """Abstract base class for request schedulers."""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self._statistics = SchedulerStatistics()
        self._queue_times: List[float] = []
    
    @abstractmethod
    async def route_request(self, request: SchedulerRequest, available_instances: List[Any]) -> SchedulingResult:
        pass
    
    @abstractmethod
    def get_preferred_tp_degrees(self, request: SchedulerRequest) -> List[int]:
        pass
    
    def record_result(self, result: SchedulingResult) -> None:
        self._statistics.total_requests += 1
        if result.success:
            self._statistics.successful_schedules += 1
            self._statistics.by_instance[result.instance_id] = self._statistics.by_instance.get(result.instance_id, 0) + 1
            self._statistics.by_tp_degree[result.tp_degree] = self._statistics.by_tp_degree.get(result.tp_degree, 0) + 1
        else:
            self._statistics.failed_schedules += 1
        self._queue_times.append(result.queue_time)
        self._statistics.avg_queue_time = sum(self._queue_times) / len(self._queue_times)
        self._statistics.max_queue_time = max(self._queue_times)
    
    def record_category(self, category: str) -> None:
        self._statistics.by_category[category] = self._statistics.by_category.get(category, 0) + 1
    
    def get_statistics(self) -> SchedulerStatistics:
        return self._statistics
    
    def reset_statistics(self) -> None:
        self._statistics = SchedulerStatistics()
        self._queue_times.clear()


class LoadBalancer:
    """Helper class for load balancing across instances."""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self._round_robin_index: Dict[int, int] = defaultdict(int)
    
    def select_instance(self, instances: List[Any], tp_degree: Optional[int] = None) -> Optional[Any]:
        if not instances:
            return None
        if tp_degree is not None:
            instances = [i for i in instances if i.tp_degree == tp_degree]
            if not instances:
                return None
        ready_instances = [i for i in instances if i.is_ready]
        if not ready_instances:
            return None
        
        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            return self._round_robin_select(ready_instances, tp_degree or 0)
        elif self.strategy == SchedulingStrategy.LEAST_CONNECTIONS:
            return min(ready_instances, key=lambda i: i.active_requests)
        else:
            return ready_instances[0]
    
    def _round_robin_select(self, instances: List[Any], tp_key: int) -> Any:
        index = self._round_robin_index[tp_key] % len(instances)
        self._round_robin_index[tp_key] += 1
        return instances[index]


def _get_actual_total_tokens_from_profiling(profiling_dir: str, question_id: str) -> Optional[int]:
    """
    从profiling目录中获取指定问题的实际总token数
    
    Args:
        profiling_dir: profiling结果目录路径
        question_id: 问题ID
        
    Returns:
        实际总token数，如果找不到则返回None
    """
    try:
        inference_results_file = Path(profiling_dir) / "inference_results.json"
        if not inference_results_file.exists():
            logger.warning(f"Profiling file not found: {inference_results_file}")
            return None
        
        with open(inference_results_file, 'r', encoding='utf-8') as f:
            inference_results = json.load(f)
        
        # 查找对应question_id的记录
        for result in inference_results:
            if result.get("question_id") == question_id:
                return result.get("actual_total_tokens")
        
        logger.warning(f"Question {question_id} not found in profiling results")
        return None
        
    except Exception as e:
        logger.error(f"Error loading actual_total_tokens from profiling: {e}")
        return None


def categorize_sequence(input_tokens: int, thresholds: Optional[Dict[str, int]] = None) -> str:
    if thresholds is None:
        thresholds = {"short": 256, "medium": 512, "long": 1024}
    if input_tokens <= thresholds.get("short", 256):
        return "short"
    elif input_tokens <= thresholds.get("medium", 512):
        return "medium"
    elif input_tokens <= thresholds.get("long", 1024):
        return "long"
    else:
        return "extra_long"


def create_scheduler_request(request_id: str, question_id: str, prompt: str,
                            input_tokens: int, thresholds: Optional[Dict[str, int]] = None,
                            actual_total_tokens: Optional[int] = None,
                            profiling_dir: Optional[str] = None,
                            total_thresholds: Optional[Dict[str, int]] = None,
                            **kwargs) -> SchedulerRequest:
    """
    创建调度请求，支持基于实际token数的智能分类。
    
    分类优先级：
    1. 首选：使用profiling数据中的actual_total_tokens + 总token阈值
    2. 备选：使用输入token数 + 输入token阈值
    3. 默认：使用基础阈值配置
    
    Args:
        request_id: 请求唯一标识
        question_id: 问题ID，用于关联profiling数据
        prompt: 请求提示文本
        input_tokens: 输入token数量
        thresholds: 输入token分类阈值
        actual_total_tokens: 实际总token数（输入+输出）
        profiling_dir: profiling结果目录路径
        total_thresholds: 总token分类阈值
        **kwargs: 其他参数
    
    Returns:
        SchedulerRequest对象
    """
    # 优先从profiling目录获取实际总token数
    if profiling_dir and question_id:
        actual_total_tokens = _get_actual_total_tokens_from_profiling(profiling_dir, question_id)
        if actual_total_tokens is not None:
            logger.debug(f"Loaded actual_total_tokens={actual_total_tokens} for {question_id} from profiling")
        else:
            logger.debug(f"Failed to load actual_total_tokens for {question_id} from profiling")
    
    # 初始化分类变量
    category = None
    classification_method = "default"
    
    # 方法1：使用实际总token数进行分类（最高优先级）
    if actual_total_tokens is not None and actual_total_tokens > 0:
        # 使用配置的总token阈值或默认值
        effective_total_thresholds = total_thresholds or {
            'short': 6000,      # <= 6000 tokens
            'medium': 12000,    # <= 12000 tokens
            'long': 18000,      # <= 18000 tokens
            'extra_long': float('inf')  # > 18000 tokens
        }
        category = categorize_sequence(actual_total_tokens, effective_total_thresholds)
        classification_method = "actual_total_tokens"
        logger.debug(f"Using actual total tokens {actual_total_tokens} for {question_id}, categorized as {category}")
    
    # 方法2：使用输入token数进行分类（次优选择）
    elif input_tokens > 0:
        # 使用配置的输入token阈值
        effective_input_thresholds = thresholds or {
            'short': 256,
            'medium': 512, 
            'long': 1024
        }
        category = categorize_sequence(input_tokens, effective_input_thresholds)
        classification_method = "input_tokens"
        logger.debug(f"Using input tokens {input_tokens} for {question_id}, categorized as {category}")
    
    # 方法3：默认分类
    else:
        category = "medium"  # 默认中等长度
        classification_method = "default"
        logger.debug(f"Using default classification for {question_id}: {category}")
    
    # 创建调度请求
    request = SchedulerRequest(
        request_id=request_id, 
        question_id=question_id, 
        prompt=prompt,
        input_tokens=input_tokens, 
        sequence_category=category, 
        actual_total_tokens=actual_total_tokens, 
        **kwargs
    )
    
    # 记录详细的分类信息到日志
    classification_info = {
        "request_id": request_id,
        "question_id": question_id,
        "input_tokens": input_tokens,
        "actual_total_tokens": actual_total_tokens,
        "sequence_category": category,
        "classification_method": classification_method,
        "has_profiling_data": profiling_dir is not None
    }
    logger.info(f"Sequence Classification Result: {classification_info}")
    
    return request
    
    return request
