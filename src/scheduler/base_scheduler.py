"""
Base Scheduler Module

Defines the abstract base class for request schedulers and common data structures.
"""

import time
import logging
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
                            **kwargs) -> SchedulerRequest:
    category = categorize_sequence(input_tokens, thresholds)
    return SchedulerRequest(
        request_id=request_id, question_id=question_id, prompt=prompt,
        input_tokens=input_tokens, sequence_category=category, **kwargs
    )
