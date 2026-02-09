"""
Homogeneous Scheduler Module

Scheduler for homogeneous configurations where all instances
have the same TP degree. Uses load balancing for request distribution.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from .base_scheduler import (
    BaseScheduler,
    SchedulerRequest,
    SchedulingResult,
    SchedulingStrategy,
    LoadBalancer
)

logger = logging.getLogger(__name__)


class HomogeneousScheduler(BaseScheduler):
    """
    Scheduler for homogeneous TP configurations.
    
    All instances have the same TP degree, so scheduling is based
    purely on load balancing without considering sequence length.
    
    Supports multiple load balancing strategies:
    - Round Robin: Simple rotation through instances
    - Least Connections: Route to instance with fewest active requests
    - Weighted: Consider instance capacity
    """
    
    def __init__(
        self,
        tp_degree: int,
        strategy: SchedulingStrategy = SchedulingStrategy.LEAST_CONNECTIONS
    ):
        """
        Initialize the homogeneous scheduler.
        
        Args:
            tp_degree: TP degree of all instances
            strategy: Load balancing strategy
        """
        super().__init__(name=f"homogeneous_tp{tp_degree}")
        self.tp_degree = tp_degree
        self.strategy = strategy
        self._load_balancer = LoadBalancer(strategy)
        
        logger.info(f"Initialized HomogeneousScheduler: TP={tp_degree}, strategy={strategy.value}")
    
    async def route_request(
        self,
        request: SchedulerRequest,
        available_instances: List[Any]
    ) -> SchedulingResult:
        """
        Route a request using load balancing.
        
        Args:
            request: Request to schedule
            available_instances: List of available vLLM instances
            
        Returns:
            SchedulingResult
        """
        start_time = time.time()
        
        # Record category for statistics
        self.record_category(request.sequence_category)
        
        # Filter instances by TP degree
        matching_instances = [
            inst for inst in available_instances
            if inst.tp_degree == self.tp_degree and inst.is_ready
        ]
        
        if not matching_instances:
            result = SchedulingResult(
                request_id=request.request_id,
                instance_id="",
                tp_degree=self.tp_degree,
                success=False,
                queue_time=time.time() - request.created_at,
                reason=f"No available instances with TP={self.tp_degree}"
            )
            self.record_result(result)
            
            # 记录同构调度失败信息
            failure_info = {
                "request_id": request.request_id,
                "question_id": request.question_id,
                "sequence_category": request.sequence_category,
                "target_tp_degree": self.tp_degree,
                "reason": f"No available instances with TP={self.tp_degree}",
                "available_instances_count": len(available_instances)
            }
            logger.warning(f"Homogeneous Routing Failed: {failure_info}")
            
            return result
        
        # Select instance using load balancer
        selected = self._load_balancer.select_instance(matching_instances)
        
        if not selected:
            result = SchedulingResult(
                request_id=request.request_id,
                instance_id="",
                tp_degree=self.tp_degree,
                success=False,
                queue_time=time.time() - request.created_at,
                reason="Load balancer failed to select instance"
            )
            self.record_result(result)
            return result
        
        # Update request metadata
        request.scheduled_at = time.time()
        request.target_instance_id = selected.instance_id
        request.target_tp_degree = selected.tp_degree
        
        result = SchedulingResult(
            request_id=request.request_id,
            instance_id=selected.instance_id,
            tp_degree=selected.tp_degree,
            success=True,
            queue_time=request.get_queue_time(),
            reason=f"Selected via {self.strategy.value}"
        )
        
        self.record_result(result)
        
        # 记录详细的同构调度结果到日志
        routing_info = {
            "request_id": request.request_id,
            "question_id": request.question_id,
            "sequence_category": request.sequence_category,
            "input_tokens": request.input_tokens,
            "actual_total_tokens": request.actual_total_tokens,
            "target_instance_id": selected.instance_id,
            "target_tp_degree": selected.tp_degree,
            "load_balance_strategy": self.strategy.value,
            "queue_time_ms": round(request.get_queue_time() * 1000, 2)
        }
        logger.info(f"Homogeneous Routing Result: {routing_info}")
        
        return result
    
    def get_preferred_tp_degrees(self, request: SchedulerRequest) -> List[int]:
        """
        Get preferred TP degrees (only the configured TP degree).
        
        Args:
            request: The request (unused in homogeneous scheduler)
            
        Returns:
            List containing only the configured TP degree
        """
        return [self.tp_degree]
    
    def set_strategy(self, strategy: SchedulingStrategy) -> None:
        """
        Change the load balancing strategy.
        
        Args:
            strategy: New strategy
        """
        self.strategy = strategy
        self._load_balancer = LoadBalancer(strategy)
        logger.info(f"Changed strategy to {strategy.value}")


class MultiTPHomogeneousScheduler(BaseScheduler):
    """
    Scheduler that manages multiple homogeneous schedulers for different TP degrees.
    
    This is useful for testing multiple homogeneous configurations
    in sequence while maintaining separate statistics.
    """
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.LEAST_CONNECTIONS
    ):
        """
        Initialize the multi-TP scheduler.
        
        Args:
            strategy: Load balancing strategy
        """
        super().__init__(name="multi_homogeneous")
        self.strategy = strategy
        self._schedulers: Dict[int, HomogeneousScheduler] = {}
        self._active_tp: Optional[int] = None
    
    def add_tp_scheduler(self, tp_degree: int) -> HomogeneousScheduler:
        """
        Add a scheduler for a specific TP degree.
        
        Args:
            tp_degree: TP degree for the new scheduler
            
        Returns:
            The new HomogeneousScheduler
        """
        if tp_degree not in self._schedulers:
            self._schedulers[tp_degree] = HomogeneousScheduler(tp_degree, self.strategy)
        return self._schedulers[tp_degree]
    
    def set_active_tp(self, tp_degree: int) -> None:
        """
        Set the active TP degree for routing.
        
        Args:
            tp_degree: TP degree to activate
        """
        if tp_degree not in self._schedulers:
            self.add_tp_scheduler(tp_degree)
        self._active_tp = tp_degree
        logger.info(f"Active TP set to {tp_degree}")
    
    async def route_request(
        self,
        request: SchedulerRequest,
        available_instances: List[Any]
    ) -> SchedulingResult:
        """
        Route request using the active TP scheduler.
        
        Args:
            request: Request to schedule
            available_instances: List of available instances
            
        Returns:
            SchedulingResult
        """
        if self._active_tp is None:
            # Auto-detect from available instances
            tp_degrees = set(inst.tp_degree for inst in available_instances if inst.is_ready)
            if tp_degrees:
                self._active_tp = min(tp_degrees)
                self.add_tp_scheduler(self._active_tp)
            else:
                return SchedulingResult(
                    request_id=request.request_id,
                    instance_id="",
                    tp_degree=0,
                    success=False,
                    queue_time=time.time() - request.created_at,
                    reason="No active TP degree and no instances available"
                )
        
        scheduler = self._schedulers.get(self._active_tp)
        if not scheduler:
            scheduler = self.add_tp_scheduler(self._active_tp)
        
        return await scheduler.route_request(request, available_instances)
    
    def get_preferred_tp_degrees(self, request: SchedulerRequest) -> List[int]:
        """Get preferred TP degrees."""
        if self._active_tp is not None:
            return [self._active_tp]
        return list(self._schedulers.keys())
    
    def get_scheduler(self, tp_degree: int) -> Optional[HomogeneousScheduler]:
        """Get scheduler for a specific TP degree."""
        return self._schedulers.get(tp_degree)
    
    def get_all_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all TP schedulers."""
        return {
            tp: scheduler.get_statistics().to_dict()
            for tp, scheduler in self._schedulers.items()
        }
    
    def reset_all_statistics(self) -> None:
        """Reset statistics for all schedulers."""
        for scheduler in self._schedulers.values():
            scheduler.reset_statistics()
        self.reset_statistics()


def create_homogeneous_scheduler(
    config: Dict[str, Any],
    tp_degree: int
) -> HomogeneousScheduler:
    """
    Create a homogeneous scheduler from configuration.
    
    Args:
        config: Configuration dictionary
        tp_degree: TP degree for the scheduler
        
    Returns:
        Configured HomogeneousScheduler
    """
    scheduling_config = config.get("scheduling", {})
    strategy_name = scheduling_config.get("load_balance_strategy", "least_connections")
    
    strategy_map = {
        "round_robin": SchedulingStrategy.ROUND_ROBIN,
        "least_connections": SchedulingStrategy.LEAST_CONNECTIONS,
        "weighted": SchedulingStrategy.WEIGHTED
    }
    
    strategy = strategy_map.get(strategy_name, SchedulingStrategy.LEAST_CONNECTIONS)
    
    return HomogeneousScheduler(tp_degree, strategy)
